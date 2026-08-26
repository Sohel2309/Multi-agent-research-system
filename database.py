import sqlite3
import os
import json
from datetime import datetime

# Works locally and on Hugging Face Spaces
DB_PATH = os.path.join(os.path.dirname(__file__), "sessions.db")


def init_db():
    """Creates the sessions table if it does not exist, and migrates
    older databases that already have a `sessions` table but lack columns
    added in later stages (Stage 3: grounding_report/grounding_score;
    Stage 4: source_quality_report/avg_source_quality/research_sources).

    CREATE TABLE IF NOT EXISTS is a no-op on an existing table -- it does
    NOT add new columns to it -- so a separate PRAGMA table_info check +
    ALTER TABLE ADD COLUMN step is required for existing sessions.db
    files to gain new columns without losing any saved sessions.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            research_data TEXT,
            analysis TEXT,
            report TEXT,
            qa_review TEXT,
            created_at TEXT NOT NULL,
            grounding_report TEXT DEFAULT '',
            grounding_score REAL,
            source_quality_report TEXT DEFAULT '',
            avg_source_quality REAL,
            research_sources TEXT DEFAULT '[]'
        )
    ''')

    c.execute("PRAGMA table_info(sessions)")
    existing_columns = {row[1] for row in c.fetchall()}
    if "grounding_report" not in existing_columns:
        c.execute("ALTER TABLE sessions ADD COLUMN grounding_report TEXT DEFAULT ''")
    if "grounding_score" not in existing_columns:
        c.execute("ALTER TABLE sessions ADD COLUMN grounding_score REAL")
    if "source_quality_report" not in existing_columns:
        c.execute("ALTER TABLE sessions ADD COLUMN source_quality_report TEXT DEFAULT ''")
    if "avg_source_quality" not in existing_columns:
        c.execute("ALTER TABLE sessions ADD COLUMN avg_source_quality REAL")
    if "research_sources" not in existing_columns:
        c.execute("ALTER TABLE sessions ADD COLUMN research_sources TEXT DEFAULT '[]'")

    conn.commit()
    conn.close()


def save_session(query: str, research_data: str, analysis: str, report: str, qa_review: str,
                  grounding_report: str = "", grounding_score=None,
                  source_quality_report: str = "", avg_source_quality=None,
                  research_sources=None) -> int:
    """Saves a completed research session. Returns the session ID.

    grounding_report/grounding_score (Stage 3) and source_quality_report/
    avg_source_quality/research_sources (Stage 4) all default to empty/
    None so any existing caller that doesn't pass them still works.
    research_sources (a list of dicts) is stored as a JSON string --
    SQLite has no native list/dict column type.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO sessions (
            query, research_data, analysis, report, qa_review, created_at,
            grounding_report, grounding_score,
            source_quality_report, avg_source_quality, research_sources
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        query,
        research_data,
        analysis,
        report,
        qa_review,
        datetime.now().strftime('%Y-%m-%d %H:%M'),
        grounding_report,
        grounding_score,
        source_quality_report,
        avg_source_quality,
        json.dumps(research_sources or []),
    ))
    conn.commit()
    session_id = c.lastrowid
    conn.close()
    return session_id


def get_all_sessions() -> list:
    """Returns all sessions (id, query, created_at) newest first."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, query, created_at FROM sessions ORDER BY created_at DESC')
    rows = c.fetchall()
    conn.close()
    return rows


def get_session_by_id(session_id: int) -> tuple:
    """Returns full session data for a given session ID."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT * FROM sessions WHERE id = ?', (session_id,))
    row = c.fetchone()
    conn.close()
    return row


def delete_session(session_id: int):
    """Deletes a session by ID."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
    conn.commit()
    conn.close()