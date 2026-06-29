import sqlite3
import os
from datetime import datetime

# Works locally and on Hugging Face Spaces
DB_PATH = os.path.join(os.path.dirname(__file__), "sessions.db")


def init_db():
    """Creates the sessions table if it does not exist."""
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
            created_at TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


def save_session(query: str, research_data: str, analysis: str, report: str, qa_review: str) -> int:
    """Saves a completed research session. Returns the session ID."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO sessions (query, research_data, analysis, report, qa_review, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        query,
        research_data,
        analysis,
        report,
        qa_review,
        datetime.now().strftime('%Y-%m-%d %H:%M')
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