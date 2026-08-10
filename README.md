# 🤖 Multi-Agent Research & Report System

A production-grade AI research platform where **5 specialized agents** collaborate to research any topic, generate a structured professional report, and automatically fact-check the output — all in **avg 27 seconds**.

**Live Demo →** [huggingface.co/spaces/Sohel2309/multi-agent-research](https://huggingface.co/spaces/Sohel2309/multi-agent-research)

---

## The Problem

Researching any topic properly takes hours — searching multiple sources, reading articles, extracting key facts, analyzing patterns, writing a structured summary, and then verifying the information is actually accurate.

Most people either skip steps (shallow research) or spend 4–6 hours doing it manually.

This system automates the entire pipeline in **under 90 seconds** using 5 specialized AI agents — each handling one stage of the research process, with the final agent fact-checking everything before you see it.

---

## How It Works

```
User Query
    ↓
🔍 Research Agent              →  Searches live web, extracts key facts
    ↓              ↘
📊 Analyst Agent    🔎 Extra Search Agent    ← run in PARALLEL (asyncio.gather)
    ↓              ↙
✍️  Writer Agent               →  Writes structured professional report
    ↓
✅ QA Agent                    →  Fact-checks report against source data
    ↓
Final Report + Verdict (PASS / PASS WITH WARNINGS / FAIL)
```

After the Research Agent finishes, the Analyst Agent and Extra Search Agent run **simultaneously** via `asyncio.gather` — one analyzes the findings while the other gathers additional statistics. The Writer Agent waits for both, then produces a richer report using all gathered context.

Each agent is powered by **Llama 3.3 70B via Groq** and orchestrated using **LangGraph** state machines with conditional routing and shared state management.

---

## Benchmarks

| Metric | Result |
|---|---|
| Avg end-to-end pipeline time | **27.1s** across 4 benchmark queries |
| QA unsupported claims flagged | **avg 3 per report** across 3 diverse topics |
| Parallel execution | Analyst + Extra Search fire concurrently via `asyncio.gather` |
| Topics tested | Quantum Computing · Remote Work · EV in India · Social Media |

---

## Agent Architecture

| Agent | Role | Input | Output |
|---|---|---|---|
| 🔍 Research Agent | Live web search + extraction | User query | Key facts & statistics |
| 📊 Analyst Agent | Pattern analysis (parallel) | Research data | Insights & implications |
| 🔎 Extra Search Agent | Additional data gathering (parallel) | Research data | Extra statistics & examples |
| ✍️ Writer Agent | Report generation | Research + Analysis + Extra context | Structured markdown report |
| ✅ QA Agent | Fact verification | Report + All sources | Verdict + flagged claims |

---

## Features

- **5-agent LangGraph pipeline** with async parallel execution, conditional routing, and shared state
- **Live web search** via Tavily API — real-time data, not stale training knowledge
- **Async parallel execution** — Analyst and Extra Search agents run simultaneously via `asyncio.gather`; benchmarked at **27.1s avg** end-to-end
- **Automated fact-checking** — QA agent cross-verifies every report claim against source data; flags avg **3 unsupported claims per report** with PASS / PASS WITH WARNINGS / FAIL verdict
- **Session history** — every report saved to SQLite, accessible and downloadable from sidebar at any time
- **4-tab result view** — Final Report / Research Data / Analysis / QA Review
- **Markdown download** — export any report as a `.md` file instantly

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Orchestration | LangGraph 1.2.6 |
| Async Parallel Execution | Python asyncio + asyncio.gather |
| LLM | Llama 3.3 70B via Groq API |
| Web Search | Tavily Search API |
| LLM Framework | LangChain 1.3.11 |
| Frontend | Streamlit 1.58.0 |
| Session Storage | SQLite |
| Deployment | Hugging Face Spaces |

---

## Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/Sohel2309/Multi-agent-research-system.git
cd Multi-agent-research-system
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure API keys**

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Both APIs offer free tiers — no credit card needed.

**5. Run the application**
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Project Structure

```
multi-agent-research-system/
├── state.py          # Shared state definition (AgentState TypedDict)
├── tools.py          # Tavily web search wrapper
├── agents.py         # All 5 agent functions with async support
├── graph.py          # LangGraph pipeline with asyncio parallel execution
├── database.py       # SQLite session storage (CRUD)
├── app.py            # Streamlit UI
└── requirements.txt  # Pinned dependencies
```

---

## Author

**Sohel Bhongade**
B.Tech, IIT Indore

[GitHub](https://github.com/Sohel2309) · [Live Demo](https://huggingface.co/spaces/Sohel2309/multi-agent-research)