# Multi-Agent Research & Report System

A multi-agent AI research system that transforms a natural-language question into a structured, evidence-backed research report.

The system uses **LangGraph** to orchestrate five specialized stages: research, analysis, additional evidence gathering, report generation, and quality/grounding verification.

**Author:** Sohel Bhongade — B.Tech, IIT Indore

---

## Overview

Traditional LLM-based research often relies on a single model call to search for information and generate an answer. This can make it difficult to separate research from analysis, verify claims, and evaluate the quality of retrieved sources.

This project explores a structured alternative:

> **Decompose research into specialized agents, then verify the generated report against the collected evidence.**

The pipeline performs web research, independently analyzes the evidence, performs additional targeted searching, generates the final report, and evaluates its claims and sources.

---

## Architecture

```text
                         User Query
                              │
                              ▼
                    ┌───────────────────┐
                    │  Research Agent   │
                    │  Web Search       │
                    └─────────┬─────────┘
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
        ┌─────────────────┐       ┌─────────────────┐
        │ Analyst Agent   │       │ Extra Search    │
        │ Evidence        │       │ Agent           │
        │ Analysis        │       │ More Evidence   │
        └────────┬────────┘       └────────┬────────┘
                 │                         │
                 └────────────┬────────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │   Writer Agent    │
                    │ Structured Report │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │     QA Agent      │
                    │ Report Validation │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Grounding /       │
                    │ Source Evaluation │
                    └─────────┬─────────┘
                              │
                              ▼
                       Final Report
```

### Agent responsibilities

| Stage | Responsibility |
|---|---|
| **Research Agent** | Performs initial web research and collects relevant sources |
| **Analyst Agent** | Analyzes and synthesizes the retrieved evidence |
| **Extra Search Agent** | Performs additional targeted searches to fill evidence gaps |
| **Writer Agent** | Converts the research into a structured report |
| **QA Agent** | Reviews the generated report for quality and consistency |
| **Grounding** | Checks report claims against the available evidence |
| **Source Evaluation** | Calculates source-quality information for the research set |

The Analyst and Extra Search stages run in parallel to avoid unnecessary sequential execution.

---

## Key Features

### 🔎 Multi-Agent Research

Instead of relying on one LLM call, the system separates research responsibilities across specialized stages.

### 🌐 Live Web Search

Research is grounded in information retrieved from the web using **Tavily**.

### 🧠 Evidence Analysis

Retrieved sources are analyzed separately from report generation, reducing the dependence on a single generation step.

### 🔄 Parallel Processing

The Analyst and Extra Search agents execute in parallel within the LangGraph workflow.

### ✍️ Structured Report Generation

The Writer Agent converts the collected evidence into a readable research report.

### ✅ Automated QA

A dedicated QA stage reviews the generated report before completion.

### 🎯 Claim-Level Grounding

The system extracts report claims and evaluates whether they are:

- Supported
- Partially supported
- Unsupported

### 📚 Source Quality Evaluation

Retrieved sources are evaluated and assigned source-quality scores.

### 💾 Research History

Research sessions can be stored and revisited through the application's session/history functionality.

### 📄 Export

Generated reports can be exported in Markdown format.

---

## Technology Stack

| Component | Technology |
|---|---|
| Agent orchestration | LangGraph |
| LLM inference | Groq |
| Model | OpenAI GPT-OSS-120B |
| Web search | Tavily |
| Frontend | Streamlit |
| Storage | SQLite |
| Deployment | Hugging Face Spaces |
| Language | Python |

---

# Evaluation

The system was evaluated against a **single-agent baseline** using the same six research questions for both pipelines.

### Benchmark design

- **6 paired real-world queries**
- **1 trial per query**
- Same query presented to both pipelines
- 5 LLM calls per multi-agent run
- 1 LLM call per single-agent run
- Rate-limit failures were excluded from paired performance calculations
- Results are **directional**, not statistically significant

### Benchmark results

| Metric | Multi-Agent | Single-Agent |
|---|---:|---:|
| Paired success rate | **6/6 (100%)** | **6/6 (100%)** |
| Average total tokens | **15,905** | **5,272** |
| Average cost / run | **$0.0045** | **$0.0013** |
| LLM calls / run | **5** | **1** |
| Average report length | **3,893 chars** | **3,954 chars** |
| Average latency | **98.98s** | **27.09s** |

The multi-agent pipeline therefore used approximately:

- **3.0× the tokens**
- **3.6× the cost**

while producing reports of approximately the same length.

This demonstrates the computational cost of the multi-stage architecture rather than claiming that multi-agent processing is universally superior.

---

## Grounding & Source Evaluation

Across the six successful multi-agent runs:

### Claim-level verification

**181 total claims**

- **149 supported — 82.3%**
- **26 partially supported — 14.4%**
- **6 unsupported — 3.3%**

### Grounding

**Average grounding score: 89.2 / 100**

Range across queries:

**85.7 – 93.1**

### Source quality

**Average source-quality score: 0.79 / 1.00**

Range:

**0.759 – 0.861**

These metrics describe the behavior of the project's internal grounding and source-evaluation pipeline. They should **not** be interpreted as externally validated factual accuracy.

---

## What the Evaluation Shows

The benchmark demonstrates that the multi-agent architecture:

1. Successfully completes end-to-end research workflows across multiple topics.
2. Adds explicit claim-level grounding and source-quality evaluation.
3. Produces an auditable breakdown of supported, partially supported, and unsupported claims.
4. Introduces a measurable computational trade-off compared with a single-call baseline.
5. Maintains approximately equivalent final report length despite the additional processing stages.

The experiment does **not** establish that the multi-agent system is universally more accurate or more useful than a single-agent system.

The grounding score is internally computed and was not validated against an independent human fact-checking benchmark.

---

## Example Research Topics

The benchmark included:

- Latest developments in quantum computing
- Impact of remote work on employee productivity
- How CRISPR gene editing works
- Comparison of solar vs. wind renewable energy
- Effects of social media on teenage mental health
- Current state of autonomous vehicle technology

---

## Project Structure

```text
multi_agent_research/
│
├── app.py
├── graph.py
├── agents.py
├── grounding.py
├── single_agent_baseline.py
├── benchmark.py
├── eval_full_capture.py
├── benchmark_queries.py
├── tools.py
│
├── eval_results/
│   └── *.jsonl
│
├── database/
│   └── ...
│
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd multi_agent_research
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### 5. Run the application

```bash
streamlit run app.py
```

---

## Benchmarking

The repository includes the evaluation infrastructure used to compare the multi-agent system against the single-agent baseline.

Example:

```powershell
python eval_full_capture.py `
  --query-indices 0,1,2,3,4,5 `
  --trials 1 `
  --pipeline multi `
  --token-budget 90000
```

Results are written incrementally to:

```text
eval_results/
```

The incremental JSONL format ensures completed evaluations remain available even if a later run encounters an API or rate-limit failure.

---

## Limitations

This is an engineering evaluation rather than a statistically rigorous research benchmark.

### Small benchmark

Only six queries were evaluated, so the results should be considered directional.

### Single trial

Each query was executed once per pipeline. Variance across repeated runs was therefore not measured.

### Grounding validation

The grounding score is produced by the project's own verification logic and has not been independently validated against human annotations.

### Latency variability

API rate limits and external service conditions can affect latency. Therefore, latency should not be treated as a clean architectural measurement.

### No ablation study

The benchmark does not isolate the individual contribution of every stage. In particular, it does not independently establish how much Stage 3/4 improves grounding compared with a pipeline without those stages.

---

## Future Work

- Run repeated trials to measure variance and confidence intervals.
- Add a human-annotated factuality benchmark.
- Perform ablation studies on individual agents.
- Compare different LLM models and configurations.
- Improve handling of API rate limits and retries.
- Evaluate report quality using human or task-specific scoring.
- Explore adaptive agent routing to reduce unnecessary LLM calls.
- Investigate whether all five agent stages are required for different query types.

---

## Deployment

The application is deployed as a Streamlit application on Hugging Face Spaces.

**Live Demo:** `<Hugging Face Space URL>`

---

## Author

**Sohel Bhongade**  
B.Tech — IIT Indore

---

## License

See `LICENSE` for details.
