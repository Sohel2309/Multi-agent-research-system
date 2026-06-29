import streamlit as st
import os
from dotenv import load_dotenv

# Load .env locally (does nothing on Hugging Face - keys come from HF Secrets)
load_dotenv()

# Must be first Streamlit command
st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="🤖",
    layout="wide"
)

# Import after env vars loaded
from graph import run_research
from database import init_db, save_session, get_all_sessions, get_session_by_id, delete_session

# Initialize database on startup
init_db()


# ─── HEADER ───────────────────────────────────────────────────────────────────

st.title("🤖 Multi-Agent Research & Report System")
st.markdown("*4 AI Agents: Research → Analyze → Write → Fact-Check*")
st.divider()


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ How it works")
    st.markdown("""
**4 AI Agents work in sequence:**

🔍 **Research Agent**
Searches the web, extracts key facts

📊 **Analyst Agent**
Finds patterns and deep insights

✍️ **Writer Agent**
Writes a professional report

✅ **QA Agent**
Fact-checks the report against sources
""")

    st.divider()

    st.header("📋 Example queries")
    examples = [
        "Latest trends in generative AI 2025",
        "Impact of electric vehicles on environment",
        "India startup ecosystem growth 2025",
        "Quantum computing current state",
        "Future of remote work"
    ]
    for example in examples:
        if st.button(example, use_container_width=True):
            st.session_state["query_input"] = example

    st.divider()

    # ── SESSION HISTORY ──────────────────────────────────────────────────────
    st.header("📚 Past Research Sessions")

    sessions = get_all_sessions()

    if not sessions:
        st.caption("No sessions yet. Run a research query to save it here.")
    else:
        for session in sessions:
            session_id, query, created_at = session
            col_a, col_b = st.columns([4, 1])
            with col_a:
                label = f"📄 {query[:28]}...\n{created_at}" if len(query) > 28 else f"📄 {query}\n{created_at}"
                if st.button(label, key=f"load_{session_id}", use_container_width=True):
                    st.session_state["loaded_session_id"] = session_id
            with col_b:
                if st.button("🗑", key=f"del_{session_id}"):
                    delete_session(session_id)
                    if st.session_state.get("loaded_session_id") == session_id:
                        st.session_state.pop("loaded_session_id", None)
                    st.rerun()

    st.divider()
    st.caption("Free tier: Groq (rate limited) · Tavily (1000 searches/month)")


# ─── LOAD A PAST SESSION ──────────────────────────────────────────────────────

if "loaded_session_id" in st.session_state:
    row = get_session_by_id(st.session_state["loaded_session_id"])
    if row:
        st.info(f"📂 Viewing past session: **{row[1]}** — {row[6]}")
        tab1, tab2, tab3, tab4 = st.tabs(["📄 Final Report", "🔍 Research Data", "📊 Analysis", "✅ QA Review"])

        with tab1:
            st.markdown(row[4])
            st.download_button(
                label="⬇️ Download Report",
                data=row[4],
                file_name=f"report_{row[1][:30].replace(' ', '_')}.md",
                mime="text/markdown"
            )
        with tab2:
            st.markdown(row[2])
        with tab3:
            st.markdown(row[3])
        with tab4:
            qa = row[5]
            if "PASS WITH WARNINGS" in qa:
               st.warning(qa)
            elif "FAIL" in qa:
               st.error(qa)
            else:
               st.success(qa)
        
        if st.button("✖️ Close this session"):
            st.session_state.pop("loaded_session_id", None)
            st.rerun()

        st.divider()


# ─── MAIN INPUT ───────────────────────────────────────────────────────────────

query = st.text_input(
    "Enter your research topic or question:",
    placeholder="e.g. Latest trends in large language models",
    key="query_input"
)

col1, col2 = st.columns([1, 5])
with col1:
    run_button = st.button("🚀 Run Research", type="primary", use_container_width=True)
with col2:
    st.caption("Takes 45–90 seconds. Four agents work in sequence.")


# ─── RUN PIPELINE ─────────────────────────────────────────────────────────────

if run_button:
    if not query.strip():
        st.warning("Please enter a research topic first.")
    elif not os.environ.get("GROQ_API_KEY"):
        st.error("GROQ_API_KEY not found. Add it in HF Spaces Secrets or your local .env file.")
    elif not os.environ.get("TAVILY_API_KEY"):
        st.error("TAVILY_API_KEY not found. Add it in HF Spaces Secrets or your local .env file.")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("🔍 Research Agent searching the web...")
            progress_bar.progress(10)

            with st.spinner("4 agents working... (45–90 seconds)"):
                result = run_research(query)

            progress_bar.progress(90)
            status_text.text("💾 Saving session...")

            session_id = save_session(
                query=result["query"],
                research_data=result.get("research_data", ""),
                analysis=result.get("analysis", ""),
                report=result.get("report", ""),
                qa_review=result.get("qa_review", "")
            )

            progress_bar.progress(100)
            status_text.text("✅ All agents completed! Session saved.")

            st.success(f"Research complete! Session #{session_id} saved to history.")
            st.divider()

            with st.expander("📋 Agent Activity Log", expanded=False):
                for msg in result.get("messages", []):
                    st.markdown(f"- {msg}")

            tab1, tab2, tab3, tab4 = st.tabs(["📄 Final Report", "🔍 Research Data", "📊 Analysis", "✅ QA Review"])

            with tab1:
                st.markdown("### 📄 Final Report")
                st.markdown(result.get("report", "No report generated."))
                st.download_button(
                    label="⬇️ Download Report as .md",
                    data=result.get("report", ""),
                    file_name=f"report_{query[:30].replace(' ', '_')}.md",
                    mime="text/markdown"
                )

            with tab2:
                st.markdown("### 🔍 Raw Research Data")
                st.markdown(result.get("research_data", "No research data."))

            with tab3:
                st.markdown("### 📊 Analysis")
                st.markdown(result.get("analysis", "No analysis."))

            with tab4:
                st.markdown("### ✅ QA Fact-Check Review")
                qa = result.get("qa_review", "No QA review.")
                if "PASS WITH WARNINGS" in qa:
                    st.warning(qa)
                elif "FAIL" in qa:
                    st.error(qa)
                else:
                    st.success(qa)

        except Exception as e:
            progress_bar.progress(0)
            status_text.text("❌ An error occurred.")
            st.error(f"Error: {str(e)}")
            st.info("Common fixes:\n- Check your API keys\n- Wait 30 seconds if you hit a rate limit, then try again")


# ─── FOOTER ───────────────────────────────────────────────────────────────────

st.divider()
st.caption("Built with LangGraph · LangChain · Groq (Llama 3.3 70B) · Tavily Search · Streamlit")