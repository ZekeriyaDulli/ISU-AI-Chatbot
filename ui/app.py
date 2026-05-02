"""
Streamlit Front-End — Agentic LLM Interface
UI/UX Designer & Front-End Developer: Azaa Almousli (STU ID: 2309115421)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from agents.orchestrator import MultiAgentOrchestrator
from rag.vector_db import VectorDBClient, DocumentIngestionPipeline

st.set_page_config(
    page_title="ISU AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

TEAM = [
    ("Zekeriya Dulli", "2309115377", "Lead Developer & System Architect"),
    ("Obada Abdulhakim Kharaz", "2309115277", "Project Manager & Demo Lead"),
    ("Hamdi ALNAQEEB", "2309116178", "Prompt Engineer & Agent Designer"),
    ("Fares STOUHI", "2309115179", "Data & RAG Pipeline Engineer"),
    ("Azaa Almousli", "2309115421", "UI/UX Designer & Front-End Developer"),
    ("Abdulaziz ALYAHYA", "2309116441", "Risk Management & Safety Monitor"),
    ("Leen Safi", "2309116117", "Evaluation & Testing Lead"),
]


@st.cache_resource(show_spinner="Loading AI system...")
def load_orchestrator():
    return MultiAgentOrchestrator()


@st.cache_resource(show_spinner="Connecting to knowledge base...")
def load_db():
    return VectorDBClient()


def render_sidebar(db: VectorDBClient):
    with st.sidebar:
        st.title("Agentic LLM System")
        st.caption("ISU — Introduction to LLMs | Final Project")
        st.divider()

        st.subheader("Knowledge Base")
        st.metric("Documents in DB", db.count)

        st.subheader("Ingest Document")
        doc_text = st.text_area("Paste document text", height=150, key="ingest_text")
        doc_source = st.text_input("Source label", value="manual_input")
        if st.button("Ingest", use_container_width=True):
            if doc_text.strip():
                pipeline = DocumentIngestionPipeline(db)
                added = pipeline.ingest_text(doc_text.strip(), source=doc_source)
                st.success(f"Ingested {added} chunk(s).")
                st.rerun()
            else:
                st.warning("Please enter some text first.")

        st.divider()
        if st.button("🔄 Reload AI System", use_container_width=True, help="Clears the cached model and reloads with latest settings"):
            st.cache_resource.clear()
            st.rerun()

        st.divider()
        st.subheader("Team")
        for name, sid, role in TEAM:
            st.markdown(f"**{name}**  \n`{sid}` — {role}")


def render_chat(orchestrator: MultiAgentOrchestrator):
    st.header("Ask the Agent System")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                with st.expander("Details"):
                    st.json(msg["meta"])

    if prompt := st.chat_input("Ask anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = orchestrator.run(prompt)

            response = result["response"]
            safety = result["safety"]
            agent_used = result["agent_used"]
            blocked = result.get("blocked", False)

            if blocked:
                st.error(
                    f"🚫 Request blocked (risk score: {safety['risk_score']:.2f}). "
                    "This topic cannot be discussed."
                )
                st.markdown(response)
                st.caption(f"Agent: `safety_monitor` | Safety: 🚫 Blocked")
            else:
                st.markdown(response)
                st.caption(f"Agent: `{agent_used}` | Safety: ✅ Passed")

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "meta": result,
        })


def main():
    orchestrator = load_orchestrator()
    db = load_db()
    render_sidebar(db)
    render_chat(orchestrator)


if __name__ == "__main__":
    main()
