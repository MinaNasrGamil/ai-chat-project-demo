# app_ui.py
import os
import time
import glob
import shutil
import re
from typing import List, Tuple, Optional

import streamlit as st
from google import genai

# must return: (context: str, routed: List[Tuple[str, float]], err: Optional[str])
from app import ask

# ---------- Constants ----------
DATA_DIR = "data"
STORE_DIR = "store"
STORE_PER_DOC = "store_per_doc"
MAX_UPLOAD_MB = 50

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STORE_DIR, exist_ok=True)
os.makedirs(STORE_PER_DOC, exist_ok=True)

FALLBACK_MODELS = [
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemma-3-27b-it",
    "gemini-3.1-flash-lite-preview",
]

# ---------- Utilities ----------


def list_pdfs() -> List[str]:
    return sorted([os.path.basename(p) for p in glob.glob(os.path.join(DATA_DIR, "*.pdf"))])


def _safe_filename(name: str) -> str:
    base = os.path.basename(name)
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_")
    if not base.lower().endswith(".pdf"):
        base += ".pdf"
    return base or f"upload_{int(time.time())}.pdf"


def save_uploaded_pdf(uploaded_file) -> str:
    if getattr(uploaded_file, "size", None) is not None and uploaded_file.size > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"File too large (> {MAX_UPLOAD_MB} MB).")
    safe_name = _safe_filename(uploaded_file.name)
    dst = os.path.join(DATA_DIR, safe_name)
    with open(dst, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dst


def _should_retry(msg: str) -> bool:
    m = msg.upper()
    return any(x in m for x in ["RESOURCE_EXHAUSTED", "429", "QUOTA", "NOT_FOUND", "UNAVAILABLE", "DEADLINE_EXCEEDED"])


def call_gemini(context: str, question: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "[GEMINI] GEMINI_API_KEY is not set."
    client = genai.Client(api_key=api_key)

    prompt = f"""
You are an AI assistant that answers questions ONLY based on the provided context.
If the answer is not found in the context, say:
"This information is not available in the provided documents."

Provide a concise answer and quote the supporting line.

Question:
{question}

Context:
{context}

Answer:
""".strip()

    last_err = None
    for model in FALLBACK_MODELS:
        try:
            resp = client.models.generate_content(model=model, contents=prompt)
            text = getattr(resp, "text", "") or ""
            return text.strip() or "(Empty response)"
        except Exception as e:
            last_err = str(e)
            if _should_retry(last_err):
                continue
            return f"[GEMINI:{model}] Error: {last_err}"

    if last_err and ("quota" in last_err.lower() or "RESOURCE_EXHAUSTED" in last_err.upper()):
        return "[LLM] All free-tier quotas exhausted. Try later."
    return f"[LLM] Error: {last_err or 'Unknown error while calling Gemini.'}"


# ---------- Page Config + Branding ----------
st.set_page_config(
    page_title="BUE — Media & Communication AI Assistant", page_icon="🎓", layout="wide")


def bue_header():
    left, mid, right = st.columns([1, 4, 1])
    with left:
        # Replace with local path if needed: st.image("assets/bue_logo.png", width=90)
        st.image(
            "https://www.bue.edu.eg/wp-content/uploads/2020/01/BUE_Logo.png", width=90)
    with mid:
        st.markdown(
            "<h2 style='margin:0;'>Faculty of Media & Communication — AI Assistant</h2>"
            "<div style='color:#666;'>Ask about official faculty documents and guidelines</div>",
            unsafe_allow_html=True,
        )
    with right:
        if "mode" not in st.session_state:
            st.session_state.mode = "student"
        toggled = st.toggle("Admin mode", value=(
            st.session_state.mode == "admin"))
        st.session_state.mode = "admin" if toggled else "student"


bue_header()
st.divider()

# ---------- Student View ----------


def render_student_view():
    st.markdown("### Ask your question")
    q = st.text_input("Type your question here", value="", max_chars=500,
                      placeholder="e.g., What are the final project submission deadlines?")

    ask_col, _ = st.columns([1, 3])
    with ask_col:
        run = st.button("Ask", type="primary")

    st.divider()

    if run:
        if not list_pdfs():
            st.warning("No documents available yet. Please try again later.")
            return
        if not q.strip():
            st.warning("Please enter your question.")
            return

        with st.spinner("Searching approved documents..."):
            context, routed, err = ask(
                query=q,
                top_m=2,            # fixed for students
                k_per_doc=3,
                top_k_global=5,
                min_route_score=0.08,
                force_router=False,
                force_docs=False,
            )

        if err:
            st.error(err)
            return

        with st.spinner("Generating answer..."):
            answer = call_gemini(context or "", q)

        st.markdown("### Answer")
        st.markdown(answer)
        st.caption("This answer is based on approved faculty documents.")

# ---------- Admin View ----------


def render_admin_view():
    st.markdown("### Admin Panel")
    with st.sidebar:
        st.header("Data & Controls")

        st.write("**PDFs in `data/`:**")
        pdfs = list_pdfs()
        if pdfs:
            st.code("\n".join(pdfs), language="text")
        else:
            st.info("No PDFs found. Upload below.")

        uploaded = st.file_uploader("Upload PDF(s)", type=[
                                    "pdf"], accept_multiple_files=True)
        if uploaded:
            saved = 0
            for uf in uploaded:
                try:
                    save_uploaded_pdf(uf)
                    saved += 1
                except Exception as e:
                    st.error(f"Failed to save {uf.name}: {e}")
            if saved:
                st.success(
                    f"Uploaded {saved} file(s). Consider forcing rebuild.")
            if st.button("Refresh list"):
                st.rerun()

        st.divider()
        st.subheader("Routing & Retrieval Params")
        top_m = st.number_input(
            "Top-M documents (router)", min_value=1, max_value=10, value=2, step=1, key="top_m")
        k_per_doc = st.number_input(
            "Top-k per document", min_value=1, max_value=10, value=3, step=1, key="k_per_doc")
        top_k_global = st.number_input(
            "Global Top-K (merged)", min_value=1, max_value=20, value=5, step=1, key="top_k_global")
        min_route_score = st.slider(
            "Min route score", 0.0, 0.5, 0.08, 0.01, key="min_route_score")

        st.divider()
        st.subheader("Rebuild Options")
        force_router = st.checkbox(
            "Force router rebuild", value=False, key="force_router")
        force_docs = st.checkbox(
            "Force per-doc rebuild", value=False, key="force_docs")
        if st.button("Clear caches (router + per-doc)"):
            try:
                if os.path.isdir(STORE_DIR):
                    shutil.rmtree(STORE_DIR)
                if os.path.isdir(STORE_PER_DOC):
                    shutil.rmtree(STORE_PER_DOC)
                os.makedirs(STORE_DIR, exist_ok=True)
                os.makedirs(STORE_PER_DOC, exist_ok=True)
                st.success("Cleared caches.")
            except Exception as e:
                st.error(f"Failed to clear caches: {e}")

    st.write("### Test a question")
    default_q = "What discount does RICRAC offer?"
    query = st.text_input("Question", value=default_q, max_chars=500)

    cols = st.columns([1, 1])
    with cols[0]:
        run_btn = st.button("Run Retrieval + Answer (Admin)", type="primary")
    with cols[1]:
        show_context = st.checkbox("Show merged context", value=True)

    st.divider()

    if run_btn:
        if not list_pdfs():
            st.warning("No PDFs in `data/`. Please upload at least one PDF.")
            return
        if not query.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Routing documents and retrieving chunks..."):
            start = time.time()
            context, routed, err = ask(
                query=query,
                top_m=int(st.session_state.get("top_m", 2)),
                k_per_doc=int(st.session_state.get("k_per_doc", 3)),
                top_k_global=int(st.session_state.get("top_k_global", 5)),
                min_route_score=float(
                    st.session_state.get("min_route_score", 0.08)),
                force_router=bool(st.session_state.get("force_router", False)),
                force_docs=bool(st.session_state.get("force_docs", False)),
            )
            took_retrieval = time.time() - start

        if err:
            st.error(err)
            return

        st.success(f"Retrieval done in {took_retrieval:.2f}s")

        st.subheader("Routed Documents")
        if routed:
            for doc, score in routed:
                try:
                    st.write(f"- **{doc}** — score: `{float(score):.4f}`")
                except Exception:
                    st.write(f"- **{doc}** — score: `{score}`")
        else:
            st.write("- None above threshold")

        if show_context and context:
            with st.expander("Merged Context (clean)", expanded=False):
                st.markdown(context)

        with st.spinner("Generating answer with Gemini..."):
            start2 = time.time()
            answer = call_gemini(context or "", query)
            took_llm = time.time() - start2

        st.subheader("Answer")
        st.markdown(answer)
        st.caption(f"LLM time: {took_llm:.2f}s")


# ---------- Router ----------
mode = st.session_state.get("mode", "student")
if mode == "admin":
    render_admin_view()
else:
    render_student_view()
