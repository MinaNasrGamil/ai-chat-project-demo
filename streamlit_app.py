import os
import io
import time
import glob
import shutil
import streamlit as st
from typing import List
from app import ask  # uses your existing backend function
from google import genai

DATA_DIR = "data"
STORE_DIR = "store"
STORE_PER_DOC = "store_per_doc"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STORE_DIR, exist_ok=True)
os.makedirs(STORE_PER_DOC, exist_ok=True)


def list_pdfs() -> List[str]:
    return sorted([os.path.basename(p) for p in glob.glob(os.path.join(DATA_DIR, "*.pdf"))])


def save_uploaded_pdf(uploaded_file) -> str:
    dst_path = os.path.join(DATA_DIR, uploaded_file.name)
    with open(dst_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dst_path


def call_gemini(context: str, question: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "[GEMINI] GEMINI_API_KEY is not set. Skipping LLM call."
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
"""
    resp = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt)
    return getattr(resp, "text", "").strip() or "(Empty response)"


def highlight_terms(text: str, terms: List[str]) -> str:
    # Simple highlight (case-insensitive) for query terms
    out = text
    for t in terms:
        if not t.strip():
            continue
        out = out.replace(t, f"**{t}**")
        out = out.replace(t.lower(), f"**{t.lower()}**")
        out = out.replace(t.upper(), f"**{t.upper()}**")
        cap = t[:1].upper() + t[1:].lower()
        out = out.replace(cap, f"**{cap}**")
    return out


# ---------- UI ----------
st.set_page_config(page_title="RAG Demo — Multi‑PDF Router",
                   page_icon="🧠", layout="wide")

st.title("🧠 RAG Demo — Multi‑PDF Router")
st.caption(
    "Multi-document retrieval with routing, TF‑IDF + FAISS, and Gemini answering.")

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
        for uf in uploaded:
            save_uploaded_pdf(uf)
        st.success("Uploaded. Consider forcing rebuild.")
        if st.button("Refresh list"):
            st.rerun()

    st.divider()
    st.subheader("Routing & Retrieval Params")
    top_m = st.number_input("Top-M documents (router)",
                            min_value=1, max_value=10, value=2, step=1)
    k_per_doc = st.number_input(
        "Top-k per document", min_value=1, max_value=10, value=3, step=1)
    top_k_global = st.number_input(
        "Global Top-K (merged)", min_value=1, max_value=20, value=5, step=1)
    min_route_score = st.slider("Min route score", 0.0, 0.5, 0.08, 0.01)

    st.divider()
    st.subheader("Rebuild Options")
    force_router = st.checkbox("Force router rebuild", value=False)
    force_docs = st.checkbox("Force per-doc rebuild", value=False)
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

st.write("### Ask a question")
default_q = "What discount does RICRAC offer?"
query = st.text_input("Question", value=default_q, max_chars=500)

cols = st.columns([1, 1])
with cols[0]:
    run_btn = st.button("Run Retrieval + Answer", type="primary")
with cols[1]:
    show_context = st.checkbox("Show merged context", value=True)

st.divider()

if run_btn:
    if not list_pdfs():
        st.warning("No PDFs in `data/`. Please upload at least one PDF.")
    elif not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Routing documents and retrieving chunks..."):
            start = time.time()
            context, routed, err = ask(
                query=query,
                top_m=int(top_m),
                k_per_doc=int(k_per_doc),
                top_k_global=int(top_k_global),
                min_route_score=float(min_route_score),
                force_router=bool(force_router),
                force_docs=bool(force_docs)
            )
            took_retrieval = time.time() - start

        if err:
            st.error(err)
        else:
            st.success(f"Retrieval done in {took_retrieval:.2f}s")

            st.subheader("Routed Documents")
            if routed:
                for doc, score in routed:
                    st.write(f"- **{doc}** — score: `{score:.4f}`")
            else:
                st.write("- None above threshold")

            if show_context and context:
                with st.expander("Clean Context (merged)", expanded=False):
                    # simple highlighting using query terms split by spaces
                    terms = [t for t in query.split(" ") if len(t.strip()) > 2]
                    st.markdown(highlight_terms(context, terms))

            with st.spinner("Generating answer with Gemini..."):
                start2 = time.time()
                answer = call_gemini(context or "", query)
                took_llm = time.time() - start2

            st.subheader("Answer")
            st.markdown(answer)
            st.caption(f"LLM time: {took_llm:.2f}s")
