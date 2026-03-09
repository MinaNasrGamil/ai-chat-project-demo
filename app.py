# app.py
import os
import re
import glob
import html
import pickle
import argparse
import faiss
import numpy as np
from typing import Dict, List, Tuple
from scipy.sparse import issparse
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from google import genai
from router import Router, _clean

STORE_DIR = "store"
STORE_PER_DOC = "store_per_doc"
DATA_DIR = "data"

os.makedirs(STORE_DIR, exist_ok=True)
os.makedirs(STORE_PER_DOC, exist_ok=True)


def to_dense_l2(X):
    X = normalize(X, norm='l2')
    if issparse(X):
        X = X.toarray()
    return X.astype('float32')


def load_and_chunk(pdf_path: str, chunk_size=500, chunk_overlap=80) -> Tuple[List[str], List]:
    docs = PyPDFLoader(pdf_path).load()
    for d in docs:
        d.page_content = _clean(d.page_content)
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n### ", "\n## ", "\n# ", "\n\n", "\n", " "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    return [c.page_content for c in chunks], docs


def build_or_load_doc_index(name: str, chunks: List[str], force: bool = False):
    base = os.path.join(STORE_PER_DOC, os.path.splitext(name)[0])
    pkl_path = base + ".pkl"
    faiss_path = base + ".faiss"

    if (not force) and os.path.exists(pkl_path) and os.path.exists(faiss_path):
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        vectorizer = obj["vectorizer"]
        texts = obj["texts"]
        index = faiss.read_index(faiss_path)
        return vectorizer, index, texts

    vectorizer = TfidfVectorizer(ngram_range=(
        1, 2), stop_words='english', min_df=1)
    tfidf = vectorizer.fit_transform(chunks)
    vecs = to_dense_l2(tfidf)
    d = vecs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(vecs)

    with open(pkl_path, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "texts": chunks}, f)
    faiss.write_index(index, faiss_path)

    return vectorizer, index, chunks


def make_profile(name: str, chunks: List[str], pages, limit: int = 3) -> str:
    first_page = pages[0].page_content if pages else ""
    sample = "\n".join(chunks[:limit])
    profile = f"{name}\n{_clean(first_page)}\n{_clean(sample)}"
    profile = re.sub(r'\s+', ' ', profile).strip()
    return profile[:1200]


def ask(query: str,
        top_m: int = 2,
        k_per_doc: int = 3,
        top_k_global: int = 5,
        min_route_score: float = 0.08,
        force_router: bool = False,
        force_docs: bool = False):

    pdfs = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))
    if not pdfs:
        return None, None, "No PDFs found in ./data"

    profiles: Dict[str, str] = {}
    per_doc: Dict[str, Dict] = {}
    total_chunks = 0

    for pdf in pdfs:
        name = os.path.basename(pdf)
        chunks, pages = load_and_chunk(pdf)
        vectorizer, index, texts = build_or_load_doc_index(
            name, chunks, force=force_docs)
        per_doc[name] = {"vectorizer": vectorizer,
                         "index": index, "texts": texts}
        profiles[name] = make_profile(name, chunks, pages)
        total_chunks += len(chunks)

    print(f"PDFs: {len(pdfs)} | Total Chunks: {total_chunks}")

    router = Router()
    need_rebuild = force_router
    if not need_rebuild:
        try:
            router.load()
            current_docs = set(profiles.keys())
            if set(router.doc_names) != current_docs:
                need_rebuild = True
        except Exception:
            need_rebuild = True

    if need_rebuild:
        print("[Router] Rebuilding router index...")
        router.build(profiles).save()
    else:
        print("[Router] Loaded cached router index.")

    routed = router.route(query, top_m=top_m, min_score=min_route_score)
    print("\n[Routing] Top documents:")
    if routed:
        for doc, sc in routed:
            print(
                f"- {doc} (score={sc:.4f}) | terms: {router.explain_terms(query, doc)}")
    else:
        print("- None above threshold")

    if not routed:
        return None, None, "This information is not available in the provided documents."

    merged = []
    for doc, prior in routed:
        entry = per_doc[doc]
        q = entry["vectorizer"].transform([query])
        q = to_dense_l2(q)
        k = min(k_per_doc, entry["index"].ntotal)
        scores, idxs = entry["index"].search(q, k)
        for sc, i in zip(scores[0], idxs[0]):
            merged.append((entry["texts"][i], float(sc)
                          * (0.7 + 0.3 * prior), doc))

    merged.sort(key=lambda x: x[1], reverse=True)
    picked = merged[:top_k_global]

    def compact(s: str, max_len=800):
        return re.sub(r'\s+', ' ', s).strip()[:max_len]

    context = "\n\n---\n\n".join(compact(t) for (t, _, _) in picked)

    return context, routed, None


def main():
    parser = argparse.ArgumentParser(
        description="Routed multi-document RAG (PDF-only).")
    parser.add_argument("--query", type=str,
                        default="What discount does RICRAC offer?")
    parser.add_argument("--top-m", type=int, default=2)
    parser.add_argument("--k-per-doc", type=int, default=3)
    parser.add_argument("--top-k-global", type=int, default=5)
    parser.add_argument("--min-route-score", type=float, default=0.08)
    parser.add_argument("--force", action="store_true",
                        help="Force rebuild router and per-doc indices.")
    parser.add_argument("--force-router", action="store_true",
                        help="Force rebuild router only.")
    parser.add_argument("--force-docs", action="store_true",
                        help="Force rebuild per-doc indices only.")
    args = parser.parse_args()

    force_router = args.force or args.force_router
    force_docs = args.force or args.force_docs

    context, routed, err = ask(
        query=args.query,
        top_m=args.top_m,
        k_per_doc=args.k_per_doc,
        top_k_global=args.top_k_global,
        min_route_score=args.min_route_score,
        force_router=force_router,
        force_docs=force_docs
    )

    if err:
        print(err)
        return

    print("\nClean Context:\n")
    print(context)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\n[GEMINI] GEMINI_API_KEY is not set. Skipping LLM call.")
        return

    client = genai.Client(api_key=api_key)
    prompt = f"""
You are an AI assistant that answers questions ONLY based on the provided context.
If the answer is not found in the context, say:
"This information is not available in the provided documents."

Provide a concise answer and quote the supporting line.

Question:
{args.query}

Context:
{context}

Answer:
"""
    resp = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt)
    print("\nAI Answer:\n")
    print(resp.text)


if __name__ == "__main__":
    main()
