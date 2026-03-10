
import os
import re
import glob
import argparse
import pickle
import faiss
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
os.makedirs(DATA_DIR, exist_ok=True)


FALLBACK_MODELS_DEFAULT = [
    "gemma-3-27b-it",
    "gemini-3.1-flash-lite-preview",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-flash-lite-latest",
    "gemini-flash-latest"
]

# Available models:
# - aqa
# - deep-research-pro-preview-12-2025
# - gemini-2.0-flash
# - gemini-2.0-flash-001
# - gemini-2.0-flash-lite
# - gemini-2.0-flash-lite-001
# - gemini-2.5-computer-use-preview-10-2025
# - gemini-2.5-flash
# - gemini-2.5-flash-image
# - gemini-2.5-flash-lite
# - gemini-2.5-flash-lite-preview-09-2025
# - gemini-2.5-flash-native-audio-latest
# - gemini-2.5-flash-native-audio-preview-09-2025
# - gemini-2.5-flash-native-audio-preview-12-2025
# - gemini-2.5-flash-preview-tts
# - gemini-2.5-pro
# - gemini-2.5-pro-preview-tts
# - gemini-3-flash-preview
# - gemini-3-pro-image-preview
# - gemini-3-pro-preview
# - gemini-3.1-flash-image-preview
# - gemini-3.1-flash-lite-preview
# - gemini-3.1-pro-preview
# - gemini-3.1-pro-preview-customtools
# - gemini-embedding-001
# - gemini-flash-latest
# - gemini-flash-lite-latest
# - gemini-pro-latest
# - gemini-robotics-er-1.5-preview
# - gemma-3-12b-it
# - gemma-3-1b-it
# - gemma-3-27b-it
# - gemma-3-4b-it
# - gemma-3n-e2b-it
# - gemma-3n-e4b-it
# - imagen-4.0-fast-generate-001
# - imagen-4.0-generate-001
# - imagen-4.0-ultra-generate-001
# - nano-banana-pro-preview
# - veo-2.0-generate-001
# - veo-3.0-fast-generate-001
# - veo-3.0-generate-001
# - veo-3.1-fast-generate-preview
# - veo-3.1-generate-preview


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


def is_short_query(q: str, max_terms: int = 2, max_len: int = 12) -> bool:
    terms = [t for t in re.split(r"\W+", q) if t.strip()]
    return (len(terms) <= max_terms) or (len(q.strip()) <= max_len)


def list_available_models() -> List[str]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return []
    try:
        client = genai.Client(api_key=api_key)
        models = client.models.list()
        names = []
        for m in models:
            # m.name is usually like "models/gemini-2.5-flash"
            name = getattr(m, "name", "")
            if not name:
                continue
            if name.startswith("models/"):
                name = name.split("/", 1)[1]
            names.append(name)
        return sorted(set(names))
    except Exception:
        return []


def resolve_model_chain(user_chain_csv: str) -> List[str]:
    if user_chain_csv:
        user_chain = [m.strip()
                      for m in user_chain_csv.split(",") if m.strip()]
    else:
        user_chain = FALLBACK_MODELS_DEFAULT[:]

    available = set(list_available_models())
    if not available:
        # If we cannot list, return user chain as-is (will try and may fail gracefully)
        return user_chain

    # Keep only models that exist in your project/quota
    filtered = [m for m in user_chain if m in available]
    # If nothing matched, fall back to first available "flash" style models
    if not filtered:
        fallback = [
            m for m in available if "flash" in m and "tts" not in m and "vision" not in m]
        filtered = fallback[:3] if fallback else user_chain
    return filtered


def try_models_with_fallback(prompt: str, chain: List[str]) -> Tuple[str, str]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "[LLM] GEMINI_API_KEY not set.", "N/A"

    client = genai.Client(api_key=api_key)
    last_err = None

    for model in chain:
        try:
            resp = client.models.generate_content(model=model, contents=prompt)
            txt = getattr(resp, "text", "").strip()
            return txt if txt else "(Empty response)", model
        except Exception as e:
            msg = str(e)
            last_err = msg
            if ("RESOURCE_EXHAUSTED" in msg) or ("429" in msg) or ("quota" in msg.lower()) or ("NOT_FOUND" in msg) or ("is not found" in msg.lower()):
                continue
            break

    if last_err:
        if ("RESOURCE_EXHAUSTED" in last_err) or ("429" in last_err) or ("quota" in last_err.lower()):
            return "[LLM] All free-tier quotas exhausted across selected models. Try later or change model.", "exhausted"
        return f"[LLM] Error: {last_err}", "error"

    return "[LLM] Unknown error.", "unknown"


def ask(query: str,
        top_m: int = 2,
        k_per_doc: int = 3,
        top_k_global: int = 5,
        min_route_score: float = 0.01,
        force_router: bool = False,
        force_docs: bool = False):

    pdfs = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))
    if not pdfs:
        return None, None, "No PDFs found in ./data"

    profiles: Dict[str, str] = {}
    per_doc: Dict[str, Dict] = {}

    for pdf in pdfs:
        name = os.path.basename(pdf)
        chunks, pages = load_and_chunk(pdf)
        vect, index, texts = build_or_load_doc_index(
            name, chunks, force=force_docs)
        per_doc[name] = {"vectorizer": vect, "index": index, "texts": texts}
        profiles[name] = make_profile(name, chunks, pages)

    if is_short_query(query):
        merged = []
        for doc_name, entry in per_doc.items():
            qv = entry["vectorizer"].transform([query])
            qv = to_dense_l2(qv)
            k = min(k_per_doc, entry["index"].ntotal)
            scores, idxs = entry["index"].search(qv, k)
            for sc, i in zip(scores[0], idxs[0]):
                merged.append((entry["texts"][i], float(sc), doc_name))
        merged.sort(key=lambda x: x[1], reverse=True)
        picked = merged[:top_k_global]
        context = "\n\n---\n\n".join(re.sub(r'\s+', ' ',
                                     t).strip()[:800] for (t, _, _) in picked)
        routed_docs = [(doc, 0.0) for (_, _, doc) in picked]
        return context, routed_docs, None

    router = Router()
    need_rebuild = force_router
    if not need_rebuild:
        try:
            router.load()
            if set(router.doc_names) != set(profiles.keys()):
                need_rebuild = True
        except:
            need_rebuild = True

    if need_rebuild:
        router.build(profiles).save()

    routed = router.route(query, top_m=top_m,
                          min_score=min_route_score, always_at_least_one=True)

    merged = []
    for doc, prior in routed:
        entry = per_doc[doc]
        qv = entry["vectorizer"].transform([query])
        qv = to_dense_l2(qv)
        k = min(k_per_doc, entry["index"].ntotal)
        scores, idxs = entry["index"].search(qv, k)
        for sc, i in zip(scores[0], idxs[0]):
            merged.append((entry["texts"][i], float(sc)
                          * (0.7 + 0.3 * prior), doc))

    merged.sort(key=lambda x: x[1], reverse=True)
    picked = merged[:top_k_global]

    context = "\n\n---\n\n".join(
        re.sub(r'\s+', ' ', t).strip()[:800]
        for (t, _, _) in picked
    )

    return context, routed, None


def main():
    parser = argparse.ArgumentParser(description="RAG CLI")
    parser.add_argument("--query", type=str,
                        default="What discount does RICRAC offer?")
    parser.add_argument("--top-m", type=int, default=2)
    parser.add_argument("--k-per-doc", type=int, default=3)
    parser.add_argument("--top-k-global", type=int, default=5)
    parser.add_argument("--min-route-score", type=float, default=0.01)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-router", action="store_true")
    parser.add_argument("--force-docs", action="store_true")
    parser.add_argument("--models", type=str, default="",
                        help="Comma-separated model chain for fallback.")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models for your key and exit.")
    args = parser.parse_args()

    if args.list_models:
        names = list_available_models()
        if not names:
            print("Could not list models (missing key or API error).")
        else:
            print("Available models:")
            for n in names:
                print(f"- {n}")
        return

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

    print("\n--- Clean Context ---\n")
    print(context)

    if not context or len(context.strip()) < 40:
        print("\n--- AI Answer ---\n")
        print("This information is not available in the provided documents.")
        return

    chain = resolve_model_chain(args.models)

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
""".strip()

    answer, used_model = try_models_with_fallback(prompt, chain)
    print("\n--- AI Answer ---\n")
    print(answer)
    print(f"\n[Used Model: {used_model}]")


if __name__ == "__main__":
    main()
