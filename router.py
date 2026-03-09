# router.py
import os, re, html, pickle
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import issparse
import faiss
import numpy as np

def _clean(t: str) -> str:
    t = t.replace('\u00ad', '')
    t = re.sub(r'-\n', '', t)
    t = re.sub(r'\s+\n', '\n', t)
    t = re.sub(r'\n{2,}', '\n', t)
    t = re.sub(r'[ \t]{2,}', ' ', t)
    t = html.unescape(t)
    return t.strip()

def _to_dense_l2(X):
    X = normalize(X, norm='l2')
    if issparse(X): X = X.toarray()
    return X.astype('float32')

class Router:
    def __init__(self, min_df=1, ngram=(1,2), stop_words='english', store_dir="store"):
        self.store_dir = store_dir
        os.makedirs(self.store_dir, exist_ok=True)
        self.vectorizer = TfidfVectorizer(ngram_range=ngram, stop_words=stop_words, min_df=min_df)
        self.doc_names: List[str] = []
        self.index = None

    def build(self, profiles: Dict[str, str]):
        self.doc_names = list(profiles.keys())
        texts = [_clean(profiles[k]) for k in self.doc_names]
        tfidf = self.vectorizer.fit_transform(texts)
        vecs = _to_dense_l2(tfidf)
        d = vecs.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(vecs)
        return self

    def save(self):
        # save vectorizer + doc_names
        with open(os.path.join(self.store_dir, "router.pkl"), "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "doc_names": self.doc_names}, f)
        faiss.write_index(self.index, os.path.join(self.store_dir, "router.faiss"))

    def load(self):
        with open(os.path.join(self.store_dir, "router.pkl"), "rb") as f:
            obj = pickle.load(f)
        self.vectorizer = obj["vectorizer"]
        self.doc_names = obj["doc_names"]
        self.index = faiss.read_index(os.path.join(self.store_dir, "router.faiss"))
        return self

    def route(self, query: str, top_m=2, min_score=0.08) -> List[Tuple[str, float]]:
        q = self.vectorizer.transform([query])
        q = _to_dense_l2(q)
        m = min(top_m, len(self.doc_names))
        scores, inds = self.index.search(q, m)
        res = []
        for sc, i in zip(scores[0], inds[0]):
            if i < 0: continue
            if sc >= min_score:
                res.append((self.doc_names[i], float(sc)))
        return res

    def explain_terms(self, query: str, doc_name: str, top_k=5) -> List[str]:
        # ارجع أهم n-grams المشتركة (تقريبية) بين السؤال والمستند
        q_vec = self.vectorizer.transform([query])
        doc_idx = self.doc_names.index(doc_name)
        # خد الـ tf-idf row للمستند
        # NOTE: مفيش direct تخزين للـ tf-idf للمستند بعد fit، فهنعيد vectorize للنص نفسه عند البناء لو حفظناه.
        # كحل بسيط: استخرج أعلى terms في السؤال واعرضها.
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        q_arr = q_vec.toarray()[0]
        top = q_arr.argsort()[::-1][:top_k]
        return feature_names[top].tolist()
