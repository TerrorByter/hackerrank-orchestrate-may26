"""
Retriever — hybrid BM25 + FAISS semantic search over the support corpus.

Uses sentence-transformers for dense embeddings (FAISS) and BM25 for sparse
keyword matching. Results are fused with Reciprocal Rank Fusion (RRF) so that
strong keyword signals (e.g. "bedrock", "minimum") can override cases where the
dense model drifts to semantically adjacent but wrong documents.
"""

import pickle
import re
import numpy as np
from pathlib import Path
from typing import Optional

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from corpus_loader import Document

RRF_K = 60  # Standard RRF constant; higher = smoother rank blending


class Retriever:
    """
    Hybrid retriever: dense (FAISS cosine) + sparse (BM25), fused via RRF.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.documents: list[Document] = []
        self.embeddings: Optional[np.ndarray] = None
        self.bm25: Optional[BM25Okapi] = None
        self._bm25_corpus: list[list[str]] = []

    # ------------------------------------------------------------------
    # Tokenisation (shared between index build and query time)
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> SentenceTransformer:
        if self.model is None:
            print(f"  Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        return self.model

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, name: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        # Use a model-specific subdirectory so MiniLM and bge-large caches
        # never collide (they have different embedding dimensions).
        safe_model = re.sub(r"[^\w\-]", "_", self.model_name.split("/")[-1])
        cache = Path(self.cache_dir) / safe_model
        cache.mkdir(parents=True, exist_ok=True)
        return cache / name

    def _try_load_cache(self) -> bool:
        if self.cache_dir is None:
            return False

        index_path = self._cache_path("faiss.index")
        meta_path = self._cache_path("documents.pkl")
        embeddings_path = self._cache_path("embeddings.npy")

        if not all(p and p.exists() for p in [index_path, meta_path, embeddings_path]):
            return False

        try:
            print("  Loading cached FAISS index...")
            self.index = faiss.read_index(str(index_path))
            with open(meta_path, "rb") as f:
                self.documents = pickle.load(f)
            self.embeddings = np.load(str(embeddings_path))
            print(f"  Loaded {len(self.documents)} cached document chunks")
        except Exception as e:
            print(f"  Warning: FAISS cache load failed: {e}")
            return False

        # Load BM25 corpus — or rebuild it if the cache predates this feature
        bm25_path = self._cache_path("bm25_corpus.pkl")
        if bm25_path and bm25_path.exists():
            try:
                with open(bm25_path, "rb") as f:
                    self._bm25_corpus = pickle.load(f)
                self.bm25 = BM25Okapi(self._bm25_corpus)
                print(f"  Rebuilt BM25 index from {len(self._bm25_corpus)} cached token lists")
            except Exception as e:
                print(f"  Warning: BM25 cache load failed, rebuilding: {e}")
                self._build_bm25(self.documents)
                self._save_cache()
        else:
            print("  BM25 cache missing — building from loaded documents...")
            self._build_bm25(self.documents)
            self._save_cache()

        return True

    def _save_cache(self):
        if self.cache_dir is None:
            return

        index_path = self._cache_path("faiss.index")
        meta_path = self._cache_path("documents.pkl")
        embeddings_path = self._cache_path("embeddings.npy")
        bm25_path = self._cache_path("bm25_corpus.pkl")

        try:
            faiss.write_index(self.index, str(index_path))
            with open(meta_path, "wb") as f:
                pickle.dump(self.documents, f)
            np.save(str(embeddings_path), self.embeddings)
            if bm25_path and self._bm25_corpus:
                with open(bm25_path, "wb") as f:
                    pickle.dump(self._bm25_corpus, f)
            print(f"  Saved index cache to {self.cache_dir}")
        except Exception as e:
            print(f"  Warning: Cache save failed: {e}")

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_bm25(self, documents: list[Document]):
        tokenized = [self._tokenize(f"{doc.title} {doc.content}") for doc in documents]
        self._bm25_corpus = tokenized
        self.bm25 = BM25Okapi(tokenized)
        print(f"  Built BM25 index over {len(tokenized)} documents")

    def build_index(self, documents: list[Document]):
        """
        Build FAISS dense index and BM25 sparse index. Uses cache when valid.
        """
        self._load_model()

        if self._try_load_cache():
            if len(self.documents) == len(documents):
                return
            print("  Cache size mismatch, rebuilding...")

        self.documents = documents
        model = self._load_model()

        texts = [f"{doc.title}\n{doc.content}" for doc in documents]

        print(f"  Embedding {len(texts)} document chunks...")
        self.embeddings = model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,
            batch_size=64,
        )

        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings.astype(np.float32))
        print(f"  Built FAISS index with {self.index.ntotal} vectors (dim={dimension})")

        self._build_bm25(documents)
        self._save_cache()

    # ------------------------------------------------------------------
    # Per-model candidate retrieval (domain-filtered)
    # ------------------------------------------------------------------

    def _faiss_candidates(
        self,
        query_vec: np.ndarray,
        domain: Optional[str],
        candidate_k: int,
    ) -> list[int]:
        # Over-fetch to absorb domain filtering losses
        n_search = min(candidate_k * 3 if domain else candidate_k, len(self.documents))
        scores, indices = self.index.search(query_vec, n_search)
        results = []
        for _, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            if domain and self.documents[idx].domain != domain.lower():
                continue
            results.append(int(idx))
            if len(results) >= candidate_k:
                break
        return results

    def _bm25_candidates(
        self,
        query: str,
        domain: Optional[str],
        candidate_k: int,
    ) -> list[int]:
        tokens = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokens)
        order = np.argsort(bm25_scores)[::-1]
        results = []
        for idx in order:
            if domain and self.documents[int(idx)].domain != domain.lower():
                continue
            results.append(int(idx))
            if len(results) >= candidate_k:
                break
        return results

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------

    def _rrf_fuse(
        self,
        faiss_ranked: list[int],
        bm25_ranked: list[int],
        top_k: int,
    ) -> list[tuple[Document, float]]:
        scores: dict[int, float] = {}
        for rank, idx in enumerate(faiss_ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, idx in enumerate(bm25_ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)
        return [(self.documents[i], scores[i]) for i in sorted_ids[:top_k]]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        domain: Optional[str] = None,
        top_k: int = 5,
    ) -> list[tuple[Document, float]]:
        """
        Retrieve top-k documents using hybrid BM25 + FAISS with RRF fusion.

        Falls back to a cross-domain search if domain-filtered results are
        too sparse (e.g. Visa corpus has only ~42 chunks).
        """
        if self.index is None or self.model is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        query_vec = self.model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        candidate_k = max(top_k * 6, 20)

        faiss_ranked = self._faiss_candidates(query_vec, domain, candidate_k)
        bm25_ranked = (
            self._bm25_candidates(query, domain, candidate_k)
            if self.bm25 is not None
            else []
        )
        results = self._rrf_fuse(faiss_ranked, bm25_ranked, top_k)

        # Fallback: supplement with cross-domain results if too few
        if domain and len(results) < top_k // 2:
            faiss_all = self._faiss_candidates(query_vec, None, candidate_k)
            bm25_all = (
                self._bm25_candidates(query, None, candidate_k)
                if self.bm25 is not None
                else []
            )
            all_results = self._rrf_fuse(faiss_all, bm25_all, top_k)
            seen = {r[0].doc_id for r in results}
            for doc, score in all_results:
                if doc.doc_id not in seen:
                    results.append((doc, score))
                    seen.add(doc.doc_id)
                if len(results) >= top_k:
                    break

        return results
