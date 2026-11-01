"""BM25 and dense candidate fusion, preserving source offsets for evaluation."""

import re
import time

import numpy as np
from rank_bm25 import BM25Okapi


def tokenize(text):
    return re.findall(r"\w+", text.casefold())


class HybridRetriever:
    def __init__(self, chunks, collection, embeddings, cfg, depth=50, rank_constant=60):
        self.chunks = {c["chunk_id"]: c for c in chunks}
        self.ids = [c["chunk_id"] for c in chunks]
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks], k1=1.5, b=0.75)
        self.collection, self.embeddings, self.cfg = collection, embeddings, cfg
        self.depth, self.rank_constant = depth, rank_constant

    def candidates(self, question):
        started = time.perf_counter()
        dense = self.collection.query(
            query_embeddings=self.embeddings.embed([self.cfg.embedding.query_prefix + question]),
            n_results=min(self.depth, len(self.ids)), include=["distances"],
        )
        scores = self.bm25.get_scores(tokenize(question))
        # Stable sorting gives deterministic corpus-order tie breaks. Exclude
        # zero-evidence lexical matches instead of promoting arbitrary chunks.
        sparse = [int(i) for i in np.argsort(-scores, kind="stable") if scores[i] > 0][:self.depth]
        return {
            "dense": dense["ids"][0],
            "bm25": [self.ids[i] for i in sparse],
            "dense_distances": dict(zip(dense["ids"][0], dense["distances"][0])),
            "candidate_ms": (time.perf_counter() - started) * 1000,
        }

    def retrieve(self, candidates, dense_weight):
        if not 0 <= dense_weight <= 1:
            raise ValueError("dense_weight must be between zero and one")
        started = time.perf_counter()
        scores = {}
        for name, weight in (("dense", dense_weight), ("bm25", 1 - dense_weight)):
            if weight == 0:
                continue
            for rank, chunk_id in enumerate(candidates[name], 1):
                scores[chunk_id] = scores.get(chunk_id, 0) + weight / (self.rank_constant + rank)
        ranked = sorted(scores, key=lambda cid: (-scores[cid], cid))[:self.cfg.retrieval.k]
        retained, context, used = [], [], 0
        for cid in ranked:
            chunk = self.chunks[cid]
            available = self.cfg.retrieval.max_context_chars - used - (2 if context else 0)
            if available <= 0:
                break
            text = chunk["text"][:available]
            metadata = dict(chunk["metadata"], end_char=chunk["metadata"]["start_char"] + len(text))
            retained.append({"chunk_id": cid, "text": text, "metadata": metadata,
                             "rrf_score": scores[cid],
                             "cosine_distance": candidates["dense_distances"].get(cid)})
            context.append(text)
            used += len(text) + (2 if len(context) > 1 else 0)
        return {"context": "\n\n".join(context), "chunks": retained,
                "retrieval_ms": round(candidates["candidate_ms"] +
                                      (time.perf_counter() - started) * 1000, 2)}
