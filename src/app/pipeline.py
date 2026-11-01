"""Long-lived question-only RAG service using the experiment implementations."""

from pathlib import Path

from app.rag import Embeddings, OpenRouter, make_retriever, open_collection, prepare_chunks


class RAGPipeline:
    """Open a prepared index once; create, invoke, and close on the same thread."""

    def __init__(self, cfg, output: Path):
        self.cfg = cfg
        output.mkdir(parents=True, exist_ok=True)
        self.api = OpenRouter(cfg, output)
        self.embeddings = Embeddings(cfg, self.api)
        try:
            chunks, report = prepare_chunks(cfg)
            self.collection = open_collection(cfg, chunks, report)
            self.retrieve = make_retriever(cfg, chunks, self.collection, self.embeddings)
        except Exception:
            self.close()
            raise

    def invoke(self, question: str) -> dict:
        if not question.strip():
            raise ValueError("A nonempty question is required")
        result = self.retrieve(question)
        return {**result, **self.api.answer(question, result["context"])}

    def close(self):
        self.embeddings.db.close()
