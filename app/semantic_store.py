import logging
import threading
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_PATH = "./models/all-MiniLM-L6-v2"
logger = logging.getLogger(__name__)


class SemanticStore:
    def __init__(self, model_path: str = MODEL_PATH):
        self._model = SentenceTransformer(model_path)
        embedding_size = self._model.get_sentence_embedding_dimension()
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_size))
        self._lock = threading.RLock()
        self._next_id = 0
        self.id_map: dict[int, str] = {}

    def find_similar(self, prompt: str, threshold: float = 0.9) -> Optional[str]:
        with self._lock:
            if self._index.ntotal == 0:
                return None

            embedding = self._encode(prompt)
            scores, ids = self._index.search(embedding, k=1)
            best_id = int(ids[0][0])
            best_score = float(scores[0][0])
            logger.info(
                "Semantic search score=%0.4f threshold=%0.4f faiss_id=%s",
                best_score,
                threshold,
                best_id,
            )

            if best_id == -1 or best_score < threshold:
                return None

            return self.id_map.get(best_id)

    def add_to_cache(self, prompt: str, response_json: str) -> int:
        with self._lock:
            embedding = self._encode(prompt)
            faiss_id = self._next_id
            self._next_id += 1

            self._index.add_with_ids(
                embedding,
                np.array([faiss_id], dtype=np.int64),
            )
            self.id_map[faiss_id] = response_json
            return faiss_id

    def _encode(self, text: str) -> np.ndarray:
        embedding = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray([embedding], dtype=np.float32)
