from typing import Dict, List, Any
import os

import numpy as np
import pandas as pd

import faiss


class FaissIndex:
    def __init__(
        self,
        data_path: str,
        embedding_model_detail: str,
        indices_name: str,
        items_name: str,
        dim: int,
        retrieval_top_k: int,
        distance_column_name: str,
        candidate_column_name: str,
    ) -> None:
        self.data_path = data_path
        self.embedding_model_detail = embedding_model_detail
        self.indices_name = indices_name
        self.items_name = items_name
        self.indices_path = os.path.join(
            self.data_path,
            self.embedding_model_detail,
            self.indices_name,
        )
        self.items_path = os.path.join(
            self.data_path,
            self.items_name,
        )

        self.dim = dim
        self.index = None
        self.df = None
        self.loaded = False

        self.retrieval_top_k = retrieval_top_k
        self.distance_column_name = distance_column_name
        self.candidate_column_name = candidate_column_name

    def add(
        self,
        embedded: np.ndarray,
    ) -> None:
        self.index.add(embedded)

    def search(
        self,
        query_embedding: np.ndarray,
    ) -> List[Dict[str, Any]]:
        if not self.loaded:
            self.load()

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        elif query_embedding.ndim == 2:
            query_embedding = query_embedding.astype(np.float32)
        else:
            raise ValueError("query_embedding must be 1D or 2D array")

        distances, indices = self.index.search(
            query_embedding,
            k=self.retrieval_top_k,
        )

        candidates = []
        for idx, distance in zip(indices[0], distances[0]):
            row = self.df.iloc[idx].to_dict()
            row[self.distance_column_name] = float(distance)
            candidates.append(row)
        return candidates

    def save(self) -> None:
        os.makedirs(
            os.path.join(
                self.data_path,
                self.embedding_model_detail,
            ),
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            self.indices_path,
        )

    def load(self) -> None:
        if self.loaded:
            return

        if not os.path.exists(self.indices_path):
            raise FileNotFoundError(f"Missing indices: {self.indices_path}")
        if not os.path.exists(self.items_path):
            raise FileNotFoundError(f"Missing items: {self.items_path}")

        self.index = faiss.read_index(self.indices_path)
        self.df = pd.read_csv(self.items_path)
        self.loaded = True
