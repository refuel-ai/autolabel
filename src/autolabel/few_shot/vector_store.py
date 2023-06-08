from __future__ import annotations

import heapq
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type

import numpy as np
import torch
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.utils import maximal_marginal_relevance
from torch import Tensor


def _results_to_docs_and_scores(results: Any) -> List[Tuple[Document, float]]:
    return [
        (Document(page_content=result[0], metadata=result[1] or {}), result[2])
        for result in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def cos_sim(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    Returns:
        cos_sim: Matrix with res(i)(j) = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def semantic_search(
    query_embeddings: Tensor,
    corpus_embeddings: Tensor,
    query_chunk_size: int = 100,
    corpus_chunk_size: int = 500000,
    top_k: int = 10,
    score_function: Callable[[Tensor, Tensor], Tensor] = cos_sim,
):
    """
    Semantic similarity search based on cosine similarity score. Implementation from this project: https://github.com/UKPLab/sentence-transformers
    """

    if isinstance(query_embeddings, (np.ndarray, np.generic)):
        query_embeddings = torch.from_numpy(query_embeddings)
    elif isinstance(query_embeddings, list):
        query_embeddings = torch.stack(query_embeddings)

    if len(query_embeddings.shape) == 1:
        query_embeddings = query_embeddings.unsqueeze(0)

    if isinstance(corpus_embeddings, (np.ndarray, np.generic)):
        corpus_embeddings = torch.from_numpy(corpus_embeddings)
    elif isinstance(corpus_embeddings, list):
        corpus_embeddings = torch.stack(corpus_embeddings)

    # Check that corpus and queries are on the same device
    if corpus_embeddings.device != query_embeddings.device:
        query_embeddings = query_embeddings.to(corpus_embeddings.device)

    queries_result_list = [[] for _ in range(len(query_embeddings))]

    for query_start_idx in range(0, len(query_embeddings), query_chunk_size):
        # Iterate over chunks of the corpus
        for corpus_start_idx in range(0, len(corpus_embeddings), corpus_chunk_size):
            # Compute cosine similarities
            cos_scores = score_function(
                query_embeddings[query_start_idx : query_start_idx + query_chunk_size],
                corpus_embeddings[
                    corpus_start_idx : corpus_start_idx + corpus_chunk_size
                ],
            )

            # Get top-k scores
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
                cos_scores,
                min(top_k, len(cos_scores[0])),
                dim=1,
                largest=True,
                sorted=False,
            )
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(cos_scores)):
                for sub_corpus_id, score in zip(
                    cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]
                ):
                    corpus_id = corpus_start_idx + sub_corpus_id
                    query_id = query_start_idx + query_itr
                    if len(queries_result_list[query_id]) < top_k:
                        heapq.heappush(
                            queries_result_list[query_id], (score, corpus_id)
                        )  # heaqp tracks the quantity of the first element in the tuple
                    else:
                        heapq.heappushpop(
                            queries_result_list[query_id], (score, corpus_id)
                        )

    # change the data format and sort
    for query_id in range(len(queries_result_list)):
        for doc_itr in range(len(queries_result_list[query_id])):
            score, corpus_id = queries_result_list[query_id][doc_itr]
            queries_result_list[query_id][doc_itr] = {
                "corpus_id": corpus_id,
                "score": score,
            }
        queries_result_list[query_id] = sorted(
            queries_result_list[query_id], key=lambda x: x["score"], reverse=True
        )
    return queries_result_list


class VectorStoreWrapper(VectorStore):
    def __init__(
        self,
        embedding_function: Optional[Embeddings] = None,
        corpus_embeddings: Optional[Tensor] = None,
        texts: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        self._embedding_function = embedding_function
        self._corpus_embeddings = corpus_embeddings
        self._texts = texts
        self._metadatas = metadatas

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, str]]] = None,
    ) -> List[str]:
        """Run texts through the embeddings and add to the vectorstore. Currently, the vectorstore is reinitialized each time, because we do not require a persistent vector store for example selection.
        Args:
            texts (Iterable[str]): Texts to add to the vectorstore.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
        Returns:
            List[str]: List of IDs of the added texts.
        """
        embeddings = None
        if self._embedding_function is not None:
            embeddings = self._embedding_function.embed_documents(list(texts))
        self._corpus_embeddings = torch.tensor(embeddings)
        self._texts = texts
        self._metadatas = metadatas
        return metadatas

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Run semantic similarity search.
        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.
        Returns:
            List[Document]: List of documents most similar to the query text.
        """
        docs_and_scores = self.similarity_search_with_score(query, k, filter=filter)
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Run semantic similarity search and retrieve distances.
        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.
        Returns:
            List[Tuple[Document, float]]: List of documents most similar to the query
                text with distance in float.
        """
        query_embeddings = torch.tensor([self._embedding_function.embed_query(query)])
        result_ids_and_scores = semantic_search(
            corpus_embeddings=self._corpus_embeddings,
            query_embeddings=query_embeddings,
            top_k=k,
        )
        result_ids = [result["corpus_id"] for result in result_ids_and_scores[0]]
        scores = [result["score"] for result in result_ids_and_scores[0]]
        results = {}
        results["documents"] = [[self._texts[index] for index in result_ids]]
        results["distances"] = [scores]
        results["metadatas"] = [[self._metadatas[index] for index in result_ids]]

        return _results_to_docs_and_scores(results)

    def max_marginal_relevance_search_by_vector(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        query_embedding = self._embedding_function.embed_query(query)
        query_embeddings = torch.tensor([query_embedding])
        result_ids_and_scores = semantic_search(
            corpus_embeddings=self._corpus_embeddings,
            query_embeddings=query_embeddings,
            top_k=fetch_k,
        )
        result_ids = [result["corpus_id"] for result in result_ids_and_scores[0]]
        scores = [result["score"] for result in result_ids_and_scores[0]]

        fetched_embeddings = torch.index_select(
            input=self._corpus_embeddings, dim=0, index=torch.tensor(result_ids)
        ).tolist()
        mmr_selected = maximal_marginal_relevance(
            np.array([query_embedding], dtype=np.float32),
            fetched_embeddings,
            k=k,
            lambda_mult=lambda_mult,
        )
        selected_result_ids = [result_ids[i] for i in mmr_selected]
        selected_scores = [scores[i] for i in mmr_selected]
        results = {}
        results["documents"] = [[self._texts[index] for index in selected_result_ids]]
        results["distances"] = [selected_scores]
        results["metadatas"] = [
            [self._metadatas[index] for index in selected_result_ids]
        ]

        return _results_to_docs_and_scores(results)

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        docs_and_scores = self.max_marginal_relevance_search_by_vector(
            query, k, fetch_k, lambda_mult=lambda_mult
        )
        return [doc for doc, _ in docs_and_scores]

    @classmethod
    def from_texts(
        cls: Type[VectorStoreWrapper],
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> VectorStoreWrapper:
        """Create a vectorstore from raw text.
        The data will be ephemeral in-memory.
        Args:
            texts (List[str]): List of texts to add to the collection.
            embedding (Optional[Embeddings]): Embedding function. Defaults to None.
            metadatas (Optional[List[dict]]): List of metadatas. Defaults to None.
        Returns:
            vector_store: Vectorstore with seedset embeddings
        """
        vector_store = cls(
            embedding_function=embedding, corpus_embeddings=None, texts=None, **kwargs
        )
        vector_store.add_texts(texts=texts, metadatas=metadatas)
        return vector_store
