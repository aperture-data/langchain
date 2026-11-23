"""ApertureDB vector store for the LangChain community integration."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.indexing.base import UpsertResponse
from langchain_core.vectorstores import VectorStore

ENGINE = "HNSW"
METRIC = "CS"
DESCRIPTOR_SET = "langchain"
BATCHSIZE = 1000
PROPERTY_PREFIX = "lc_"
TEXT_PROPERTY = "text"
UNIQUEID_PROPERTY = "uniqueid"


def _sdk_store() -> Any:
    # SDK is optional for applications importing langchain-community.
    try:
        from aperturedb.LangChain import ApertureDB as SDKVectorStore
    except ImportError as exc:
        raise ImportError(
            "Install the ApertureDB SDK with pip install 'aperturedb>=1.0.3'."
        ) from exc
    return SDKVectorStore


class ApertureDB(VectorStore):
    """Store LangChain documents and embeddings in a named ApertureDB vector index."""

    def __init__(
        self,
        embeddings: Embeddings,
        descriptor_set: str = DESCRIPTOR_SET,
        dimensions: Optional[int] = None,
        engine: Optional[str] = None,
        metric: Optional[str] = None,
        log_level: int = logging.WARN,
        properties: Optional[Dict] = None,
        **kwargs: Any,
    ) -> None:
        """Open or create a vector store with the given index metadata."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self._store = _sdk_store()(
            embeddings=embeddings,
            descriptor_set=descriptor_set,
            dimensions=dimensions,
            engine=engine,
            metric=metric,
            properties=properties,
            **kwargs,
        )
        self.descriptor_set = self._store.descriptor_class
        self.embedding_function = embeddings
        self.connection = self._store.connection
        self.utils = self._store.utils
        self.dimensions = self._store.dimensions
        self.engine = self._store.engine
        self.metric = self._store.metric
        self.properties = self._store.properties

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_function

    def upsert(self, items: Sequence[Document], /, **kwargs: Any) -> UpsertResponse:
        """Insert documents, replacing existing documents with matching IDs."""
        ids = self._store.add_texts(
            [doc.page_content for doc in items],
            [doc.metadata for doc in items],
            ids=[doc.id for doc in items],
            **kwargs,
        )
        return UpsertResponse(succeeded=ids, failed=[])

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        return self._store.add_texts(texts, metadatas, **kwargs)

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> bool:
        return self._store.delete(ids, **kwargs)

    def get_by_ids(self, ids: Sequence[str], /) -> List[Document]:
        return self._store.get_by_ids(ids)

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        return self._store.similarity_search(query, k, **kwargs)

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        return self._store.similarity_search_by_vector(embedding, k, **kwargs)

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        return self._store.similarity_search_with_score(query, k, **kwargs)

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        return self._store.max_marginal_relevance_search_by_vector(
            embedding, k, fetch_k, lambda_mult, **kwargs
        )

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        return self._store.max_marginal_relevance_search(
            query, k, fetch_k, lambda_mult, **kwargs
        )

    @classmethod
    def from_texts(
        cls: Type[ApertureDB],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> ApertureDB:
        ids = kwargs.pop("ids", None)
        store = cls(embeddings=embedding, **kwargs)
        store.add_texts(texts, metadatas, ids=ids)
        return store

    @classmethod
    def from_documents(
        cls: Type[ApertureDB],
        documents: List[Document],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> ApertureDB:
        ids = kwargs.pop("ids", None)
        return cls.from_texts(
            [doc.page_content for doc in documents],
            embedding,
            [doc.metadata for doc in documents],
            ids=ids if ids is not None else [doc.id for doc in documents],
            **kwargs,
        )

    @classmethod
    def delete_vectorstore(cls, descriptor_set: str) -> None:
        """Remove a vector store and its documents."""
        _sdk_store().delete_vectorstore(descriptor_set)

    @classmethod
    def list_vectorstores(cls) -> List[dict]:
        """Return descriptor-set names, dimensions, engines, metrics and metadata."""
        rows = _sdk_store().list_vectorstores()
        if not rows or "kind" not in rows[0] or "_name" in rows[0]:
            return rows

        from aperturedb.DescriptorEngines import select_engine

        stores: Dict[str, dict] = {}
        for index in rows:
            name = index["class"]
            params = index["params"]
            store = stores.setdefault(
                name,
                {
                    **index.get("metadata", {}),
                    "_name": name,
                    "_dimensions": params["dimensions"],
                    "_engines": [],
                    "_metrics": [],
                },
            )
            engine = select_engine({index["kind"]: {}})
            if engine not in store["_engines"]:
                store["_engines"].append(engine)
            metrics = params["metric"]
            for metric in [metrics] if isinstance(metrics, str) else metrics:
                if metric not in store["_metrics"]:
                    store["_metrics"].append(metric)
        return list(stores.values())
