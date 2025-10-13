# repos/base.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Any
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever


class VectorStore(ABC):
    """
    Abstract persistence boundary for a local vector store.
    """

    name: str
    dir: Path

    @abstractmethod
    def exists(self) -> bool: ...

    @abstractmethod
    def create_from_documents(self, chunks: List[Document]) -> None: ...

    @abstractmethod
    def add_documents(self, chunks: List[Document]) -> int:
        """Returns number of chunks added."""

    @abstractmethod
    def retrieve(self, img_path: str, k: int = None) -> List[Document]:
        """Return a retriever bound to this store."""
