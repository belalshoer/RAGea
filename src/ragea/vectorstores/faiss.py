from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import faiss
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from .base import VectorStore

@dataclass
class VectorStoreCfg:
    k: int = 4
    base_dir: Path = Path("./vector_stores")
    allow_dangerous_deserialization: bool = True

class FaissVectorStore(VectorStore):
    def __init__(self, name: str, embedding, cfg: VectorStoreCfg = VectorStoreCfg()):
        self.name = name
        self.cfg = cfg
        self.dir = cfg.base_dir / name
        self.k = cfg.k
        self.embedding = embedding
        self.vs: Optional[FAISS] = None
        self._loaded: bool = False

    def exists(self) -> bool:
        return self.dir.exists()

    def create_from_documents(self, chunks: List[Document]) -> None:
        if self.exists():
            raise FileExistsError(f"Vector store '{self.name}' already exists at {self.dir}")
        
        dim = len(self.embedding.embed_query("test"))
        index = faiss.IndexFlatL2(dim)
        self.vs = FAISS.from_documents(chunks, self.embedding)
        self.loaded = True
        self._save()

    def _load(self) -> None:
        if self._loaded:
            return
        if not self.exists():
            raise FileNotFoundError(f"Index '{self.name}' not found at {self.dir}")
        
        self.vs = FAISS.load_local(
            str(self.dir),
            self.embedding,
            allow_dangerous_deserialization=self.cfg.allow_dangerous_deserialization,
        )

        self._loaded = True

    def add_documents(self, chunks: List[Document]) -> int:
        self._load()
        self.vs.add_documents(chunks)
        self._save()
      
    def _save(self) -> None:
        self.vs.save_local(str(self.dir))

    def retrieve(self, img_path: str, k: int = None) -> List[Document]:
        self._load()
        embedding = self.embedding.encode_image([img_path])
        return self.vs.similarity_search_by_vector(embedding[0], k = (k if k else self.k))
    
