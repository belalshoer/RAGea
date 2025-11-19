from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import faiss
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from features import Siglip2FeatureExtractor 
from .base import VectorStore
from PIL.Image import Image
import tqdm

@dataclass
class VectorStoreCfg:
    k: int = 4
    bs: int = 5000
    base_dir: Path = Path("./stores")
    embedding = Siglip2FeatureExtractor()
    allow_dangerous_deserialization: bool = True

class FaissVectorStore(VectorStore):
    def __init__(self, name: str, cfg: VectorStoreCfg = VectorStoreCfg()):
        self.name = name
        self.cfg = cfg
        self.dir = cfg.base_dir / name
        self.bs = cfg.bs
        self.k = cfg.k
        self.embedding = cfg.embedding
        self.vs: Optional[FAISS] = None
        self._loaded: bool = False

    def exists(self) -> bool:
        return self.dir.exists()

    def create_from_documents(self, chunks: List[Document]) -> None:
        if self.exists():
            raise FileExistsError(f"Vector store '{self.name}' already exists at {self.dir}")
        
        dim = len(self.embedding.embed_query("test"))
        
        index = faiss.IndexFlatL2(dim)
        self.vs = FAISS(
            embedding_function=self.embedding,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        self.loaded = True
        self._save()
    
        return self.add_documents(chunks) 

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

        print("Adding documents to vector store..")
        for i in tqdm.tqdm(range(0, len(chunks), self.bs)):
            self.vs.add_documents(chunks[i: i+self.bs]) 

        self._save()

        return self.vs.index.ntotal
      
    def _save(self) -> None:
        self.vs.save_local(str(self.dir))

    def retrieve(self, img: Image, k: int = None, search_type: str = None) -> List[Document]:
        self._load()
        embedding = self.embedding.encode_image([img])
        docs = self.vs.similarity_search_by_vector(embedding[0], k = (k if k else self.k), search_type=search_type)

        retrieved_data = [{
            "en_caption": d.page_content,
            "img_id": d.metadata.get("img_id"),
            "source": d.metadata.get("source")
        } for d in docs]

        return retrieved_data 

