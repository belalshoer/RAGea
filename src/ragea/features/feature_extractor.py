from transformers import AutoProcessor, AutoModel
from langchain_core.embeddings import Embeddings 
from dataclasses import dataclass
from typing import List
from PIL.Image import Image
import torch
from tqdm.auto import tqdm

@dataclass
class FeatureExtractorCfg:
    model_name = "google/siglip2-giant-opt-patch16-384"
    use_fast = True

class Siglip2FeatureExtractor(Embeddings):
    def __init__(self,  cfg: FeatureExtractorCfg = FeatureExtractorCfg()):
        self.model_name = cfg.model_name
        self.use_fast = cfg.use_fast

        self.processor = AutoProcessor.from_pretrained(
            self.model_name, 
            use_fast=self.use_fast
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            dtype=torch.float16,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.encode_text(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.encode_text([text])[0]

    @torch.inference_mode()
    def encode_image(self, images: List[Image]):
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)

        img_emb = self.model.get_image_features(**inputs)
        img_emb = torch.nn.functional.normalize(img_emb, p=2, dim=1)

        return img_emb.cpu().tolist()
        
    @torch.inference_mode()
    def encode_text(self, texts: List[str]):
        for text in texts:
            text = text.lower()
        inputs = self.processor(text=texts, return_tensors="pt", padding="max_length", max_length = 64, truncation=True).to(self.device)

        txt_emb = self.model.get_text_features(**inputs)
        txt_emb = torch.nn.functional.normalize(txt_emb, p=2, dim=1)

        return txt_emb.cpu().tolist()
