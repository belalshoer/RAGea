from transformers import AutoProcessor, AutoModel
from langchain_core.embeddings import Embeddings 
from typing import List
from PIL.Image import Image
import torch
import time
from tqdm.auto import tqdm

class Siglip2FeatureExtractor(Embeddings):
    def __init__(self, model_name="google/siglip2-base-patch16-224", use_fast=True):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()
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
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)

        txt_emb = self.model.get_text_features(**inputs)
        txt_emb = torch.nn.functional.normalize(txt_emb, p=2, dim=1)

        return txt_emb.cpu().tolist()

    # @torch.inference_mode()
    # def encode_text(self, texts: List[str], batch_size: int = 512, show_progress: bool = True):
    #     total = len(texts)
    #     embeddings = []

    #     it = range(0, total, batch_size)
    #     if show_progress:
    #         it = tqdm(it,
    #                 total=(total + batch_size - 1) // batch_size,
    #                 desc="Embedding",
    #                 unit="batch",
    #                 dynamic_ncols=True)

    #     start = time.time()
    #     processed = 0

    #     for i in it:
    #         batch = texts[i:i + batch_size]
    #         inputs = self.processor(
    #             text=batch,
    #             return_tensors="pt",
    #             padding=True,
    #             truncation=True
    #         ).to(self.device)

    #         with torch.autocast("cuda", enabled=(self.device.type == "cuda")):
    #             txt_emb = self.model.get_text_features(**inputs)

    #         txt_emb = torch.nn.functional.normalize(txt_emb, p=2, dim=1)

    #         # move each batch off GPU to keep memory flat
    #         embeddings.append(txt_emb.cpu())

    #         # progress/ETA metrics
    #         processed += len(batch)
    #         if show_progress:
    #             elapsed = time.time() - start
    #             rate = processed / elapsed if elapsed > 0 else 0.0
    #             remaining = total - processed
    #             eta_sec = remaining / rate if rate > 0 else 0.0
    #             postfix = {
    #                 "items/s": f"{rate:,.1f}",
    #                 "done": f"{processed}/{total}",
    #                 "ETA": f"{eta_sec/60:.1f}m"
    #             }
    #             if self.device.type == "cuda":
    #                 postfix["GPU_mem"] = f"{torch.cuda.memory_allocated()/1e9:.2f} GB"
    #             it.set_postfix(postfix)

    #     embs = torch.cat(embeddings, dim=0)  # now on CPU
    #     return embs.tolist()

