from __future__ import annotations
from typing import Optional, Tuple, Union, Sequence, List
from dataclasses import dataclass
from PIL import Image
import torch
from transformers import LlavaNextForConditionalGeneration, AutoProcessor

DEFAULT_SYSTEM_PROMPT = "You are a helpful image captioning model."
DEFAULT_USER_PROMPT = "Caption the image in one concise sentence."


def ensure_llava_patch_size(processor, model, default: int = 14) -> int:
    patch: Optional[int] = getattr(getattr(processor, "image_processor", None), "patch_size", None)
    if patch is None:
        patch = getattr(getattr(model.config, "vision_config", None), "patch_size", None)
    if patch is None:
        patch = default
    processor.patch_size = int(patch)
    return processor.patch_size


def _load_rgb(img_or_path: Union[str, Image.Image]) -> Image.Image:
    if isinstance(img_or_path, str):
        return Image.open(img_or_path).convert("RGB")
    return img_or_path.convert("RGB")


@dataclass
class Captioner:
    
    model: LlavaNextForConditionalGeneration
    processor: AutoProcessor
    device: str

    @staticmethod
    def load(
        model_id: str = "neulab/Pangea-7B-hf",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "Captioner":
        """
        Load model & processor once, then reuse the Captioner object across files.
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.float16 if device == "cuda" else torch.float32

        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,            # respect dtype
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=True
        )
        ps = ensure_llava_patch_size(processor, model)
        print(f"[info] Using patch_size={ps}")

        # Keep embeddings in sync with the processor's tokenizer if needed
        if model.get_input_embeddings().weight.shape[0] != len(processor.tokenizer):
            model.resize_token_embeddings(len(processor.tokenizer))
      
        if  getattr(model, "hf_device_map", None) in (None, {}, "none"):
            model.to(device)
        return Captioner(model=model, processor=processor, device=device)

    def _build_prompt(
        self,
        user_prompt: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> str:
        # Text template with <image> token expected by many LLaVA-style processors
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n<image>\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def caption(
        self,
        image: Union[str, Image.Image],
        user_prompt: str = DEFAULT_USER_PROMPT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_new_tokens: int = 128,
        min_new_tokens: int = 8,
        temperature: float = 0.2,
        top_p: float = 0.9,
        do_sample: Optional[bool] = None,
    ) -> str:
        """
        Single image captioning; unchanged public API.
        """
        img = _load_rgb(image)
        text = self._build_prompt(user_prompt=user_prompt, system_prompt=system_prompt)

        # Default sampling behavior: sample iff temperature > 0
        if do_sample is None:
            do_sample = temperature is not None and temperature > 0

        inputs = self.processor(
            images=img,
            text=text,
            return_tensors="pt"
        ).to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                use_cache=True,
            )

        full_text = self.processor.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        ).strip()

        if full_text.startswith(text):
            full_text = full_text[len(text):].strip()

        caption = full_text.splitlines()[-1].strip() if "\n" in full_text else full_text
        return caption

    def caption_many(
        self,
        images: Sequence[Union[str, Image.Image]],
        user_prompt: str = DEFAULT_USER_PROMPT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_new_tokens: int = 128,
        min_new_tokens: int = 8,
        temperature: float = 0.2,
        top_p: float = 0.9,
        do_sample: Optional[bool] = None,
        batch_size: Optional[int] = None,
    ) -> List[str]:
        """
        Batched captioning. Accepts a sequence of image paths or PIL Images.
        Set `batch_size` to process large sets in chunks (avoids OOM).
        Returns a list of captions in the same order as `images`.
        """
        if do_sample is None:
            do_sample = temperature is not None and temperature > 0

        # Materialize/normalize inputs to PIL
        pil_images = [_load_rgb(x) for x in images]
        text = self._build_prompt(user_prompt=user_prompt, system_prompt=system_prompt)

        def _chunks(seq, n):
            if n is None or n <= 0:
                yield seq
            else:
                for i in range(0, len(seq), n):
                    yield seq[i:i+n]

        results: List[str] = []
        with torch.inference_mode():
            for chunk in _chunks(pil_images, batch_size):
                # Repeat the same prompt for each image in the chunk
                texts = [text] * len(chunk)
                inputs = self.processor(
                    images=chunk,
                    text=texts,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)

                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    use_cache=True,
                )

                decoded = self.processor.tokenizer.batch_decode(
                    output_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )

                # Clean up echoes and extra chatter, one-by-one
                for full in decoded:
                    s = full.strip()
                    if s.startswith(text):
                        s = s[len(text):].strip()
                    s = s.splitlines()[-1].strip() if "\n" in s else s
                    results.append(s)

        return results


# Convenience top-level functions for quick use in other files
def load_captioner(
    model_id: str = "neulab/Pangea-7B-hf",
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Captioner:
    return Captioner.load(model_id=model_id, device=device, dtype=dtype)


def caption_image(
    image_path: Union[str, Image.Image],
    user_prompt: str = DEFAULT_USER_PROMPT,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    model_id: Optional[str] = None,
    captioner: Optional[Captioner] = None,
    **gen_kwargs,
) -> str:
    cap = captioner or Captioner.load(model_id or "neulab/Pangea-7B-hf")
    return cap.caption(
        image=image_path,
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        **gen_kwargs,
    )


def caption_images(
    image_paths: Sequence[Union[str, Image.Image]],
    user_prompt: str = DEFAULT_USER_PROMPT,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    model_id: Optional[str] = None,
    captioner: Optional[Captioner] = None,
    batch_size: Optional[int] = None,
    **gen_kwargs,
) -> List[str]:
    """
    New helper: batch caption multiple images. Returns list of captions.
    """
    cap = captioner or Captioner.load(model_id or "neulab/Pangea-7B-hf")
    return cap.caption_many(
        images=image_paths,
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        batch_size=batch_size,
        **gen_kwargs,
    )


if __name__ == "__main__":
    # Single
    path = "example.png"
    print(path, caption_image(image_path=path))

    # Batch
    paths = ["example.png", "example2.jpg", "example3.jpeg"]
    caps = caption_images(paths, batch_size=2)
    for p, c in zip(paths, caps):
        print(p, c)
