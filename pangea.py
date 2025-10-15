from __future__ import annotations
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from PIL import Image
import torch
from transformers import LlavaNextForConditionalGeneration, AutoProcessor
from typing import Optional

DEFAULT_SYSTEM_PROMPT = "You are a helpful image captioning model."
DEFAULT_USER_PROMPT = "Caption the image in one concise sentence."


def ensure_llava_patch_size(processor, model, default: int = 14) -> int:
    """
    Ensure processor.patch_size is an int. Returns the patch size used.
    Tries (in order): processor.image_processor.patch_size,
                      model.config.vision_config.patch_size,
                      fallback default (14).
    """
    patch: Optional[int] = getattr(getattr(processor, "image_processor", None), "patch_size", None)
    if patch is None:
        patch = getattr(getattr(model.config, "vision_config", None), "patch_size", None)
    if patch is None:
        patch = default
    processor.patch_size = int(patch)
    return processor.patch_size
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
            dtype=torch.float16,           
            device_map="auto",
            trust_remote_code=True,
        )
        processor = processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    use_fast=True           
)
        ps = ensure_llava_patch_size(processor, model)
        print(f"[info] Using patch_size={ps}")

        # Keep embeddings in-sync with the processor's tokenizer if needed
        if model.get_input_embeddings().weight.shape[0] != len(processor.tokenizer):
            model.resize_token_embeddings(len(processor.tokenizer))

        model.to(device)
        return Captioner(model=model, processor=processor, device=device)

    def _build_prompt(
        self,
        user_prompt: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> str:
     
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"<image>\n{user_prompt}"},
        ]
       
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
        Generate a caption for an image with a customizable user message.
        - `image`: path or PIL.Image
        - `user_prompt`: the instruction you want to give the model (e.g., "Write a funny IG caption")
        """
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")

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

        # Decode and try to strip any echoed prompt
        full_text = self.processor.decode(
            output_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        ).strip()

        # Many chat models echo the prompt; remove it if present
        if full_text.startswith(text):
            full_text = full_text[len(text):].strip()

        # As a last pass, take the last line as the caption if there’s extra chatter
        caption = full_text.splitlines()[-1].strip() if "\n" in full_text else full_text
        return caption


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
    """
    One-call helper: either pass an existing Captioner or we’ll load one on the fly.
    """
    cap = captioner or Captioner.load("neulab/Pangea-7B-hf")
    return cap.caption(
        image=image_path,
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        **gen_kwargs,
    )
if __name__== "__main__":
    path="example.png"
    caption =caption_image(image_path=path)
    print(path, caption)
