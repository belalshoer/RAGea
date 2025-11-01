from typing import Optional, Tuple, Union, Sequence, List
from dataclasses import dataclass
from PIL import Image
import torch
from transformers import LlavaNextForConditionalGeneration, AutoProcessor
from generation.prompt import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT

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
class CaptionerCfg:
    model_name = "neulab/Pangea-7B-hf"
    use_fast = True
    dtype = torch.float16


class Captioner:

    def __init__(self, cfg: CaptionerCfg = CaptionerCfg()):
        self.model_name = cfg.model_name
        self.dtype = cfg.dtype
        self.use_fast = cfg.use_fast


        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_name,
            dtype=self.dtype,            
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            use_fast = self.use_fast
        )

        self.model.resize_token_embeddings(len(self.processor.tokenizer))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        
        ps = ensure_llava_patch_size(self.processor, self.model)
        print(f"[info] Using patch_size={ps}")

    def _build_prompt(
        self,
        user_prompt: str = DEFAULT_USER_PROMPT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> str:
        """
        Build the text template with <image> token expected by many LLaVA-style processors.
        """
        
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n<image>\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def caption_image(
        self,
        image: Union[str, Image.Image],
        user_prompt: str = DEFAULT_USER_PROMPT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_new_tokens: int = 128,
        min_new_tokens: int = 8,
        temperature: float = 0.1,
        top_p: float = 0.5,
        do_sample: bool = True,
        **gen_kwargs

    ) -> str:
        """
        Single image captioning.
        """

        img = _load_rgb(image)
        text = self._build_prompt(user_prompt=user_prompt, system_prompt=system_prompt)


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
                **gen_kwargs
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

    def caption_images(
        self,
        images: Sequence[Union[str, Image.Image]],
        user_prompt: str = DEFAULT_USER_PROMPT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_new_tokens: int = 128,
        min_new_tokens: int = 8,
        temperature: float = 0.1,
        top_p: float = 0.5,
        do_sample: bool = True,
        batch_size: Optional[int] = 32,
        **gen_kwargs
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
                    **gen_kwargs
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