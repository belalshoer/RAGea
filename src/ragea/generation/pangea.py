from typing import Optional, Tuple, Union, Sequence, List
from dataclasses import dataclass
from PIL import Image
import torch
from transformers import LlavaNextForConditionalGeneration, AutoProcessor
from generation.prompt import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT
import tqdm

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
        lang: str,
        user_prompt: str = DEFAULT_USER_PROMPT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        similar_captions: List[str] = None
    ) -> str:
        """
        Build the text template with <image> token expected by many LLaVA-style processors.
        """
        if not similar_captions:
            return (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n<image>\n{user_prompt}\nWrite the caption in {lang}.<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        else:
            similar_captions_concat = "\n".join(similar_captions)
            return (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n<image>\n{user_prompt}\nWrite the caption in {lang}.\n"
                f"You can find here some examples of captions close to the input image, make sure to mention all objects in the image.\n"
                f"{similar_captions_concat}"
                f"<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

    def caption_image(
        self,
        image: Union[str, Image.Image],
        lang: str,
        similar_captions: List[str] = None,
        user_prompt: str = DEFAULT_USER_PROMPT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_new_tokens: int = 128,
        min_new_tokens: int = 8,
        temperature: float = 0.2,
        top_p: float = 0.9,
        do_sample: bool = True,
        **gen_kwargs
    ) -> str:
        """
        Single image captioning.
        """

        img = _load_rgb(image)
        if similar_captions:
            text = self._build_prompt(user_prompt=user_prompt, system_prompt=system_prompt, lang=lang, 
                        similar_captions=similar_captions)
        else:
            text = self._build_prompt(user_prompt=user_prompt, system_prompt=system_prompt, lang=lang)


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
                pad_token_id= self.processor.tokenizer.eos_token_id,
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
        lang: str,
        images: Sequence[Union[str, Image.Image]],
        similar_captions: Sequence[List[str]] = None,
        user_prompt: str = DEFAULT_USER_PROMPT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_new_tokens: int = 128,
        min_new_tokens: int = 8,
        temperature: float = 0.2,
        top_p: float = 0.9,
        do_sample: bool = True,
        batch_size: Optional[int] = 4,
        **gen_kwargs
    ) -> List[str]:
        """
        Batched captioning. Accepts a sequence of image paths or PIL Images.
        Set `batch_size` to process large sets in chunks (avoids OOM).
        Returns a list of captions in the same order as `images`.
        """
        if do_sample is None:
            do_sample = temperature is not None and temperature > 0

        if similar_captions:
            pil_images = [(_load_rgb(x), c) for x, c in zip(images, similar_captions)]
        else:
            pil_images = [_load_rgb(x) for x in images]


        def _chunks(seq, n = None):
            if n is None or n <= 0:
                yield seq
            else:
                for i in range(0, len(seq), n):
                    yield seq[i:i+n]

        results: List[str] = []
        num_chunks = 1 if not batch_size or batch_size <= 0 else len(pil_images) // batch_size
        with torch.inference_mode():
            for chunk in tqdm.tqdm(_chunks(pil_images, batch_size), total=num_chunks):
                if similar_captions:
                    chunk_images = [c[0] for c in chunk]
                    chunk_captions = [c[1] for c in chunk]

                    texts = [self._build_prompt(user_prompt=user_prompt, system_prompt=system_prompt, lang=lang, similar_captions=similar_captions) for similar_captions in chunk_captions]
                else:
                    chunk_images = chunk
                    texts = [self._build_prompt(user_prompt=user_prompt, system_prompt=system_prompt, lang=lang)] * len(chunk)
                
                inputs = self.processor(
                    images=chunk_images,
                    text=texts,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)

                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id= self.processor.tokenizer.eos_token_id,
                    use_cache=True,
        
                    **gen_kwargs
                )

                decoded = self.processor.tokenizer.batch_decode(
                    output_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )

            # Clean echoes using the *matching* prompt
                for full, prompt in zip(decoded, texts):
                    s = full.strip()
                    if s.startswith(prompt):
                        s = s[len(prompt):].strip()
                    s = s.splitlines()[-1].strip() if "\n" in s else s
                    results.append(s)
        return results