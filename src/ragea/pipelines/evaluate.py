import argparse
from typing import Any, Dict, List, Optional
from datasets import load_dataset
import pandas as pd
import time
from tqdm.auto import tqdm

from vectorstores import FaissVectorStore
from generation.pangea import caption_images, caption_image, Captioner
from pipelines.metrics import evaluate_captions  # uses BLEU@4, ROUGE-L, ChrF++, BERTScore

captioner = Captioner.load("neulab/Pangea-7B-hf")

LANG_SPLITS = [
    "ar","bn","cs","da","de","el","en","es","fa","fi","fil","fr","hi","hr","hu",
    "id","it","he","ja","ko","mi","nl","no","pl","pt","quz","ro","ru","sv","sw",
    "te","th","tr","uk","vi","zh"
]

LANG_NAME = {
    "ar":"Arabic","bn":"Bengali","cs":"Czech","da":"Danish","de":"German","el":"Greek",
    "en":"English","es":"Spanish","fa":"Persian","fi":"Finnish","fil":"Filipino",
    "fr":"French","hi":"Hindi","hr":"Croatian","hu":"Hungarian","id":"Indonesian",
    "it":"Italian","he":"Hebrew","ja":"Japanese","ko":"Korean","mi":"Māori","nl":"Dutch",
    "no":"Norwegian","pl":"Polish","pt":"Portuguese","quz":"Quechua","ro":"Romanian",
    "ru":"Russian","sv":"Swedish","sw":"Swahili","te":"Telugu","th":"Thai","tr":"Turkish",
    "uk":"Ukrainian","vi":"Vietnamese","zh":"Chinese"
}

def _chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def caption_backend_pangea(images: List[Any], user_prompt: str, batch_size: int = 4, lang: str = "") -> List[str]:
    """
    Use generation.pangea with visible progress.
    Prefer true batched path; if it errors, fallback to sequential with progress.
    """
    preds: List[str] = []
    pbar = tqdm(total=len(images), desc=f"Captioning [{lang or 'pangea'}]", unit="img", leave=False)
    try:
        # Manually batch so we can update the progress bar as we go
        for batch in _chunked(images, batch_size):
            out = caption_images(image_paths=batch, batch_size=batch_size, user_prompt=user_prompt, captioner=captioner)
            preds.extend(out)
            pbar.update(len(batch))
    except Exception:
        # Fallback: sequential with progress
        pbar.close()
        pbar = tqdm(total=len(images), desc=f"Fallback [{lang or 'pangea'}]", unit="img", leave=False)
        for img in images:
            preds.append(caption_image(image_path=img, user_prompt=user_prompt, captioner=captioner))
            pbar.update(1)
    finally:
        pbar.close()
    return preds

def caption_backend_pipeline(images: List[Any], user_prompt: str, lang: str, vs) -> List[str]:
    """Pipeline-style backend with visible per-image progress."""
    captions = []
    pbar = tqdm(total=len(images), desc=f"Captioning [{lang}] (pipeline)", unit="img", leave=False)
    for img in images:
        t0=time.perf_counter()
        retrieved = vs.retrieve(img=img, lang=lang, k=4)
        t1= time.perf_counter()
        prompt = user_prompt + " use these captions of similar images as a guidline: " + "\n".join(retrieved)
        t2= time.perf_counter()
        print(f"retrieval:{t1-t0}, captioning: {t2-t1}, total:{t2-t0}")
        captions.append(caption_image(image_path=img, user_prompt=prompt, captioner=captioner))
        pbar.update(1)
    pbar.close()
    return captions

def eval_split(
    lang: str,
    base_prompt: str,
    backend: str,
    no_lang_hint: bool,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    ds = load_dataset("neulab/PangeaBench-xm100", split=lang)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    images = [ex["image"] for ex in ds]       # PIL.Image.Image
    image_ids = [ex["image_id"] for ex in ds]
    refs = [ex["caption"] for ex in ds]

    # Per-split prompt; by default, ask for the split's language.
    user_prompt = base_prompt.strip()
    if not no_lang_hint:
        lang_name = LANG_NAME.get(lang, lang)
        user_prompt = f"{user_prompt}\nWrite the caption in {lang_name}."

    if backend == "pangea":
        preds = caption_backend_pangea(images, user_prompt, batch_size=4, lang=lang)
    elif backend == "pipeline":
        vs = FaissVectorStore('english_coco_index3')
        preds = caption_backend_pipeline(images, user_prompt, lang=lang, vs=vs)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Brief note during scoring (no granularity available inside evaluate_captions)
    tqdm.write(f"[scoring] {lang} ...")
    metrics = evaluate_captions(refs, preds, lang=lang)

    return {
        "lang": lang,
        "n": len(ds),
        "metrics": metrics,
        "image_ids": image_ids,
        "references": refs,
        "predictions": preds,
        "prompt_used": user_prompt,
    }

def evaluate():
    ap = argparse.ArgumentParser(description="Evaluate Pangea on XM-100 (consistent multilingual metrics).")
    ap.add_argument("--backend", choices=["pangea", "pipeline"], default="pangea",
                    help="Use pangea functions or (mock) HF pipeline.")
    ap.add_argument("--prompt", default="Caption the image in one concise sentence.",
                    help="Base user prompt for captioning.")
    ap.add_argument("--langs", nargs="+", default=["all"],
                    help='Language splits to evaluate (e.g., "en fr de") or "all".')
    ap.add_argument("--max_samples", type=int, default=None,
                    help="Optional limit per language (<=100).")
    ap.add_argument("--no_lang_hint", action="store_true",
                    help="If set, do not append 'Write the caption in <Language>.' to the prompt.")
    ap.add_argument("--save_csv", type=str, default=None,
                    help="Optional path to write CSV of results.")

    args = ap.parse_args()
    langs = LANG_SPLITS if (len(args.langs) == 1 and args.langs[0].lower() == "all") else args.langs

    all_rows = []
    # Progress over languages
    for lang in tqdm(langs, desc="Languages", unit="lang"):
        tqdm.write(f"\n=== Evaluating split: {lang} ===")
        res = eval_split(
            lang=lang,
            base_prompt=args.prompt,
            backend=args.backend,
            no_lang_hint=args.no_lang_hint,
            max_samples=args.max_samples,
        )
        m = res["metrics"]
        lang_full = LANG_NAME.get(lang, lang)

        line = (
            f"[metrics] {lang} ({lang_full}) backend:{args.backend}: "
            f"BLEU@4={m.get('BLEU@4', float('nan')):.4f}  "
            f"ROUGE-L={m.get('ROUGE-L', float('nan')):.4f}  "
            f"ChrF++={m.get('ChrF++', float('nan')):.4f}  "
            f"BERT-F1={m.get('bert_F1', float('nan')):.4f}"
        )

        with open("results.txt", "a", encoding="utf-8") as f:
            f.write(line + "\n")

        for iid, ref, pred in zip(res["image_ids"], res["references"], res["predictions"]):
            all_rows.append({"lang": lang, "image_id": iid, "reference": ref, "prediction": pred})
        if args.save_csv:
            pd.DataFrame(all_rows).to_csv(args.save_csv, index=False)
            tqdm.write(f"[info] wrote CSV to {args.save_csv}")

    if args.save_csv:
        pd.DataFrame(all_rows).to_csv(args.save_csv, index=False)
        tqdm.write(f"[info] wrote CSV to {args.save_csv}")


