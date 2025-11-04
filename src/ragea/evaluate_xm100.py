import argparse
from typing import Any, Dict, List, Optional
from datasets import load_dataset
import pandas as pd
from generation import Captioner
from utils import evaluate_captions 
from pathlib import Path
import os
import tqdm

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

captioner = Captioner()

def caption_backend_pangea(images: List[Any], user_prompt: str = None, lang: str = None) -> List[str]:
    
    try:
        if user_prompt:
            return captioner.caption_images(images=images, user_prompt=user_prompt, lang=lang)
        else:
            return captioner.caption_images(images=images, lang=lang)
    except Exception:
        if user_prompt:
            return [captioner.caption_image(image=img, user_prompt=user_prompt, lang=lang) for img in tqdm.tqdm(images)]
        else:
            return [captioner.caption_image(image=img, lang=lang) for img in tqdm.tqdm(images)]


def eval_xm100_split(
    lang: str,
    user_prompt: str = None,
    no_lang_hint: bool = False,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    ds = load_dataset("neulab/PangeaBench-xm100", split=lang)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    images = [ex["image"] for ex in ds]     
    image_ids = [ex["image_id"] for ex in ds]
    refs = [ex["caption"] for ex in ds]

    if not no_lang_hint:
        lang_name = LANG_NAME.get(lang, lang)
        preds = caption_backend_pangea(images, user_prompt, lang=lang_name)
    else:
        preds = caption_backend_pangea(images, user_prompt)


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

def evaluate_xm100(
    prompt: str = None, 
    langs: List[str] = "all",
    max_samples: int = None,
    no_lang_hint: bool = False,
    results_path: str = "results",
    with_retrieval: bool = False
):

    langs = LANG_SPLITS if (len(langs) == 1 and langs[0].lower() == "all") else langs
    os.makedirs("outputs", exist_ok=True)
    
    all_rows = []
    for lang in langs:
        print(f"\n=== Evaluating split: {lang} ===", flush=True)

        res = eval_xm100_split(
            lang=lang,
            user_prompt=prompt,
            no_lang_hint=no_lang_hint,
            max_samples=max_samples,
        )

        m = res["metrics"]
        lang_full = LANG_NAME.get(lang, lang)

        line = (
            f"[metrics] {lang} ({lang_full}) : "
            f"BLEU@4={m.get('BLEU@4', float('nan')):.4f}  "
            f"ROUGE-L={m.get('ROUGE-L', float('nan')):.4f}  "
            f"ChrF++={m.get('ChrF++', float('nan')):.4f}  "
            f"BERT-F1={m.get('bert_F1', float('nan')):.4f}"
        )

        with open(Path("outputs") / f"{results_path}.txt", "a", encoding="utf-8") as f:
            f.write(line + "\n")
        for iid, ref, pred in zip(res["image_ids"], res["references"], res["predictions"]):
            all_rows.append({"lang": lang, "image_id": iid, "reference": ref, "prediction": pred})


    full_path = Path("outputs") / f"{results_path}.csv"
    pd.DataFrame(all_rows).to_csv(full_path, index=False)
    print(f"[info] wrote CSV to {full_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate Pangea on XM-100 (consistent multilingual metrics).")
    ap.add_argument("--prompt", default = None,
                    help="Base user prompt for captioning.")
    ap.add_argument("--langs", nargs="+", default=["all"],
                    help='Language splits to evaluate (e.g., "en fr de") or "all".')
    ap.add_argument("--max_samples", type=int, default=None,
                    help="Optional limit per language (<=100).")
    ap.add_argument("--no_lang_hint", action="store_true",
                    help="If set, do not append 'Write the caption in <Language>.' to the prompt.")
    ap.add_argument("--results_path", type=str, default=None,
                    help="Optional name of the CSV file of results and the text file containing the evaluation metrics.")

    args = ap.parse_args()

    payload = {k: v for k, v in vars(args).items() if v is not None}
    evaluate_xm100(**payload)