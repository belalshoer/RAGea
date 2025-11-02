from typing import List, Dict
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import wordpunct_tokenize
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sacrebleu.metrics import CHRF


# Languages where char-level tokenization is safer
CJK_LIKE = {"zh", "ja", "ko", "th"}

def _char_tokens(text: str) -> List[str]:
    return list(text.replace(" ", ""))

def _word_tokens(text: str) -> List[str]:
    return wordpunct_tokenize(text)

class _TokenizerAdapter:
    """Adapt a tokenization fn to what RougeScorer(tokenizer=...) expects."""
    def __init__(self, fn): self._fn = fn
    def tokenize(self, text: str): return self._fn(text)

def _tokens_for_bleu(text: str, lang: str) -> List[str]:
    return _char_tokens(text) if (lang or "en").lower() in CJK_LIKE else _word_tokens(text)

def _tokenizer_for_rouge(lang: str):
    return _TokenizerAdapter(_char_tokens) if (lang or "en").lower() in CJK_LIKE else _TokenizerAdapter(_word_tokens)

def evaluate_captions(
    references: List[str],
    hypotheses: List[str],
    lang: str = "en",
) -> Dict[str, float]:
    """
    Compute the same metrics for any language: BLEU@4, ROUGE-L, ChrF++, BERTScore.
    """
    assert len(references) == len(hypotheses), "Number of references and hypotheses must match"

    # BERTScore: single multilingual backbone for language-agnostic behavior
    P, R, F1 = bert_score(
        cands=hypotheses,
        refs=references,
        model_type="xlm-roberta-large",
        verbose=False
    )

    # ROUGE-L: English stemming only for 'en'; char-tokenizer for CJK/Thai
    rouge = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=((lang or "en").lower() == "en"),
        tokenizer=_tokenizer_for_rouge(lang)
    )

    # BLEU@4 (tokenized + smoothing)
    smooth_fn = SmoothingFunction().method1
    bleu_scores, rouge_scores = [], []

    for ref, hypo in zip(references, hypotheses):
        ref_tok = _tokens_for_bleu(ref, lang)
        hypo_tok = _tokens_for_bleu(hypo, lang)

        bleu = sentence_bleu(
            [ref_tok],
            hypo_tok,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smooth_fn
        )
        bleu_scores.append(bleu)

        rouge_l = rouge.score(ref, hypo)["rougeL"].fmeasure
        rouge_scores.append(rouge_l)

    # ChrF++ (character F + word n-grams); returns percent -> normalize to 0..1
    chrf = CHRF(word_order=2)
    chrf_val = chrf.corpus_score(hypotheses, [references]).score / 100.0

    return {
        "BLEU@4": float(np.mean(bleu_scores)),
        "ROUGE-L": float(np.mean(rouge_scores)),
        "ChrF++": float(chrf_val),
        "bert_P": float(P.mean()),
        "bert_R": float(R.mean()),
        "bert_F1": float(F1.mean()),
    }

# ------------------ Example ------------------ #
if __name__ == "__main__":
    from argparse import ArgumentParser
    import pandas as pd
    parser =ArgumentParser()
    parser.add_argument("csv_file")


    results=pd.read_csv("results.csv")
    langs= results.lang.unique()
    LANG_NAME = {
    "ar":"Arabic","bn":"Bengali","cs":"Czech","da":"Danish","de":"German","el":"Greek",
    "en":"English","es":"Spanish","fa":"Persian","fi":"Finnish","fil":"Filipino",
    "fr":"French","hi":"Hindi","hr":"Croatian","hu":"Hungarian","id":"Indonesian",
    "it":"Italian","he":"Hebrew","ja":"Japanese","ko":"Korean","mi":"Māori","nl":"Dutch",
    "no":"Norwegian","pl":"Polish","pt":"Portuguese","quz":"Quechua","ro":"Romanian",
    "ru":"Russian","sv":"Swedish","sw":"Swahili","te":"Telugu","th":"Thai","tr":"Turkish",
    "uk":"Ukrainian","vi":"Vietnamese","zh":"Chinese"
}
    
    for lang in langs:
        
        refs=results[results["lang"]==lang].reference.tolist()
        hyps=results[results["lang"]==lang].prediction.tolist()
        metrics=evaluate_captions(refs, hyps, lang=lang)
        line=LANG_NAME[lang]+str(metrics)
        with open("results.txt", "a", encoding="utf-8") as f:
            f.write(line + "\n")   

