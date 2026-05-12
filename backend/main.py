"""
Hybrid NLP Framework for Automatic Grammar Error Detection, Analysis,
and Correction Using Rule-Based and Transformer Algorithms
FastAPI Backend — main.py
"""

import os
import re
import math
import time
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Lazy-loaded heavy singletons ──────────────────────────────────────────────
_spacy_nlp     = None
_lang_tool     = None
_t5_tokenizer  = None
_t5_model      = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NLP Grammar Analysis API",
    description="Hybrid NLP Framework — Rule-Based + Transformer Grammar Correction",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # In production, restrict to your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    text: str


class AnalyzeResponse(BaseModel):
    module1: dict
    module2: dict
    module3: dict
    module4: dict


# ── Model loaders (singleton pattern) ─────────────────────────────────────────
def get_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        try:
            _spacy_nlp = spacy.load("en_core_web_sm")
            log.info("spaCy model loaded: en_core_web_sm")
        except OSError:
            log.warning("en_core_web_sm not found — downloading now…")
            from spacy.cli import download
            download("en_core_web_sm")
            _spacy_nlp = spacy.load("en_core_web_sm")
    return _spacy_nlp


def get_lang_tool():
    global _lang_tool
    if _lang_tool is None:
        import language_tool_python
        log.info("Starting LanguageTool JVM (first launch may take ~30 s)…")
        _lang_tool = language_tool_python.LanguageTool("en-US")
        log.info("LanguageTool ready.")
    return _lang_tool


def get_t5():
    global _t5_tokenizer, _t5_model
    if _t5_model is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        model_name = "vennify/t5-base-grammar-correction"
        log.info(f"Loading T5 model: {model_name} (downloads ~900 MB on first run)…")
        _t5_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _t5_model     = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        _t5_model.eval()
        log.info("T5 grammar-correction model loaded.")
    return _t5_tokenizer, _t5_model


# ── Startup: pre-warm spaCy + NLTK (fast); T5 loads on first request) ─────────
@app.on_event("startup")
async def startup_event():
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    get_spacy()
    log.info("Startup complete. T5 model will load on first /analyze request.")


# ── MODULE 1: Preprocessing & Linguistic Analysis ─────────────────────────────
def run_module1(text: str) -> dict:
    nlp = get_spacy()
    doc = nlp(text)

    sentences = list(doc.sents)
    tokens_data = []
    stop_count  = 0
    unique_words = set()

    for token in doc:
        if token.is_space:
            continue
        if token.is_stop:
            stop_count += 1
        if token.is_alpha:
            unique_words.add(token.lower_)

        pos_map = {
            "NOUN": "NOUN", "PROPN": "PROPN", "VERB": "VERB", "AUX": "AUX",
            "ADJ": "ADJ", "ADV": "ADV", "PRON": "PRON", "DET": "DET",
            "ADP": "ADP", "CCONJ": "CCONJ", "SCONJ": "SCONJ",
            "PUNCT": "PUNCT", "NUM": "NUM", "PART": "PART",
            "INTJ": "INTJ", "X": "X", "SPACE": "SPACE",
        }
        tokens_data.append({
            "word":   token.text,
            "pos":    pos_map.get(token.pos_, token.pos_),
            "lemma":  token.lemma_,
            "isStop": token.is_stop,
            "dep":    token.dep_,
        })

    ner_data = [
        {"text": ent.text, "label": ent.label_}
        for ent in doc.ents
    ]

    word_tokens   = [t for t in tokens_data if t["pos"] not in ("PUNCT", "SPACE")]
    total_words   = len(word_tokens)
    total_sents   = len(sentences)
    avg_sent_len  = round(total_words / max(total_sents, 1), 1)

    summary = (
        f"The text contains {total_words} words across {total_sents} sentence(s). "
        f"{len(unique_words)} unique words were identified, "
        f"with {stop_count} stop words filtered out. "
        f"Average sentence length: {avg_sent_len} words."
    )
    if ner_data:
        labels = list({e["label"] for e in ner_data})
        summary += f" Named entities detected: {', '.join(labels)}."

    return {
        "totalWords":        total_words,
        "totalSentences":    total_sents,
        "totalTokens":       len(tokens_data),
        "stopWords":         stop_count,
        "uniqueWords":       len(unique_words),
        "avgSentenceLength": avg_sent_len,
        "tokens":            tokens_data[:25],   # cap at 25 for UI
        "ner":               ner_data,
        "summary":           summary,
    }


# ── MODULE 2: Rule-Based Error Detection ──────────────────────────────────────
_ERROR_TYPE_MAP = {
    "GRAMMAR":          "Grammar",
    "TYPOS":            "Spelling",
    "PUNCTUATION":      "Punctuation",
    "CASING":           "Capitalization",
    "STYLE":            "Style",
    "TYPOGRAPHY":       "Typography",
    "CONFUSED_WORDS":   "Word Choice",
}

_CATEGORY_KEYS = {
    "Subject-Verb Agreement": ["AGREEMENT", "SVA", "SUBJECT_VERB"],
    "Tense Error":            ["TENSE", "VERB_TENSE"],
    "Article Mistake":        ["A_VS_AN", "ARTICLE", "DET_"],
    "Spelling":               ["SPELL", "TYPO"],
    "Punctuation":            ["PUNCTUATION", "COMMA", "PERIOD"],
    "Capitalization":         ["CASING", "UPPER", "LOWER"],
    "Word Choice":            ["CONFUSED", "WRONG_WORD"],
    "Other":                  [],
}


def _categorize(rule_id: str, category: str) -> str:
    key = f"{rule_id}||{category}".upper()
    for cat_name, keywords in _CATEGORY_KEYS.items():
        if any(kw in key for kw in keywords):
            return cat_name
    return "Other"


def _count_categories(errors: list) -> dict:
    counts = {k: 0 for k in _CATEGORY_KEYS}
    for e in errors:
        counts[e["type"]] = counts.get(e["type"], 0) + 1
    return {
        "subjectVerb": counts.get("Subject-Verb Agreement", 0),
        "tense":       counts.get("Tense Error", 0),
        "article":     counts.get("Article Mistake", 0),
        "spelling":    counts.get("Spelling", 0),
        "punctuation": counts.get("Punctuation", 0),
        "other":       sum(
            v for k, v in counts.items()
            if k not in ("Subject-Verb Agreement","Tense Error","Article Mistake","Spelling","Punctuation")
        ),
    }


def run_module2(text: str) -> dict:
    tool   = get_lang_tool()
    matches = tool.check(text)

    errors = []
    seen   = set()

    for m in matches:
        # Skip false-positive "whitespace" rules
        if m.ruleId in ("WHITESPACE_RULE", "SENTENCE_WHITESPACE"):
            continue

        wrong = text[m.offset: m.offset + m.errorLength]
        if not wrong.strip():
            continue

        fix = m.replacements[0] if m.replacements else wrong
        key = f"{wrong}|{fix}"
        if key in seen:
            continue
        seen.add(key)

        err_type = _categorize(m.ruleId, m.category)

        errors.append({
            "type":        err_type,
            "wrongText":   wrong,
            "suggestedFix": fix,
            "explanation": m.message or f"Possible {err_type.lower()} issue detected.",
        })

    return {
        "totalErrors":      len(errors),
        "errors":           errors,
        "errorCategories":  _count_categories(errors),
    }


# ── MODULE 3: Transformer-Based Correction ────────────────────────────────────
def _correct_sentence(tokenizer, model, sentence: str) -> tuple[str, float]:
    import torch

    prefix  = "grammar: "
    inputs  = tokenizer(
        prefix + sentence,
        return_tensors="pt",
        max_length=256,
        truncation=True,
        padding=True,
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=4,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True,
        )

    corrected = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    # Rough confidence from beam scores
    try:
        import torch.nn.functional as F
        scores     = outputs.scores         # list of (1, vocab) tensors
        log_probs  = [F.log_softmax(s, dim=-1).max().item() for s in scores]
        confidence = round(math.exp(sum(log_probs) / max(len(log_probs), 1)) * 100, 1)
        confidence = max(50.0, min(99.0, confidence))
    except Exception:
        confidence = 85.0

    return corrected.strip(), confidence


def run_module3(text: str) -> dict:
    tokenizer, model = get_t5()

    import spacy
    nlp = get_spacy()
    doc = nlp(text)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]

    comparisons  = []
    all_scores   = []
    corrected_parts = []

    for sent in sentences:
        corrected, conf = _correct_sentence(tokenizer, model, sent)
        all_scores.append(conf)

        change_type = "No change"
        if corrected.lower() != sent.lower():
            # Detect dominant change type
            orig_words = sent.split()
            corr_words = corrected.split()
            if len(orig_words) != len(corr_words):
                change_type = "Structural Rewrite"
            else:
                diffs = sum(1 for a, b in zip(orig_words, corr_words) if a != b)
                change_type = "Minor Fix" if diffs <= 2 else "Multiple Corrections"

        comparisons.append({
            "original":  sent,
            "corrected": corrected,
            "changeType": change_type,
        })
        corrected_parts.append(corrected)

    corrected_text   = " ".join(corrected_parts)
    avg_confidence   = round(sum(all_scores) / max(len(all_scores), 1), 1)
    changed          = sum(1 for c in comparisons if c["changeType"] != "No change")

    if changed == 0:
        fluency = "The text is already grammatically well-formed. No significant corrections were required."
    elif changed == len(sentences):
        fluency = f"All {len(sentences)} sentence(s) were corrected for grammar, tense, and fluency."
    else:
        fluency = f"{changed} of {len(sentences)} sentence(s) required corrections for improved grammatical fluency."

    return {
        "correctedText":       corrected_text,
        "confidenceScore":     avg_confidence,
        "sentenceComparisons": comparisons,
        "fluencyImprovement":  fluency,
    }


# ── MODULE 4: Analytics & Feedback ────────────────────────────────────────────
def _readability_score(text: str) -> float:
    """Flesch Reading Ease (0–100, higher = easier)."""
    words      = re.findall(r"\b\w+\b", text)
    sentences  = re.split(r"[.!?]+", text)
    sentences  = [s for s in sentences if s.strip()]
    syllables  = sum(_count_syllables(w) for w in words)

    if not words or not sentences:
        return 50.0

    asl  = len(words) / len(sentences)          # avg sentence length
    asw  = syllables / len(words)               # avg syllables per word
    fre  = 206.835 - 1.015 * asl - 84.6 * asw
    return round(max(0.0, min(100.0, fre)), 1)


def _count_syllables(word: str) -> int:
    word = word.lower()
    count = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _writing_level(grammar_score: float, readability: float) -> str:
    avg = (grammar_score + readability) / 2
    if avg >= 85:  return "Advanced"
    if avg >= 70:  return "Upper-Intermediate"
    if avg >= 55:  return "Intermediate"
    if avg >= 40:  return "Elementary"
    return "Beginner"


def run_module4(text: str, module2: dict, module3: dict) -> dict:
    total_errors  = module2["totalErrors"]
    word_count    = max(len(re.findall(r"\b\w+\b", text)), 1)

    # Grammar score: penalise 3 pts per error, capped at 0
    grammar_score    = max(0.0, round(100 - (total_errors * 3 * (10 / word_count) * 10), 1))
    readability      = _readability_score(text)

    # Clarity: blend of unique-word ratio and sentence-length variance
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    words     = re.findall(r"\b\w+\b", text)
    unique_r  = len(set(w.lower() for w in words)) / max(len(words), 1)
    lens      = [len(re.findall(r"\b\w+\b", s)) for s in sentences]
    variance  = (sum((l - (sum(lens)/max(len(lens),1)))**2 for l in lens)
                 / max(len(lens), 1)) if lens else 0
    clarity_score = round(max(0, min(100, 50 + unique_r * 30 - variance * 0.5)), 1)

    # Error distribution with colour codes
    cat = module2["errorCategories"]
    dist_map = {
        "Subject-Verb":  {"count": cat["subjectVerb"], "color": "#fc8181"},
        "Tense Error":   {"count": cat["tense"],       "color": "#f6ad55"},
        "Article":       {"count": cat["article"],     "color": "#9f7aea"},
        "Spelling":      {"count": cat["spelling"],    "color": "#63b3ed"},
        "Punctuation":   {"count": cat["punctuation"], "color": "#4fd1c5"},
        "Other":         {"count": cat["other"],       "color": "#a0aec0"},
    }
    error_dist = [
        {"category": k, "count": v["count"], "color": v["color"]}
        for k, v in dist_map.items()
        if v["count"] > 0
    ]

    # Most common mistake
    most_common = max(dist_map, key=lambda k: dist_map[k]["count"]) if total_errors else "None"

    # Improvements
    improvements = []
    if cat["subjectVerb"]:
        improvements.append("Ensure subjects and verbs agree in number (e.g., 'She is' not 'She are').")
    if cat["tense"]:
        improvements.append("Maintain consistent verb tenses throughout the text.")
    if cat["article"]:
        improvements.append("Use articles (a/an/the) correctly based on noun definiteness.")
    if cat["spelling"]:
        improvements.append("Run a spell-check pass before submission.")
    if cat["punctuation"]:
        improvements.append("Review sentence-ending punctuation and comma placement.")
    if not improvements:
        improvements.append("Great work! Focus on expanding vocabulary and sentence variety.")
    improvements.append("Read sentences aloud to catch unnatural phrasing.")
    improvements.append("Use active voice where possible for clearer communication.")

    level = _writing_level(grammar_score, readability)

    changed = sum(1 for c in module3["sentenceComparisons"] if c["changeType"] != "No change")
    feedback_parts = [
        f"Your text was analysed across {len(sentences)} sentence(s) with {total_errors} grammar issue(s) detected.",
        f"The grammar score of {grammar_score}/100 reflects the density of errors relative to text length.",
        f"Readability score: {readability}/100 — {'easy to read' if readability > 60 else 'moderately complex' if readability > 40 else 'complex'}.",
        f"The transformer model corrected {changed} sentence(s) for improved fluency.",
        f"Overall writing level: {level}.",
    ]
    feedback = " ".join(feedback_parts)

    return {
        "grammarScore":         grammar_score,
        "readabilityScore":     readability,
        "clarityScore":         clarity_score,
        "totalErrorsDetected":  total_errors,
        "errorDistribution":    error_dist,
        "mostCommonMistake":    most_common,
        "writingLevel":         level,
        "improvements":         improvements[:5],
        "feedback":             feedback,
    }


# ── API Endpoints ──────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "running",
        "api":    "NLP Grammar Analysis API",
        "docs":   "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="'text' field must not be empty.")
    if len(text) > 5000:
        raise HTTPException(status_code=400, detail="Text too long. Maximum 5000 characters.")

    try:
        log.info(f"Analyzing text ({len(text)} chars)…")
        t0 = time.time()

        m1 = run_module1(text)
        log.info(f"  M1 done in {time.time()-t0:.2f}s")

        m2 = run_module2(text)
        log.info(f"  M2 done in {time.time()-t0:.2f}s")

        m3 = run_module3(text)
        log.info(f"  M3 done in {time.time()-t0:.2f}s")

        m4 = run_module4(text, m2, m3)
        log.info(f"  M4 done in {time.time()-t0:.2f}s  [TOTAL]")

        return AnalyzeResponse(module1=m1, module2=m2, module3=m3, module4=m4)

    except Exception as e:
        log.exception("Error during analysis")
        raise HTTPException(status_code=500, detail=str(e))


# ── Run directly ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
