"""FLORES-200 passages: real, professionally translated, parallel across languages.

WHY THIS REPLACES THE CONSTRUCTED CORPUS
  The language findings were measured on controlled_corpus, which generates 224
  passages by recombining 57 sentences I wrote myself. That is the weakest part of the
  evidence: one awkward translation could drive a whole language's result, and no
  native speaker checked any of it.

  FLORES-200 is human-translated by professionals and fully parallel — the same source
  sentences rendered in every language. So it keeps the property that made the
  constructed corpus worth building (content is matched across languages, so language
  is the only thing varying) while removing the part a reviewer would not trust.

  Served here from the mteb/flores mirror: the official facebook/flores and
  openlanguagedata/flores_plus repos are gated and no HF_TOKEN is available.

LANGUAGE SET
  The seven from the constructed corpus, so the two runs are directly comparable, plus
  five more non-Latin scripts. The old corpus had exactly one non-Latin language, which
  made "is this a script effect or a language effect" barely testable. It now has six.

USAGE
  from bench.flores_corpus import build
  texts, langs = build(n_per_language=24)
"""

from __future__ import annotations

import re

FLORES = "mteb/flores"
SPLIT = "devtest"

# column -> our language name. First seven match controlled_corpus.
LANGUAGES: dict[str, str] = {
    "eng_Latn": "english",
    "fra_Latn": "french",
    "deu_Latn": "german",
    "spa_Latn": "spanish",
    "ita_Latn": "italian",
    "por_Latn": "portuguese",
    "jpn_Jpan": "japanese",
    "rus_Cyrl": "russian",
    "arb_Arab": "arabic",
    "ell_Grek": "greek",
    "zho_Hans": "chinese",
    "kor_Hang": "korean",
}

LATIN = {"english", "french", "german", "spanish", "italian", "portuguese"}

SENTENCES_PER_PASSAGE = 2      # FLORES rows are single sentences; 2 gives ~50 words


def build(n_per_language: int = 24, split: str = SPLIT) -> tuple[list[str], list[str]]:
    """Parallel passages. Passage i of language A and of language B are translations."""
    from datasets import load_dataset
    d = load_dataset(FLORES, "default", split=split)
    texts, langs = [], []
    for col, name in LANGUAGES.items():
        rows = d[col]
        for k in range(n_per_language):
            j = k * SENTENCES_PER_PASSAGE
            chunk = [s for s in rows[j:j + SENTENCES_PER_PASSAGE] if s]
            if len(chunk) < SENTENCES_PER_PASSAGE:
                break
            texts.append(" ".join(chunk))
            langs.append(name)
    return texts, langs


# Language NAMES and endonyms only — no script terms. "Cyrillic" does not identify a
# language, and "kanji" does not separate Japanese from Chinese. Stricter than the cue
# list in controlled_corpus, which counted "kanji"/"cjk" for Japanese.
CUES: dict[str, tuple[str, ...]] = {
    "english": ("english",),
    "french": ("french", "français", "france"),
    "german": ("german", "deutsch", "germany"),
    "spanish": ("spanish", "español", "castilian", "spain"),
    "italian": ("italian", "italiano", "italy"),
    "portuguese": ("portuguese", "português", "portugal", "brazil"),
    "japanese": ("japanese", "japan", "nihongo"),
    "russian": ("russian", "русский", "russia"),
    "arabic": ("arabic", "عربي", "arab"),
    "greek": ("greek", "ελληνικ", "greece"),
    "chinese": ("chinese", "mandarin", "china"),
    "korean": ("korean", "korea"),
}

# Positive evidence that a span of text IS a given language. Scripts are decisive for
# the non-Latin set. Japanese is detected by kana and Chinese by Han-without-kana, so
# the two do not collapse into each other despite sharing Han characters.
_KANA = r"[぀-ヿ]"
_HAN = r"[一-鿿]"

MARKERS: dict[str, str] = {
    "french": r"\b(le|la|les|des|une|est|été|pour|dans|avec|sur|qui|vous|nous|ce|cette|au)\b|[àéèêçù]",
    "german": r"\b(der|die|das|und|ist|wurde|wurden|nicht|eine|einen|für|mit|auf|im|zu|den)\b|[äöüß]",
    "spanish": r"\b(el|la|los|las|una|fue|para|con|por|que|del|se|es|en|su)\b|[ñ¿¡]",
    "italian": r"\b(il|lo|gli|della|delle|è|stato|stata|per|con|che|nel|una|sono|ha)\b",
    "portuguese": r"\b(o|os|as|uma|foi|para|com|que|do|da|não|está|no|na|ao)\b|[ãõç]",
    "japanese": _KANA,
    "russian": r"[Ѐ-ӿ]",
    "arabic": r"[؀-ۿ]",
    "greek": r"[Ͱ-Ͽ]",
    "korean": r"[가-힯]",
}


def marked_as(text: str, language: str) -> bool:
    """Is this span positively marked as `language`?"""
    if language == "chinese":
        return bool(re.search(_HAN, text) and not re.search(_KANA, text))
    if language == "japanese":
        return bool(re.search(_KANA, text))
    pat = MARKERS.get(language)
    return bool(pat and re.search(pat, text, re.I))


def names(readout: str, language: str) -> bool:
    return any(c in (readout or "").lower() for c in CUES.get(language, ()))
