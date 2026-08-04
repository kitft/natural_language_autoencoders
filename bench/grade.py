"""Grading a free-text readout against a target emotion.

Three graders, deliberately layered weakest-to-strongest:

  substring  — the scaffold's `target in readout.lower()`. Kept only as a
               reference point; it misses "furious" for anger and matches
               "surprisingly" for surprise.
  lexicon    — word-boundary match against a curated synonym set. Deterministic,
               reproducible by anyone without an API key, and the default.
  llm        — `claude -p` judge (see bench/judge.py). Strongest, but requires
               working Claude Code auth, so it must never be the only grader.

A faithfulness claim resting on a single LLM judge is a soft claim, so the
released benchmark reports the lexicon grader as primary and the judge as
corroboration.
"""

from __future__ import annotations

import re
from typing import Iterable

# Word-boundary matched, case-insensitive. Includes inflections explicitly rather
# than stemming — stemmers conflate "confused"/"confusing" with "confuse" but also
# collapse pairs we want kept apart.
EMOTION_LEXICON: dict[str, tuple[str, ...]] = {
    # Expanded after an audit found readouts scored zero while plainly naming the
    # emotion periphrastically — "suggesting a FRUSTRATED or sarcastic comment" was
    # a miss for anger. Additions are near-synonyms only. Bare generics ("question",
    # "ask", "interest", "upset", "positive") are deliberately EXCLUDED: they would
    # inflate curiosity and confusion, our weakest emotions, and cross-fire between
    # anger and sadness. The alpha=0 false-positive rate is the check on whether the
    # expansion went too far.
    #
    # Profanity is included for anger specifically: steered generations express rage
    # through swearing far more often than through the word "angry".
    "anger": ("anger", "angry", "angered", "furious", "fury", "rage", "enraged",
              "irate", "livid", "outrage", "outraged", "seething", "incensed",
              "indignant", "resentful", "hostile", "irritated", "mad",
              "fuck", "fucking", "damn", "hell", "bullshit", "pissed",
              "frustrated", "frustration", "annoyed", "annoyance", "exasperated",
              "exasperation", "fuming", "aggravated", "irked", "wrath", "temper",
              "resentment", "irritation", "ranting", "rant"),
    "joy": ("joy", "joyful", "joyous", "happy", "happiness", "delight", "delighted",
            "elated", "cheerful", "glad", "thrilled", "jubilant", "blissful",
            "upbeat", "gleeful", "elation", "pleased", "ecstatic", "overjoyed",
            "merry", "celebratory", "exuberant"),
    "sadness": ("sad", "sadness", "sorrow", "sorrowful", "unhappy", "miserable",
                "melancholy", "grief", "grieving", "depressed", "despondent",
                "downcast", "heartbroken", "mournful", "gloomy", "despair",
                "dejected", "forlorn", "glum", "tearful", "crying", "weeping",
                "disheartened", "sombre", "somber", "wistful"),
    "fear": ("fear", "fearful", "afraid", "scared", "terrified", "terror",
             "frightened", "fright", "anxious", "anxiety", "panic", "dread",
             "alarmed", "apprehensive", "nervous", "worried", "terrifying",
             "scary", "frightening", "uneasy", "trepidation", "foreboding",
             "petrified", "worry", "panicked"),
    "disgust": ("disgust", "disgusted", "disgusting", "revolting", "revulsion",
                "repulsed", "repulsive", "gross", "nauseated", "nauseating",
                "sickened", "distaste", "loathing", "abhorrent", "repugnant",
                "repelled", "revolted", "vile", "sickening", "squeamish",
                "distasteful", "grossed out"),
    "surprise": ("surprise", "surprised", "surprising", "astonished",
                 "astonishment", "shocked", "shock", "amazed", "amazement",
                 "startled", "stunned", "unexpected", "dumbfounded", "astounded",
                 "taken aback", "incredulous", "disbelief", "shocking",
                 "unexpectedly"),
    "gratitude": ("grateful", "gratitude", "thankful", "thanks", "thank",
                  "appreciative", "appreciation", "indebted", "thankfulness",
                  "gratefully", "appreciate", "appreciated"),
    "love": ("love", "loving", "loved", "affection", "affectionate", "adoration",
             "adore", "fondness", "devoted", "devotion", "romantic", "tenderness",
             "beloved", "adoring", "cherish", "infatuation", "romance", "endearing"),
    "amusement": ("amused", "amusement", "amusing", "funny", "hilarious", "humor",
                  "humour", "humorous", "laugh", "laughter", "comic", "comical",
                  "witty", "entertaining", "giggle", "giggling", "chuckle",
                  "joking", "lighthearted", "playful", "laughing", "comedy",
                  "jokey", "banter"),
    "admiration": ("admiration", "admire", "admiring", "impressed", "impressive",
                   "awe", "respect", "esteem", "praise", "remarkable",
                   "extraordinary", "commendable", "admirable", "applaud",
                   "laudable", "reverence", "kudos", "praising", "complimentary"),
    # No bare "question"/"ask"/"interest": those fire on any interrogative text and
    # would manufacture recovery for our weakest emotion.
    "curiosity": ("curious", "curiosity", "intrigued", "intriguing", "inquisitive",
                  "wondering", "questioning", "exploratory", "curiousness",
                  "inquisitiveness", "intrigue", "inquiring"),
    "confusion": ("confused", "confusion", "confusing", "puzzled", "puzzling",
                  "baffled", "bewildered", "perplexed", "unclear", "disoriented",
                  "muddled", "puzzlement", "bewilderment", "unsure", "uncertain",
                  "misunderstanding", "ambiguous", "nonplussed"),
}

_COMPILED = {
    emotion: re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b",
                        re.IGNORECASE)
    for emotion, words in EMOTION_LEXICON.items()
}


def is_degenerate(text: str, *, min_distinct_ratio: float = 0.5,
                  max_ngram_repeats: int = 4, n: int = 4) -> bool:
    """Has generation collapsed into a repetition loop?

    Steering at high alpha swamps the residual stream and produces output like
    "Thank!Thank!Great!Thank!..." — which scores enormously on a raw lexicon count
    while containing no coherent emotional prose. Any steering metric that does
    not exclude these is measuring degeneration, not steering.
    """
    words = (text or "").split()
    if len(words) < n + 1:
        return True
    if len(set(words)) / len(words) < min_distinct_ratio:
        return True
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return max(grams.count(g) for g in set(grams)) > max_ngram_repeats


def lexicon_hits(text: str, emotion: str) -> int:
    """How many target-emotion lexemes appear in `text`."""
    return len(_COMPILED[emotion].findall(text or ""))


def grade_readout(readout: str, target_emotion: str, method: str = "lexicon") -> bool:
    """Does `readout` name the target emotion?"""
    if method == "substring":
        return target_emotion.lower() in (readout or "").lower()
    if method == "lexicon":
        return lexicon_hits(readout, target_emotion) > 0
    raise ValueError(f"unknown grading method {method!r}; use bench.judge for 'llm'")


def dominant_emotion(text: str, candidates: Iterable[str] | None = None) -> str | None:
    """Emotion with the most lexicon hits, or None if nothing matches.

    Ties resolve to None rather than an arbitrary pick — a tie genuinely means
    the text does not single out one emotion, and silently choosing would inflate
    apparent specificity.
    """
    cands = list(candidates) if candidates is not None else list(EMOTION_LEXICON)
    scored = [(lexicon_hits(text, e), e) for e in cands]
    best = max(s for s, _ in scored)
    if best == 0:
        return None
    top = [e for s, e in scored if s == best]
    return top[0] if len(top) == 1 else None
