"""A controlled corpus where every attribute is exact and balanced BY CONSTRUCTION.

WHY BUILD RATHER THAN LABEL
  Two claim audits in this project were undermined by their corpus, not their method:
  guessed thresholds gave a 98% majority class, and a "nationality-free" corpus turned
  out to be written in British orthography. Both are labelling failures.

  GECOBench (arXiv 2406.11547) takes the alternative: construct text so that ground
  truth follows from how it was generated. Applied here, every attribute is exactly
  known and balanced 50/50, so no threshold is guessed and no base rate is assumed.

ATTRIBUTES, each varied independently
  language      English / French / German / Spanish — tests language and nationality
                claims, which the AV makes freely
  question      contains interrogatives, or none
  digits        contains numerals, or none
  caps          contains ALL-CAPS words, or none
  person        first person or third person
  code          contains a code fragment, or none

Every passage is assembled from per-attribute clauses, so a passage's attribute vector
is a fact about its construction rather than a judgement about its content.
"""

from __future__ import annotations

import itertools

import numpy as np

# Per-language clause banks. Content is deliberately parallel across languages so that
# language is the only thing varying when the language attribute flips.
_BODY = {
    "english": ["The committee reviewed the proposal during the afternoon session.",
                "The results were filed with the regional office before the deadline.",
                "The equipment was moved to the storage area near the loading bay."],
    "french": ["Le comité a examiné la proposition pendant la séance de l'après-midi.",
               "Les résultats ont été déposés au bureau régional avant la date limite.",
               "L'équipement a été déplacé vers la zone de stockage près du quai."],
    "german": ["Der Ausschuss prüfte den Vorschlag während der Nachmittagssitzung.",
               "Die Ergebnisse wurden vor der Frist beim Regionalbüro eingereicht.",
               "Die Ausrüstung wurde in den Lagerbereich neben der Laderampe gebracht."],
    "spanish": ["El comité revisó la propuesta durante la sesión de la tarde.",
                "Los resultados se presentaron en la oficina regional antes del plazo.",
                "El equipo fue trasladado a la zona de almacenamiento junto al muelle."],
    "italian": ["Il comitato ha esaminato la proposta durante la sessione pomeridiana.",
                "I risultati sono stati depositati all'ufficio regionale entro il termine.",
                "L'attrezzatura è stata spostata nell'area di stoccaggio vicino al molo."],
    "portuguese": ["O comité analisou a proposta durante a sessão da tarde.",
                   "Os resultados foram entregues no escritório regional antes do prazo.",
                   "O equipamento foi movido para a área de armazenamento junto ao cais."],
    # non-Latin script: if language reporting is script-driven rather than
    # language-driven, this is where the two come apart
    "japanese": ["委員会は午後の会議で提案を検討しました。",
                 "結果は締め切り前に地域事務所に提出されました。",
                 "機材は積み込み場の近くの保管区域に移動されました。"],
}
_QUESTION = {"english": "Who signed off on this, and when did they do it?",
             "french": "Qui a approuvé cela, et quand l'a-t-il fait ?",
             "german": "Wer hat das genehmigt, und wann geschah das?",
             "spanish": "¿Quién aprobó esto, y cuándo lo hizo?", "italian": "Chi ha approvato questo, e quando l'ha fatto?",
             "portuguese": "Quem aprovou isto, e quando o fez?",
             "japanese": "誰がこれを承認し、いつ行われましたか。"}
_DIGITS = {"english": "The total came to 47 units across 3 separate shipments in 2019.",
           "french": "Le total s'élevait à 47 unités réparties en 3 envois en 2019.",
           "german": "Insgesamt waren es 47 Einheiten in 3 Sendungen im Jahr 2019.",
           "spanish": "El total fue de 47 unidades en 3 envíos distintos en 2019.", "italian": "Il totale era di 47 unità in 3 spedizioni separate nel 2019.",
           "portuguese": "O total foi de 47 unidades em 3 remessas em 2019.",
           "japanese": "合計は2019年に3回の出荷で47ユニットでした。"}
_CAPS = {"english": "This is URGENT and requires IMMEDIATE attention.",
         "french": "Ceci est URGENT et exige une attention IMMEDIATE.",
         "german": "Dies ist DRINGEND und erfordert SOFORTIGE Aufmerksamkeit.",
         "spanish": "Esto es URGENTE y requiere atención INMEDIATA.", "italian": "Questo è URGENTE e richiede attenzione IMMEDIATA.",
         "portuguese": "Isto é URGENTE e exige atenção IMEDIATA.",
         "japanese": "これはURGENTであり、IMMEDIATEな対応が必要です。"}
_FIRST = {"english": "I checked the records myself and I kept my own copy.",
          "french": "J'ai vérifié les dossiers moi-même et j'ai gardé ma copie.",
          "german": "Ich habe die Unterlagen selbst geprüft und meine Kopie behalten.",
          "spanish": "Yo revisé los registros y guardé mi propia copia.", "italian": "Ho controllato io stesso i registri e ho tenuto la mia copia.",
          "portuguese": "Eu verifiquei os registos e guardei a minha cópia.",
          "japanese": "私は自分で記録を確認し、自分の写しを保管しました。"}
_THIRD = {"english": "The clerk checked the records and retained a duplicate copy.",
          "french": "Le greffier a vérifié les dossiers et conservé une copie.",
          "german": "Der Sachbearbeiter prüfte die Unterlagen und behielt eine Kopie.",
          "spanish": "El secretario revisó los registros y conservó una copia.", "italian": "L'impiegato ha controllato i registri e conservato una copia.",
          "portuguese": "O funcionário verificou os registos e guardou uma cópia.",
          "japanese": "担当者は記録を確認し、写しを保管しました。"}
_CODE = "def total(items): return sum(x.units for x in items if x.valid)"

ATTRS = ("language", "question", "digits", "caps", "person", "code")


def build(seed: int = 0, reps: int = 1) -> tuple[list[str], list[dict]]:
    """Return passages and their exact attribute vectors.

    Language cycles across the binary combinations so it is balanced against every
    other attribute rather than confounded with any of them.
    """
    rng = np.random.default_rng(seed)
    langs = list(_BODY)
    texts, labels = [], []
    combos = list(itertools.product([0, 1], repeat=5)) * reps   # 32 per rep
    for i, (q, d, c, p, co) in enumerate(combos):
        lang = langs[i % len(langs)]   # cycles, so language stays balanced
        parts = [_BODY[lang][i % len(_BODY[lang])]]
        if q:
            parts.append(_QUESTION[lang])
        if d:
            parts.append(_DIGITS[lang])
        if c:
            parts.append(_CAPS[lang])
        parts.append(_FIRST[lang] if p else _THIRD[lang])
        if co:
            parts.append(_CODE)
        rng.shuffle(parts)
        texts.append(" ".join(parts))
        labels.append({"language": lang, "question": bool(q), "digits": bool(d),
                       "caps": bool(c), "person": "first" if p else "third",
                       "code": bool(co)})
    return texts, labels


# What counts as the readout naming each attribute value. Generous on recall; the
# shuffled control is what guards against a matcher that fires indiscriminately.
CUES: dict[str, dict[str, tuple[str, ...]]] = {
    "language": {
        "english": ("english",),
        "italian": ("italian", "italiano", "italy"),
        "portuguese": ("portuguese", "português", "portugal", "brazil"),
        "japanese": ("japanese", "japan", "kanji", "cjk", "nihongo"),
        "french": ("french", "français", "france"),
        "german": ("german", "deutsch", "germany"),
        "spanish": ("spanish", "español", "castilian", "spain"),
    },
    "question": {True: ("question", "interrogative", "asking", "query", "?"),
                 False: ("statement", "declarative", "assertion")},
    "digits": {True: ("number", "numeral", "digit", "figure", "quantit", "count"),
               False: ()},
    "caps": {True: ("caps", "capital", "uppercase", "shouting", "emphatic"),
             False: ()},
    "person": {"first": ("first-person", "first person", "personal account", "narrator"),
               "third": ("third-person", "third person", "impersonal")},
    "code": {True: ("code", "function", "python", "programming", "snippet", "def "),
             False: ()},
}


def names(readout: str, attr: str, value) -> bool:
    return any(c in (readout or "").lower() for c in CUES[attr].get(value, ()))
