"""Rule-based danger-sign detection (design section 6).

Phrasings live in data/danger_signs.yaml. Text and phrasings go through the
same normalisation, and a phrasing matches when its words appear in order with
at most MAX_GAP other words between neighbours. Missing a sign is far worse
than a false alarm, so negation only suppresses a match in the narrow patterns
handled in _negated; anything else fires.
"""

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import yaml

DATA = Path(__file__).resolve().parents[2] / "data" / "danger_signs.yaml"
LANGUAGES = ("en", "hi", "hi-Latn")

MAX_GAP = 4
# English negators scope forward over this many words: "no heavy bleeding".
NEGATION_WINDOW = 3

# A clause boundary: punctuation, or a word that starts a new clause.
_BOUNDARY = "|"
_PUNCTUATION = re.compile(r"[.,!?;:()\[\]\"/\n\u0964\u0965]")
_TOKEN = re.compile(r"[0-9a-z\u0900-\u0963\u0966-\u097f]+|\|")
_CLAUSE_WORDS = {
    "but", "and", "lekin", "par", "magar", "aur", "लेकिन", "पर", "मगर", "और",
}  # fmt: skip

_ENGLISH_NEGATORS = {
    "no", "not", "never", "without", "nothing", "none",
    "dont", "doesnt", "didnt", "isnt", "wasnt", "hasnt", "havent", "hadnt",
}  # fmt: skip
_HINDI_NEGATORS = {"nahi", "na", "mat", "नहीं", "नही", "ना", "न", "मत"}
# "khoon nahi aaya" denies the sign; "khoon nahi ruk raha" does not. A Hindi
# negator after a sign only suppresses it before one of these verbs or a
# clause end.
_DENIAL_VERBS = {
    "hai", "he", "h", "hain", "tha", "thi", "the", "hua", "hui", "hue", "ho",
    "aa", "aaya", "aayi", "aai", "aaye", "aate", "aata", "aati",
    "है", "हैं", "था", "थी", "थे", "हुआ", "हुई", "हुए", "हो",
    "आ", "आया", "आयी", "आई", "आए", "आये", "आता", "आती", "आते",
}  # fmt: skip

# Latin spellings that the generic rules in _latin do not merge.
_LATIN_VARIANTS = {
    "nhi": "nahi", "nahin": "nahi", "nai": "nahi", "nahiin": "nahi",
    "rha": "raha", "rhi": "rahi", "rhe": "rahe",
    "gyi": "gayi", "gya": "gaya",
    "sans": "sas", "svas": "sas", "shvas": "sas",
    "me": "mein", "mai": "mein", "mei": "mein", "men": "mein",
}  # fmt: skip


@dataclass(frozen=True)
class DangerSign:
    id: str
    source: str
    phrases: dict[str, list[str]]


def _latin(token: str) -> str:
    token = re.sub(r"(.)\1{2,}", r"\1\1", token)
    token = token.replace("ee", "i").replace("oo", "u")
    token = token.replace("chch", "ch").replace("cch", "ch")
    token = token.replace("w", "v").replace("z", "j").replace("ph", "f")
    token = re.sub(r"(.)\1", r"\1", token)
    return _LATIN_VARIANTS.get(token, token)


def _canonical(token: str) -> str:
    if token.isascii():
        return _latin(token)
    # Nukta and chandrabindu are often typed without, or as anusvara.
    return token.replace("\u093c", "").replace("\u0901", "\u0902")


def _tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[\u200b-\u200d\u2060\ufeff'’]", "", text)
    text = _PUNCTUATION.sub(f" {_BOUNDARY} ", text)
    tokens = [_canonical(t) if t != _BOUNDARY else t for t in _TOKEN.findall(text)]
    return [_BOUNDARY if t in _CLAUSE_WORDS else t for t in tokens]


def _phrase_tokens(phrase: str) -> list[str]:
    return [t for t in _tokens(phrase) if t != _BOUNDARY]


def _canonical_set(words: set[str]) -> set[str]:
    return {_canonical(w) for w in words}


_EN_NEG = _canonical_set(_ENGLISH_NEGATORS)
_HI_NEG = _canonical_set(_HINDI_NEGATORS)
_DENIAL = _canonical_set(_DENIAL_VERBS)


def _matches(tokens: list[str], phrase: list[str]) -> list[tuple[int, int, set[int]]]:
    """Return (first, last, positions) for each occurrence of phrase."""
    found = []
    for start, token in enumerate(tokens):
        if token != phrase[0]:
            continue
        positions = {start}
        at = start
        for word in phrase[1:]:
            gap = 0
            at += 1
            while at < len(tokens) and tokens[at] != word:
                if tokens[at] != _BOUNDARY:
                    gap += 1
                if gap > MAX_GAP:
                    break
                at += 1
            if at >= len(tokens) or tokens[at] != word:
                break
            positions.add(at)
        else:
            found.append((start, at, positions))
    return found


def _negated(tokens: list[str], first: int, last: int, positions: set[int]) -> bool:
    between = [tokens[i] for i in range(first, last + 1) if i not in positions]
    if any(t in _EN_NEG or t in _HI_NEG for t in between):
        return True
    before = []
    for t in reversed(tokens[max(0, first - NEGATION_WINDOW) : first]):
        if t == _BOUNDARY:
            break
        before.append(t)
    if any(t in _EN_NEG for t in before):
        return True
    after = tokens[last + 1 : last + 3] + [_BOUNDARY]
    return after[0] in _HI_NEG and after[1] in _DENIAL | {_BOUNDARY}


def detect(text: str) -> list[str]:
    """Return the sorted ids of the danger signs mentioned in text."""
    tokens = _tokens(text)
    return sorted(
        sign.id
        for sign, phrases in _COMPILED
        if any(
            not _negated(tokens, first, last, positions)
            for phrase in phrases
            for first, last, positions in _matches(tokens, phrase)
        )
    )


def _load() -> tuple[list[DangerSign], dict[str, str]]:
    data = yaml.safe_load(DATA.read_text(encoding="utf-8"))
    signs = [DangerSign(**sign) for sign in data["signs"]]
    for sign in signs:
        if set(sign.phrases) != set(LANGUAGES):
            raise ValueError(f"{DATA}: {sign.id} needs phrases in {LANGUAGES}")
    return signs, data["urgent_reply"]


SIGNS, URGENT_REPLY = _load()
_COMPILED = [
    (sign, [_phrase_tokens(p) for ps in sign.phrases.values() for p in ps])
    for sign in SIGNS
]
