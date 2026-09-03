from __future__ import annotations

from dataclasses import dataclass


LSM_ALPHABET = (
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "Ñ",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
)

DYNAMIC_LETTERS = frozenset({"J", "K", "Ñ", "Q", "X", "Z"})


@dataclass(frozen=True)
class DactylologyLetter:
    letter: str
    dynamic: bool
    note: str


LETTER_METADATA = {
    letter: DactylologyLetter(
        letter=letter,
        dynamic=letter in DYNAMIC_LETTERS,
        note=(
            "Dynamic LSM dactylology letter; motion is part of the letter."
            if letter in DYNAMIC_LETTERS
            else "Static LSM dactylology letter."
        ),
    )
    for letter in LSM_ALPHABET
}

DACTYLOLOGY_METADATA = {
    "system": "LSM",
    "language": "Lengua de Señas Mexicana",
    "mode": "dactylology",
    "output_kind": "manual_spelling",
    "letter_count": len(LSM_ALPHABET),
    "dynamic_letters": tuple(letter for letter in LSM_ALPHABET if letter in DYNAMIC_LETTERS),
    "is_lexical_translation": False,
    "uses_trained_model": False,
    "description": (
        "Dactylology spells written characters with the manual alphabet; "
        "it is not lexical sign translation."
    ),
}

_LETTER_SET = frozenset(LSM_ALPHABET)


class DactylologyError(ValueError):
    pass


def normalize_character(character: str) -> str:
    """normalize character."""
    if len(character) != 1:
        raise DactylologyError("Expected exactly one character.")

    normalized = character.upper()
    if normalized not in _LETTER_SET:
        raise DactylologyError(f"Unsupported LSM dactylology character: {character!r}")

    return normalized


def is_valid_character(character: str) -> bool:
    try:
        normalize_character(character)
    except DactylologyError:
        return False
    return True


def validate_word(word: str) -> tuple[str, ...]:
    """validate word."""
    stripped = word.strip()
    if not stripped:
        raise DactylologyError("Expected a non-empty word.")

    letters = []
    for character in stripped:
        if character.isspace():
            raise DactylologyError("Dactylology word spelling does not accept spaces.")
        letters.append(normalize_character(character))

    return tuple(letters)


def spell_word(word: str, *, separator: str = "-") -> str:
    return separator.join(validate_word(word))


def describe_letter(character: str) -> DactylologyLetter:
    return LETTER_METADATA[normalize_character(character)]


def requires_motion(character: str) -> bool:
    return describe_letter(character).dynamic