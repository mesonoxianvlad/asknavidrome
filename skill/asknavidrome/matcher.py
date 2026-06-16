"""Voice-friendly matching helpers for music names.

The matcher is intentionally conservative: it performs exact comparisons
after canonicalization, not fuzzy or partial matching. It also avoids arbitrary
word compaction, custom aliases, and configuration files.
"""

import re
from typing import NamedTuple
import unicodedata


VARIANT_LIMIT = 25

_UNIT_NUMBER_WORDS = {
    'zero': 0,
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10,
    'eleven': 11,
    'twelve': 12,
    'thirteen': 13,
    'fourteen': 14,
    'fifteen': 15,
    'sixteen': 16,
    'seventeen': 17,
    'eighteen': 18,
    'nineteen': 19,
}

_TENS_NUMBER_WORDS = {
    'twenty': 20,
    'thirty': 30,
    'forty': 40,
    'fifty': 50,
    'sixty': 60,
    'seventy': 70,
    'eighty': 80,
    'ninety': 90,
}

_EQUIVALENCE_GROUPS = (
    ('2', 'two', 'to', 'too'),
    ('4', 'four', 'for'),
    ('8', 'eight', 'ate'),
    ('u', 'you'),
    ('r', 'are'),
    ('b', 'bee', 'be'),
    ('c', 'see', 'sea'),
    ('and', 'n'),
)

_CANONICAL_BY_TOKEN = {
    token: group[0]
    for group in _EQUIVALENCE_GROUPS
    for token in group
}

_VARIANTS_BY_CANONICAL = {
    group[0]: group
    for group in _EQUIVALENCE_GROUPS
}

_COMPACTABLE_LETTERS = {'b', 'c', 'r', 'u'}


class _ParsedNumber(NamedTuple):
    value: int
    consumed: int


def normalize_text(text: str) -> str:
    """Return a lower-case, punctuation-free representation of text."""
    if text is None:
        return ''

    normalized = unicodedata.normalize('NFKD', str(text))
    ascii_text = normalized.encode('ascii', 'ignore').decode('ascii')
    lower_text = ascii_text.lower().replace('&', ' and ')
    words_only = re.sub(r'[^a-z0-9]+', ' ', lower_text)

    return re.sub(r'\s+', ' ', words_only).strip()


def match_key(text: str) -> str:
    """Return the canonical key used for exact voice-aware comparisons."""
    return ' '.join(_compact(_canonicalize(_tokenize(text))))


def names_match(query: str, candidate: str) -> bool:
    """Return True when two names match after voice-aware normalization."""
    query_key = match_key(query)
    candidate_key = match_key(candidate)

    return bool(query_key) and query_key == candidate_key


def query_variants(term: str) -> list:
    """Generate deterministic search terms for common Alexa mishearings."""
    tokens = _tokenize(term)
    if not tokens:
        return []

    canonical_tokens = _canonicalize(tokens)
    compact_tokens = _compact(canonical_tokens)

    variants = []
    seen = set()

    def add_variant(value: str) -> None:
        if len(variants) >= VARIANT_LIMIT:
            return

        normalized_value = normalize_text(value)
        if normalized_value and normalized_value not in seen:
            seen.add(normalized_value)
            variants.append(normalized_value)

    def add_token_variant(value_tokens: list) -> None:
        add_variant(' '.join(value_tokens))
        add_variant(' '.join(_compact(_canonicalize(value_tokens))))

    add_variant(str(term).strip())
    add_variant(' '.join(canonical_tokens))
    add_variant(' '.join(compact_tokens))

    _add_one_token_variants(canonical_tokens, add_token_variant)
    _add_multi_token_variants(canonical_tokens, add_token_variant)

    return variants


def _tokenize(text: str) -> list:
    return normalize_text(text).split()


def _canonicalize(tokens: list) -> list:
    canonical_tokens = []
    index = 0

    while index < len(tokens):
        compact_letter_number = _split_compact_letter_number(tokens[index])
        if compact_letter_number is not None:
            canonical_tokens.extend(compact_letter_number)
            index += 1
            continue

        numeric_token = _canonical_numeric_token(tokens[index])
        if numeric_token is not None:
            canonical_tokens.append(numeric_token)
            index += 1
            continue

        parsed_number = _parse_number_words(tokens, index)
        if parsed_number is not None:
            canonical_tokens.append(str(parsed_number.value))
            index += parsed_number.consumed
            continue

        canonical_tokens.append(_CANONICAL_BY_TOKEN.get(tokens[index], tokens[index]))
        index += 1

    return canonical_tokens


def _compact(tokens: list) -> list:
    compacted = []
    index = 0

    while index < len(tokens):
        if (
            index + 1 < len(tokens) and
            tokens[index] in _COMPACTABLE_LETTERS and
            _is_number_token(tokens[index + 1])
        ):
            compacted.append(f'{tokens[index]}{tokens[index + 1]}')
            index += 2
        else:
            compacted.append(tokens[index])
            index += 1

    return compacted


def _add_one_token_variants(canonical_tokens: list, add_token_variant) -> None:
    for index, token in enumerate(canonical_tokens):
        for replacement in _variant_options(token):
            if replacement == token:
                continue

            add_token_variant(_replace_token(canonical_tokens, {index: replacement}))


def _add_multi_token_variants(canonical_tokens: list, add_token_variant) -> None:
    replacement_options = {}
    for index, token in enumerate(canonical_tokens):
        options = _variant_options(token)[1:]
        if options:
            replacement_options[index] = options

    if len(replacement_options) < 2:
        return

    max_options = max(len(options) for options in replacement_options.values())
    for option_index in range(max_options):
        replacements = {
            index: options[option_index]
            for index, options in replacement_options.items()
            if option_index < len(options)
        }

        add_token_variant(_replace_token(canonical_tokens, replacements))


def _variant_options(token: str) -> tuple:
    if _is_number_token(token):
        return _number_variant_options(int(token))

    return _VARIANTS_BY_CANONICAL.get(token, (token,))


def _number_variant_options(number: int) -> tuple:
    canonical = str(number)
    variants = [canonical, _number_to_words(number)]
    variants.extend(_VARIANTS_BY_CANONICAL.get(canonical, ())[1:])

    return _dedupe_tuple(tuple(variants))


def _replace_token(tokens: list, replacements: dict) -> list:
    replaced_tokens = []

    for index, token in enumerate(tokens):
        replacement = replacements.get(index)
        if replacement is None:
            replaced_tokens.append(token)
        else:
            replaced_tokens.extend(replacement.split())

    return replaced_tokens


def _canonical_numeric_token(token: str) -> str:
    match = re.fullmatch(r'(\d{1,2})s?', token)
    if match is None:
        return None

    return str(int(match.group(1)))


def _parse_number_words(tokens: list, index: int):
    token = tokens[index]

    if token in _TENS_NUMBER_WORDS:
        number = _TENS_NUMBER_WORDS[token]
        next_index = index + 1

        if next_index < len(tokens) and tokens[next_index] in _UNIT_NUMBER_WORDS:
            unit = _UNIT_NUMBER_WORDS[tokens[next_index]]
            if unit > 0:
                return _ParsedNumber(number + unit, 2)

        return _ParsedNumber(number, 1)

    if token in _UNIT_NUMBER_WORDS:
        return _ParsedNumber(_UNIT_NUMBER_WORDS[token], 1)

    return None


def _split_compact_letter_number(token: str) -> list:
    match = re.fullmatch(r'([a-z]+)(\d{1,2})s?', token)
    if match is None:
        return None

    letter = _CANONICAL_BY_TOKEN.get(match.group(1), match.group(1))
    if letter not in _COMPACTABLE_LETTERS:
        return None

    return [letter, str(int(match.group(2)))]


def _number_to_words(number: int) -> str:
    if number < 20:
        return _word_for_number(number, _UNIT_NUMBER_WORDS)

    tens = number - (number % 10)
    unit = number % 10

    tens_word = _word_for_number(tens, _TENS_NUMBER_WORDS)
    if unit == 0:
        return tens_word

    return f'{tens_word} {_word_for_number(unit, _UNIT_NUMBER_WORDS)}'


def _word_for_number(number: int, words: dict) -> str:
    for word, value in words.items():
        if value == number:
            return word

    return str(number)


def _is_number_token(token: str) -> bool:
    return token.isdigit() and int(token) in range(0, 100)


def _dedupe_tuple(values: tuple) -> tuple:
    deduped_values = []
    seen = set()

    for value in values:
        if value not in seen:
            seen.add(value)
            deduped_values.append(value)

    return tuple(deduped_values)
