"""Voice-friendly matching helpers for music names."""

from itertools import product
import re
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

_NUMBER_HOMOPHONES = {
    'to': '2',
    'too': '2',
    'for': '4',
    'ate': '8',
}

_LETTER_WORDS = {
    'you': 'u',
    'are': 'r',
    'bee': 'b',
    'be': 'b',
    'see': 'c',
    'sea': 'c',
}

_CONNECTOR_WORDS = {
    'and': 'and',
    'n': 'and',
}

_TOKEN_CANONICALS = {
    **_NUMBER_HOMOPHONES,
    **_LETTER_WORDS,
    **_CONNECTOR_WORDS,
}

_LETTER_TOKENS = {'b', 'c', 'r', 'u'}

_LETTER_VARIANTS = {
    'b': ('b', 'bee', 'be'),
    'c': ('c', 'see', 'sea'),
    'r': ('r', 'are'),
    'u': ('u', 'you'),
}

_CONNECTOR_VARIANTS = {
    'and': ('and', 'n'),
}

_HOMOPHONE_NUMBER_VARIANTS = {
    '2': ('2', 'two', 'to', 'too'),
    '4': ('4', 'four', 'for'),
    '8': ('8', 'eight', 'ate'),
}


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
    tokens = _canonical_tokens(normalize_text(text).split())

    return ' '.join(_compact_letter_number_tokens(tokens))


def names_match(query: str, candidate: str) -> bool:
    """Return True when two names match after voice-aware normalization."""
    query_key = match_key(query)
    candidate_key = match_key(candidate)

    return bool(query_key) and query_key == candidate_key


def query_variants(term: str) -> list:
    """Generate deterministic search terms for common Alexa mishearings."""
    normalized_term = normalize_text(term)
    if not normalized_term:
        return []

    variants = []
    seen = set()

    def add_variant(value: str) -> None:
        if len(variants) >= VARIANT_LIMIT:
            return

        normalized_value = normalize_text(value)
        if normalized_value and normalized_value not in seen:
            seen.add(normalized_value)
            variants.append(normalized_value)

    add_variant(str(term).strip())

    canonical_tokens = _canonical_tokens(normalized_term.split())
    compact_tokens = _compact_letter_number_tokens(canonical_tokens)
    add_variant(' '.join(compact_tokens))

    options = [_variant_options(token) for token in canonical_tokens]
    for token_choice in product(*options):
        add_variant(' '.join(token_choice))
        add_variant(' '.join(_compact_letter_number_tokens(list(token_choice))))

        if len(variants) >= VARIANT_LIMIT:
            break

    return variants


def _canonical_tokens(tokens: list) -> list:
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
            number, consumed = parsed_number
            canonical_tokens.append(str(number))
            index += consumed
            continue

        canonical_tokens.append(_TOKEN_CANONICALS.get(tokens[index], tokens[index]))
        index += 1

    return canonical_tokens


def _canonical_numeric_token(token: str) -> str:
    match = re.fullmatch(r'(\d{1,2})s?', token)
    if match is None:
        return None

    return str(int(match.group(1)))


def _parse_number_words(tokens: list, index: int) -> tuple:
    token = tokens[index]

    if token in _TENS_NUMBER_WORDS:
        number = _TENS_NUMBER_WORDS[token]
        next_index = index + 1

        if next_index < len(tokens) and tokens[next_index] in _UNIT_NUMBER_WORDS:
            unit = _UNIT_NUMBER_WORDS[tokens[next_index]]
            if unit > 0:
                return number + unit, 2

        return number, 1

    if token in _UNIT_NUMBER_WORDS:
        return _UNIT_NUMBER_WORDS[token], 1

    return None


def _split_compact_letter_number(token: str) -> list:
    match = re.fullmatch(r'([a-z]+)(\d{1,2})s?', token)
    if match is None:
        return None

    letter = _TOKEN_CANONICALS.get(match.group(1), match.group(1))
    if letter not in _LETTER_TOKENS:
        return None

    return [letter, str(int(match.group(2)))]


def _compact_letter_number_tokens(tokens: list) -> list:
    compacted = []
    index = 0

    while index < len(tokens):
        if (
            index + 1 < len(tokens) and
            tokens[index] in _LETTER_TOKENS and
            _is_number_token(tokens[index + 1])
        ):
            compacted.append(f'{tokens[index]}{tokens[index + 1]}')
            index += 2
        else:
            compacted.append(tokens[index])
            index += 1

    return compacted


def _variant_options(token: str) -> tuple:
    if _is_number_token(token):
        number = int(token)
        if number in range(0, 100):
            word_variant = _number_to_words(number)
            return _dedupe_tuple((
                token,
                word_variant,
                *_HOMOPHONE_NUMBER_VARIANTS.get(token, ()),
            ))

    if token in _LETTER_VARIANTS:
        return _LETTER_VARIANTS[token]

    if token in _CONNECTOR_VARIANTS:
        return _CONNECTOR_VARIANTS[token]

    return (token,)


def _number_to_words(number: int) -> str:
    if number < 20:
        for word, value in _UNIT_NUMBER_WORDS.items():
            if value == number:
                return word

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
