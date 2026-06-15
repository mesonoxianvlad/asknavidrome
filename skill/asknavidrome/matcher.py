"""Voice-friendly matching helpers for music names."""

from itertools import product
import re
import unicodedata


_TOKEN_VARIANT_GROUPS = (
    ('2', 'two', 'to', 'too'),
    ('3', 'three'),
    ('30', 'thirty'),
    ('u', 'you'),
)

_TOKEN_VARIANTS = {
    token: group
    for group in _TOKEN_VARIANT_GROUPS
    for token in group
}

_CANONICAL_TOKENS = {
    token: group[0]
    for group in _TOKEN_VARIANT_GROUPS
    for token in group
}

_COMPACT_TOKEN_VARIANTS = {
    'u2': ('u2', 'u 2', 'u two', 'you 2', 'you two', 'you to', 'you too')
}


def normalize_text(text: str) -> str:
    """Return a lower-case, punctuation-free representation of text."""
    if text is None:
        return ''

    normalized = unicodedata.normalize('NFKD', str(text))
    ascii_text = normalized.encode('ascii', 'ignore').decode('ascii')
    lower_text = ascii_text.lower()
    words_only = re.sub(r'[^a-z0-9]+', ' ', lower_text)

    return re.sub(r'\s+', ' ', words_only).strip()


def match_key(text: str) -> str:
    """Return the canonical key used for exact voice-aware comparisons."""
    tokens = [
        _CANONICAL_TOKENS.get(token, token)
        for token in normalize_text(text).split()
    ]

    return ' '.join(_compact_tokens(tokens))


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
        normalized_value = normalize_text(value)
        if normalized_value and normalized_value not in seen:
            seen.add(normalized_value)
            variants.append(normalized_value)

    add_variant(str(term).strip())

    tokens = normalized_term.split()
    options = [
        _COMPACT_TOKEN_VARIANTS.get(token, _TOKEN_VARIANTS.get(token, (token,)))
        for token in tokens
    ]

    for token_choice in product(*options):
        add_variant(' '.join(token_choice))

        canonical_tokens = [
            _CANONICAL_TOKENS.get(token, token)
            for token in ' '.join(token_choice).split()
        ]
        add_variant(' '.join(_compact_tokens(canonical_tokens)))

    return variants


def _compact_tokens(tokens: list) -> list:
    compacted = []
    index = 0

    while index < len(tokens):
        if index + 1 < len(tokens) and tokens[index] == 'u' and tokens[index + 1] == '2':
            compacted.append('u2')
            index += 2
        else:
            compacted.append(tokens[index])
            index += 1

    return compacted
