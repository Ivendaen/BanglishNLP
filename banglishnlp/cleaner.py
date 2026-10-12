
import re
import unicodedata


# Match URLs beginning with http://, https://, or www.
URL_PATTERN = re.compile(
    r"\b(?:https?://|www\.)[^\s<>()]+",
    flags=re.IGNORECASE
)

# Match social media mentions, e.g. @sakib123
MENTION_PATTERN = re.compile(
    r"(?<![\w@])@[A-Za-z0-9_]+"
)

# Match repeated punctuation, e.g. !!!, ???, ...
REPEATED_PUNCTUATION = re.compile(r"([!?.,;:।])\1+")


def clean_text(text: str, punctuation: str = "normalize") -> str:
    """
    Clean Bangla-English code-mixed text.

    Features:
        - Remove URLs
        - Remove user mentions
        - Normalize repeated punctuation or remove punctuation
        - Remove repeated spaces, tabs, and newlines
        - Preserve Bangla, English, Banglish, and emojis

    Args:
        text: Input text.
        punctuation:
            "normalize" - reduce repeated punctuation
            "remove"    - remove punctuation
            "keep"      - leave punctuation unchanged

    Returns:
        Cleaned text.
    """

    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    if punctuation not in {"normalize", "remove", "keep"}:
        raise ValueError(
            "punctuation must be 'normalize', 'remove', or 'keep'"
        )

    # STEP 1: Remove URLs, preserving trailing sentence punctuation.
    def replace_url(match):
        url = match.group(0)
        trailing = url[len(url.rstrip(".,!?;:।")):]
        return f" {trailing} "

    text = URL_PATTERN.sub(replace_url, text)

    # STEP 2: Remove user mentions
    text = MENTION_PATTERN.sub(" ", text)

    # STEP 3: Handle punctuation
    if punctuation == "normalize":
        # Normalize common punctuation variants
        text = text.translate(str.maketrans({
            "“": '"',
            "”": '"',
            "‘": "'",
            "’": "'",
            "…": "...",
            "—": "-",
        }))

        # Reduce repeated punctuation
        text = REPEATED_PUNCTUATION.sub(r"\1", text)

    elif punctuation == "remove":
        # Replace Unicode punctuation with spaces,
        # including English and Bengali punctuation.
        text = "".join(
            " " if unicodedata.category(char).startswith("P") else char
            for char in text
        )

    # STEP 4: Normalize whitespace everywhere
    text = re.sub(r"\s+", " ", text)

    # STEP 5: Remove leading and trailing spaces
    return text.strip()
