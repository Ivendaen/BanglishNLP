
import re
import unicodedata


# STEP 1: URL Detection
# Detect URLs with or without https:// or www.
# Examples:
# https://example.com
# http://example.com
# www.example.com
# example.com
# example.org/about

URL_PATTERN = re.compile(
    r"(?<![\w@.])"
    r"(?:"
    r"(?:https?://|www\.)[^\s<>()]+"
    r"|"
    r"(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}"
    r"(?:[/?#][^\s<>()]*)?"
    r")",
    flags=re.IGNORECASE
)


# STEP 2: User Mention Detection
# Supports English and Bengali user mentions.
# Examples: @rahim, @sakib123, @রহিম, @সাকিব১২৩

MENTION_PATTERN = re.compile(
    r"(?<![\w@])@[\w\u0980-\u09FF]+"
)


# STEP 3: Repeated Punctuation Detection
# Examples: !!!, ???, ..., ,,,, ।।।।

REPEATED_PUNCTUATION = re.compile(
    r"([!?.,;:।])\1+"
)


def clean_text(text: str, punctuation: str = "normalize") -> str:
    """
    Clean Bangla-English code-mixed text.

    Features:
        - Remove URLs (with or without protocols)
        - Remove English and Bengali user mentions
        - Preserve email addresses
        - Normalize mixed and repeated punctuation
        - Optionally remove or keep punctuation
        - Normalize repeated whitespace anywhere in text
        - Preserve Bangla, English, Banglish, and emojis

    Args:
        text: Input text.

        punctuation:
            "normalize" - Normalize repeated punctuation
            "remove"    - Remove punctuation
            "keep"      - Keep punctuation unchanged

    Returns:
        Cleaned text.
    """

    # Validate input
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    if punctuation not in {"normalize", "remove", "keep"}:
        raise ValueError(
            "punctuation must be 'normalize', 'remove', or 'keep'"
        )

    # STEP 1: Remove URLs
    # Preserve trailing sentence punctuation.

    def replace_url(match):
        url = match.group(0)
        trailing = url[len(url.rstrip(".,!?;:।")):]
        return f" {trailing} "

    text = URL_PATTERN.sub(replace_url, text)

    # STEP 2: Remove English and Bengali user mentions
    text = MENTION_PATTERN.sub(" ", text)

    # STEP 3: Handle punctuation
    if punctuation == "normalize":

        # Normalize common Unicode punctuation variants
        text = text.translate(str.maketrans({
            "“": '"',
            "”": '"',
            "‘": "'",
            "’": "'",
            "…": "...",
            "—": "-",
        }))

        # Normalize mixed punctuation sequences.
        # Examples:
        # ?!?!?  -> ?!
        # !?!!?? -> !?
        # ...!!! -> !
        # !!!??? -> !?

        def normalize_mixed_punctuation(match):
            symbols = match.group(0)

            # Preserve first-occurrence order of ! and ?
            if "!" in symbols and "?" in symbols:
                if symbols.index("!") < symbols.index("?"):
                    return "!?"
                return "?!"

            # Prefer ! over periods
            if "!" in symbols:
                return "!"

            # Prefer ? over periods
            if "?" in symbols:
                return "?"

            # Multiple periods become one period
            return "."

        # Normalize mixed periods, exclamation marks,
        # and question marks
        text = re.sub(
            r"[.!?]{2,}",
            normalize_mixed_punctuation,
            text
        )

        # Normalize other repeated punctuation
        # Examples: ,,, -> , and ।।। -> ।
        text = REPEATED_PUNCTUATION.sub(r"\1", text)

    elif punctuation == "remove":

        # Remove Unicode punctuation, including
        # English and Bengali punctuation
        text = "".join(
            " " if unicodedata.category(char).startswith("P") else char
            for char in text
        )

    # STEP 4: Normalize whitespace everywhere
    # Handles repeated spaces, tabs, and newlines
    text = re.sub(r"\s+", " ", text)

    # STEP 5: Remove leading and trailing whitespace
    return text.strip()
