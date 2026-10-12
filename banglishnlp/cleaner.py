
import re


def clean_text(text: str) -> str:
    """
    Clean basic noise from Bangla-English mixed text.

    - Remove URLs
    - Normalize extra spaces
    - Preserve Bangla and English text
    """

    if not isinstance(text, str):
        raise TypeError("Input must be a string")

    # Step 1: Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Step 2: Replace multiple spaces/newlines with one space
    text = re.sub(r"\s+", " ", text)

    # Step 3: Remove spaces from the beginning and end
    return text.strip()
