
import re


TOKEN_PATTERN = re.compile(
    r"(?<![\w@])(?:https?://|www\.)[^\s<>()]+"
    r"|(?<![\w@])[\w.+-]+@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}"
    r"|[^\s\u200d](?:\ufe0f|[\U0001F3FB-\U0001F3FF])*(?:\u200d[^\s\u200d](?:\ufe0f|[\U0001F3FB-\U0001F3FF])*)+"
    r"|[\U0001F1E6-\U0001F1FF]{2}"
    r"|[#*0-9]\ufe0f?\u20e3"
    r"|[\U0001F300-\U0001FAFF\u2600-\u27BF](?:\ufe0f|[\U0001F3FB-\U0001F3FF])?"
    r"|[#@][\w\u0980-\u09FF]+"
    r"|\d+(?:[.,]\d+)*"
    r"|[A-Za-z]+(?:['’][A-Za-z]+)*"
    r"|[\u0980-\u09FF]+"
    r"|[^\s]",
    re.UNICODE | re.IGNORECASE,
)


def tokenize_with_spans(text: str):
    if not isinstance(text, str):
        raise TypeError("Input must be a string")

    results = []

    for match in TOKEN_PATTERN.finditer(text):
        token = match.group()
        start = match.start()
        end = match.end()

        if token.lower().startswith(
            ("http://", "https://", "www.")
        ):
            cleaned = token.rstrip(".,!?;:।")

            if cleaned:
                end = start + len(cleaned)

                results.append((cleaned, start, end))

                for pos in range(end, match.end()):
                    results.append(
                        (text[pos], pos, pos + 1)
                    )

                continue

        results.append((token, start, end))

    return results


def tokenize(text: str):
    if not isinstance(text, str):
        raise TypeError("Input must be a string")

    return [
        token
        for token, start, end in tokenize_with_spans(text)
    ]
