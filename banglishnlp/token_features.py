
def get_script(token):
    has_bengali = any(
        "\u0980" <= char <= "\u09FF"
        for char in token
    )

    has_latin = any(
        "a" <= char.lower() <= "z"
        for char in token
    )

    if has_bengali and has_latin:
        return "MIXED"

    if has_bengali:
        return "BENGALI"

    if has_latin:
        return "LATIN"

    return "OTHER"


def token_features(token, previous, following):
    word = token.casefold()

    features = {
        "word": word,
        "script": get_script(token),
        "length": min(len(token), 20),
        "is_digit": token.isdigit(),
        "has_digit": any(c.isdigit() for c in token),
        "previous_word": previous.casefold(),
        "next_word": following.casefold(),
        "previous_script": get_script(previous),
        "next_script": get_script(following),
    }

    for length in (1, 2, 3, 4):
        features[f"prefix_{length}"] = word[:length]
        features[f"suffix_{length}"] = word[-length:]

    return features
