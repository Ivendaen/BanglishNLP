
from banglishnlp.token_language_tagger import TokenLanguageTagger


def main():
    tagger = TokenLanguageTagger()

    samples = [
        "Ami ajke university te jabo!",
        "আমি today class e যাব।",
        "Bhai assignment ta complete korso?",
        "I really love Bangladesh!",
        "Amar exam tomorrow.",
        "Rafsan Dhaka te thake.",
        "Hello 😊 #বাংলা",
        "Ami 500 taka diye phone kinlam.",
    ]

    for i, text in enumerate(samples, start=1):
        print(f"\n=== EXAMPLE {i} ===")
        print("TEXT:", text)

        results = tagger.tag(text)

        for item in results:
            print(
                f"{item['token']:<20} "
                f"{item['label']}"
            )


if __name__ == "__main__":
    main()
