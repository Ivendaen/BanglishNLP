
import csv
import json
import sqlite3
import tempfile
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data/raw/transliteration/aksharantar"
OUTPUT_DIR = ROOT / "data/processed/transliteration/aksharantar"

FILES = {
    "train": "ben_train.json",
    "dev": "ben_valid.json",
    "test": "ben_test.json",
}


def prepare_data():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # SQLite tracks duplicates on disk, so the large training
    # dataset does not need to be loaded entirely into memory.
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "duplicates.sqlite"
        db = sqlite3.connect(db_path)

        db.execute("""
            CREATE TABLE pairs (
                split TEXT,
                romanized TEXT,
                bangla TEXT,
                PRIMARY KEY (split, romanized, bangla)
            )
        """)

        db.execute("""
            CREATE TABLE words (
                romanized TEXT,
                split TEXT,
                PRIMARY KEY (romanized, split)
            )
        """)

        # Process each official partition separately.
        for split, filename in FILES.items():
            input_path = RAW_DIR / filename
            output_path = OUTPUT_DIR / f"{split}.csv"

            if not input_path.exists():
                raise FileNotFoundError(input_path)

            total = 0
            saved = 0
            duplicates = 0
            invalid = 0

            with open(
                input_path, "r", encoding="utf-8-sig"
            ) as infile, open(
                output_path, "w", encoding="utf-8-sig",
                newline=""
            ) as outfile:

                writer = csv.writer(outfile)
                writer.writerow([
                    "id", "romanized", "bangla",
                    "source", "score"
                ])

                for line_number, line in enumerate(infile, 1):
                    if not line.strip():
                        continue

                    total += 1

                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON in {filename}, "
                            f"line {line_number}"
                        ) from exc

                    romanized = record.get("english word")
                    bangla = record.get("native word")

                    if not isinstance(romanized, str) or \
                       not isinstance(bangla, str):
                        invalid += 1
                        continue

                    romanized = unicodedata.normalize(
                        "NFC", romanized
                    ).strip()

                    bangla = unicodedata.normalize(
                        "NFC", bangla
                    ).strip()

                    if not romanized or not bangla:
                        invalid += 1
                        continue

                    # Remove duplicate source-target pairs
                    # within the same official partition.
                    cursor = db.execute(
                        "INSERT OR IGNORE INTO pairs VALUES (?, ?, ?)",
                        (split, romanized, bangla)
                    )

                    if cursor.rowcount == 0:
                        duplicates += 1
                        continue

                    # Record source words to detect leakage
                    # across train, dev, and test.
                    db.execute(
                        "INSERT OR IGNORE INTO words VALUES (?, ?)",
                        (romanized.casefold(), split)
                    )

                    writer.writerow([
                        record.get("unique_identifier", ""),
                        romanized,
                        bangla,
                        record.get("source", ""),
                        record.get("score")
                    ])

                    saved += 1

            db.commit()

            print(f"\n{split.upper()}")
            print("Original records:", total)
            print("Invalid records:", invalid)
            print("Duplicates removed:", duplicates)
            print("Saved records:", saved)
            print("Output:", output_path)

        # Check whether the same Romanized input occurs
        # in multiple official partitions.
        overlap = db.execute("""
            SELECT COUNT(*) FROM (
                SELECT romanized
                FROM words
                GROUP BY romanized
                HAVING COUNT(*) > 1
            )
        """).fetchone()[0]

        print("\nDATA LEAKAGE CHECK")
        print("Romanized inputs shared across splits:", overlap)

        if overlap:
            print(
                "WARNING: Review cross-split overlap "
                "before training or evaluation."
            )
        else:
            print("PASS: No Romanized input overlap.")

        db.close()

    print("\nDataset preparation completed!")


if __name__ == "__main__":
    prepare_data()
