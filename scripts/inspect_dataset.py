
from pathlib import Path
import json

file_path = Path(
    "data/raw/transliteration/aksharantar/ben_train.json"
)

print("File exists:", file_path.exists())

if file_path.exists():
    print("\nFIRST DATASET RECORD:\n")

    with open(file_path, "r", encoding="utf-8") as file:
        first_line = file.readline()

    print(first_line[:1000])

    try:
        record = json.loads(first_line)
        if isinstance(record, dict):
            print("\nAvailable fields:")
            print(list(record.keys()))
    except json.JSONDecodeError:
        print("\nWe need to inspect the JSON structure further.")
