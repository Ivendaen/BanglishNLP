
import csv
from pathlib import Path

from banglishnlp.cleaner import clean_text

# Project folders
ROOT = Path(__file__).resolve().parents[1]

GOLD_FILE = (
    ROOT / "data" / "annotations" / "cleaner_test_gold.csv"
)

OUTPUT_FILE = (
    ROOT / "data" / "processed" / "cleaner_test_evaluation.csv"
)

# Read the gold-standard dataset
with open(GOLD_FILE, "r", encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)
    records = list(reader)

# Compare cleaner output with expected output
results = []
correct = 0

for record in records:
    original = record["noisy_text"]
    expected = record["expected_clean_text"]

    actual = clean_text(original)

    is_correct = actual == expected

    if is_correct:
        correct += 1

    results.append({
        "id": record["id"],
        "noisy_text": original,
        "expected_clean_text": expected,
        "actual_clean_text": actual,
        "match": is_correct,
    })

# Calculate evaluation metrics
total = len(results)
accuracy = (correct / total * 100) if total else 0

print("=" * 50)
print("BANGLISHNLP TEXT CLEANING EVALUATION")
print("=" * 50)

print(f"Total sentences: {total}")
print(f"Correct predictions: {correct}")
print(f"Incorrect predictions: {total - correct}")
print(f"Exact-match accuracy: {accuracy:.2f}%")

# Show the first five mismatches
print("\nFIRST 5 MISMATCHES")
print("-" * 50)

mismatches = [
    result for result in results
    if not result["match"]
]

if not mismatches:
    print("No mismatches found!")
else:
    for item in mismatches[:5]:
        print(f"\nID: {item['id']}")
        print("Expected:", repr(item["expected_clean_text"]))
        print("Actual:  ", repr(item["actual_clean_text"]))

# Save detailed evaluation results
OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8-sig",
    newline=""
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "id",
            "noisy_text",
            "expected_clean_text",
            "actual_clean_text",
            "match",
        ],
    )

    writer.writeheader()
    writer.writerows(results)

print("\nEvaluation results saved to:")
print(OUTPUT_FILE)
