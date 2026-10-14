
from pathlib import Path
import pyarrow.parquet as pq

# Locate the BanglishNLP project folder
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "raw" / "collected"

# Find downloaded Parquet files
files = sorted(DATA_DIR.glob("*.parquet"))

if not files:
    raise FileNotFoundError(
        "No Parquet file found in data/raw/collected/"
    )

# Inspect the first Parquet file
file_path = files[0]
parquet_file = pq.ParquetFile(file_path)

print("=" * 50)
print("BANGLISHNLP DATASET INSPECTION")
print("=" * 50)

print("\nDataset file:", file_path.name)
print("Total rows:", parquet_file.metadata.num_rows)

print("\nAVAILABLE COLUMNS:")
for field in parquet_file.schema_arrow:
    print("-", field.name, ":", field.type)

print("\nFIRST 5 RECORDS:")
print("=" * 50)

# Read only the first few records
batch = next(
    parquet_file.iter_batches(batch_size=5),
    None
)

if batch is not None:
    records = batch.to_pylist()

    for i, record in enumerate(records, 1):
        print(f"\nRecord {i}")

        for key, value in record.items():
            preview = str(value).replace("\n", " ")
            print(f"{key}: {preview[:200]}")

print("\nInspection completed!")
