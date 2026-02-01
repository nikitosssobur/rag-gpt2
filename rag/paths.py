from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

SAMPLE_DATA_PATH = DATA_DIR / "sample"

VECTOR_DB_PATH = DATA_DIR / "vector_db"

if __name__ == '__main__':
    print("")