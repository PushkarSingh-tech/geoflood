"""Validate a CSV against the dataset contract. Usage: python scripts/validate_data.py path/to/file.csv"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml import preprocess as pp

if len(sys.argv) != 2:
    sys.exit("Usage: python scripts/validate_data.py file.csv")
rep = pp.validate_dataframe(pp.load_csv(sys.argv[1]))
print(json.dumps(rep, indent=2))
sys.exit(0 if rep["ok"] else 1)
