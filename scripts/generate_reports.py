"""
Run this script once to generate the 10 synthetic clinical PDFs.
Usage: python scripts/generate_reports.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data_gen.synthetic import generate_all

if __name__ == "__main__":
    generate_all()
