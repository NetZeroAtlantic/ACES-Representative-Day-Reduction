from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aces_tsam.data import inspect_sqlite


def main() -> None:
    parser = argparse.ArgumentParser(description="List SQLite tables and columns for an ACES 8760 database.")
    parser.add_argument("database", help="Path to the SQLite database")
    parser.add_argument("--output", default="sqlite_schema.csv", help="CSV destination")
    args = parser.parse_args()
    schema = inspect_sqlite(args.database)
    schema.to_csv(args.output, index=False)
    print(schema.to_string(index=False))
    print(f"Schema written to {args.output}")


if __name__ == "__main__":
    main()
