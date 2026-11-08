"""Step 1b: per-column profile for train and test.

For every column: dtype, missing count and %, number of unique values.
For numeric columns also: min, max, mean, median, std.

Usage (from the repo root):
    uv run python scripts/profile_columns.py
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "titanic-data"


def profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        row = {
            "column": col,
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_%": round(100 * s.isna().mean(), 1),
            "unique": s.nunique(),
        }
        if pd.api.types.is_numeric_dtype(s):
            row.update(
                min=s.min(),
                max=s.max(),
                mean=s.mean(),
                median=s.median(),
                std=s.std(),
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index("column")


def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")

    for name in ("train", "test"):
        df = pd.read_csv(DATA_DIR / f"{name}.csv")
        print(f"\n=== {name} ({len(df)} rows) ===")
        # Non-numeric columns get NaN for the stats; print blanks instead.
        print(profile(df).to_string(na_rep=""))


if __name__ == "__main__":
    main()
