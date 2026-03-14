
from pathlib import Path
import argparse
import pandas as pd

def parse_count_file(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            gene = parts[0]
            try:
                count = float(parts[-1])
            except ValueError:
                continue
            rows.append((gene, count))
    if not rows:
        raise ValueError(f"No gene counts parsed from {path}")
    return pd.DataFrame(rows, columns=["gene", path.stem]).drop_duplicates(subset=["gene"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/raw/tcga_prad/files")
    parser.add_argument("--output-dir", default="data/raw/tcga_prad")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = list(input_dir.rglob("*.tsv")) + list(input_dir.rglob("*.txt"))
    if not candidates:
        raise FileNotFoundError(f"No count files found under {input_dir}")

    merged = None
    for fp in candidates:
        one = parse_count_file(fp)
        merged = one if merged is None else merged.merge(one, on="gene", how="outer")

    merged = merged.fillna(0.0)
    merged.to_csv(output_dir / "expression_matrix.csv", index=False)
    print(f"Wrote {output_dir / 'expression_matrix.csv'}")

if __name__ == "__main__":
    main()
