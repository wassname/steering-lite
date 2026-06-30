from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from tabulate import tabulate


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def fnum(row: dict[str, str], key: str) -> float:
    return float(row[key])


def row_at(rows: list[dict[str, str]], foundation: str, c: float) -> dict[str, str]:
    matches = [r for r in rows if r["foundation"] == foundation and math.isclose(fnum(r, "c"), c)]
    assert len(matches) == 1, (foundation, c, len(matches))
    return matches[0]


def authority_foundation(rows: list[dict[str, str]]) -> str:
    names = sorted({r["foundation"] for r in rows})
    exact = [n for n in names if n == "Authority"]
    if exact:
        return exact[0]
    matches = [n for n in names if "authority" in n.lower()]
    assert len(matches) == 1, matches
    return matches[0]


def signed_delta(rows: list[dict[str, str]], foundation: str, c: float, value_col: str) -> tuple[float, float, float]:
    base = fnum(row_at(rows, foundation, 0.0), value_col)
    pos = fnum(row_at(rows, foundation, c), value_col) - base
    neg = fnum(row_at(rows, foundation, -c), value_col) - base
    return base, pos, neg


def paired_cs(rows: list[dict[str, str]]) -> list[float]:
    cs = sorted({fnum(r, "c") for r in rows})
    pos = [c for c in cs if c > 0.0]
    pairs = [c for c in pos if any(math.isclose(other, -c) for other in cs)]
    assert pairs, cs
    return pairs


def instrument_rows(name: str, rows: list[dict[str, str]], value_col: str) -> list[list]:
    authority = authority_foundation(rows)
    out = []
    for c in paired_cs(rows):
        base, pos, neg = signed_delta(rows, authority, c, value_col)
        out.append([name, authority, value_col, c, base, pos, neg, pos > 0 and neg < 0])
    return out


def mfv_coherence_rows(rows: list[dict[str, str]]) -> list[list]:
    authority = authority_foundation(rows)
    base_margin = fnum(row_at(rows, authority, 0.0), "mean_margin")
    out = []
    for c in [0.0, *paired_cs(rows), *[-c for c in paired_cs(rows)]]:
        row = row_at(rows, authority, c)
        margin = fnum(row, "mean_margin")
        out.append([
            c,
            fnum(row, "pmass"),
            margin,
            margin / base_margin,
            fnum(row, "frac_unscorable"),
            fnum(row, "mean_nll_prefill"),
        ])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    args = ap.parse_args()

    mfv_path = args.out / "mfv_profiles.csv"
    assert mfv_path.exists(), mfv_path

    mfv = read_rows(mfv_path)

    print("\nMFV Authority direction over paired c values")
    print(tabulate(
        instrument_rows("MFV", mfv, "dlogit"),
        headers=["instrument", "axis", "readout", "c", "base", "delta(+c)", "delta(-c)", "signed direction"],
        tablefmt="pipe",
        floatfmt="+.3f",
    ))

    print("\nMFV coherence evidence")
    print(tabulate(
        mfv_coherence_rows(mfv),
        headers=["c", "pmass", "mean_margin", "margin/base", "frac_unscorable", "mean_nll_prefill"],
        tablefmt="pipe",
        floatfmt="+.3f",
    ))


if __name__ == "__main__":
    main()
