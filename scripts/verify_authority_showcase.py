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


def off_axis_max(rows: list[dict[str, str]], authority: str, c: float, value_col: str) -> float:
    foundations = sorted({r["foundation"] for r in rows if r["foundation"] != authority})
    vals = []
    for foundation in foundations:
        _, pos, neg = signed_delta(rows, foundation, c, value_col)
        vals.extend([abs(pos), abs(neg)])
    return max(vals)


def instrument_table(name: str, rows: list[dict[str, str]], c: float, value_col: str) -> tuple[list, bool]:
    authority = authority_foundation(rows)
    base, pos, neg = signed_delta(rows, authority, c, value_col)
    off = off_axis_max(rows, authority, c, value_col)
    ok = pos > 0 and neg < 0
    table = [[name, authority, value_col, base, pos, neg, off, ok]]
    return table, ok


def coherence_rows(rows: list[dict[str, str]], c: float, cols: list[str]) -> list[list]:
    out = []
    for cc in [0.0, c, -c]:
        sample = [r for r in rows if math.isclose(fnum(r, "c"), cc)][0]
        out.append([cc] + [fnum(sample, col) for col in cols])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--small-c", type=float, default=0.5)
    args = ap.parse_args()

    rows = []
    ok = True
    mfv_path = args.out / "mfv_profiles.csv"
    mfq2_path = args.out / "mfq2_profiles.csv"
    assert mfv_path.exists() or mfq2_path.exists(), args.out

    mfv = None
    mfq2 = None
    if mfv_path.exists():
        mfv = read_rows(mfv_path)
        table, table_ok = instrument_table("MFV", mfv, args.small_c, "dlogit")
        rows += table
        ok = ok and table_ok
    if mfq2_path.exists():
        mfq2 = read_rows(mfq2_path)
        table, table_ok = instrument_table("MFQ-2", mfq2, args.small_c, "C")
        rows += table
        ok = ok and table_ok

    print("\nAuthority direction at smallest nonzero c")
    print(tabulate(
        rows,
        headers=["instrument", "axis", "readout", "base", "delta(+c)", "delta(-c)", "max off-axis |delta|", "pass"],
        tablefmt="pipe",
        floatfmt="+.3f",
    ))

    if mfv is not None:
        print("\nMFV coherence")
        print(tabulate(
            coherence_rows(mfv, args.small_c, ["pmass", "mean_margin", "frac_unscorable"]),
            headers=["c", "pmass", "mean_margin", "frac_unscorable"],
            tablefmt="pipe",
            floatfmt="+.3f",
        ))

    if mfq2 is not None:
        print("\nMFQ-2 coherence")
        print(tabulate(
            coherence_rows(mfq2, args.small_c, ["pmass"]),
            headers=["c", "pmass"],
            tablefmt="pipe",
            floatfmt="+.3f",
        ))

    if not ok:
        raise SystemExit("FAIL: +c did not raise Authority and -c did not lower Authority at small c")


if __name__ == "__main__":
    main()
