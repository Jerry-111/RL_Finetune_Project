#!/usr/bin/env python3
"""Compare two action-decision CSV logs and report first mismatch."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare native-policy action logs")
    parser.add_argument("--ref-log", type=Path, required=True, help="Reference CSV (native loop)")
    parser.add_argument("--rap-log", type=Path, required=True, help="RAP pipeline CSV")
    parser.add_argument("--out-report", type=Path, default=None, help="Optional text report path")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    ref_rows = load_rows(args.ref_log)
    rap_rows = load_rows(args.rap_log)
    n = min(len(ref_rows), len(rap_rows))

    lines: list[str] = []
    lines.append(f"ref_rows={len(ref_rows)}")
    lines.append(f"rap_rows={len(rap_rows)}")
    lines.append(f"compared_rows={n}")
    if len(ref_rows) != len(rap_rows):
        lines.append("row_count_match=False")
    else:
        lines.append("row_count_match=True")

    compare_cols = ["ego_slot", "ego_id", "ego_action", "action_count", "action_sum", "action_crc32"]
    mismatch_count = 0
    first_mismatch = None
    for i in range(n):
        r = ref_rows[i]
        p = rap_rows[i]
        for col in compare_cols:
            if r.get(col) != p.get(col):
                mismatch_count += 1
                if first_mismatch is None:
                    first_mismatch = (i, col, r.get(col), p.get(col))
                break

    lines.append(f"mismatch_rows={mismatch_count}")
    if first_mismatch is None:
        lines.append("first_mismatch=none")
    else:
        i, col, rv, pv = first_mismatch
        lines.append(f"first_mismatch_row={i}")
        lines.append(f"first_mismatch_col={col}")
        lines.append(f"first_ref_value={rv}")
        lines.append(f"first_rap_value={pv}")

    report = "\n".join(lines)
    print(report)
    if args.out_report is not None:
        args.out_report.parent.mkdir(parents=True, exist_ok=True)
        args.out_report.write_text(report + "\n", encoding="utf-8")
        print(f"[done] wrote report: {args.out_report}")


if __name__ == "__main__":
    main()
