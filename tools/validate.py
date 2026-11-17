"""Local validation for gta6-twitch-ccu-forecast-01 (not part of the entry).

Runs main.py on the kit's 4 practice cases, checks every output against the
format rules, scores them against the practice answers (SullyGnome hourly
averages, fetched once per game from the endpoint in the kit's NOTICE.txt
and saved in tools/practice_answers.json), and runs a leave-one-out check
over the comparables using main.py's own model functions.

Usage:  python3 tools/validate.py [kit-dir]
"""

import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
KIT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    os.environ.get("GTA6_KIT", str(Path.home() / "workspace/makefun/gta6-twitch-ccu-forecast-01-kit"))
)

TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]


def load_main():
    spec = importlib.util.spec_from_file_location("entry_main", WORK / "main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pinball(y, q, tau):
    d = math.log(max(1, y)) - math.log(max(1e-9, q))
    return 2 * max(tau * d, (tau - 1) * d)


def score_case(actual, hours):
    """Mean pinball loss over scored hours x levels."""
    tot, n = 0.0, 0
    for row in hours:
        y = actual.get(row["hour"])
        if y is None:
            continue
        for tau, q in zip(TAUS, row["q"]):
            tot += pinball(y, q, tau)
            n += 1
    return tot / n if n else float("nan")


def validate_output(path: Path) -> list[str]:
    problems = []
    raw = path.read_bytes()
    if len(raw) > 65536:
        problems.append(f"file too large: {len(raw)} bytes")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return [f"not JSON: {e}"]
    if set(data) != {"hours"}:
        problems.append(f"top-level keys {sorted(data)} != ['hours']")
        return problems
    hours = data["hours"]
    if len(hours) != 23:
        problems.append(f"{len(hours)} rows, expected 23")
    seen = sorted(r.get("hour") for r in hours if isinstance(r, dict))
    if seen != list(range(2, 25)):
        problems.append(f"hours present {seen}, expected 2..24")
    for row in hours:
        if not isinstance(row, dict) or set(row) != {"hour", "q"}:
            problems.append(f"bad row keys: {row}")
            continue
        q = row["q"]
        if len(q) != 5:
            problems.append(f"hour {row['hour']}: {len(q)} quantiles")
            continue
        for i, v in enumerate(q):
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                problems.append(f"hour {row['hour']}: q[{i}] not finite: {v!r}")
            elif v <= 0 or v > 1e9:
                problems.append(f"hour {row['hour']}: q[{i}] out of range: {v}")
        if any(q[i] > q[i + 1] for i in range(4)):
            problems.append(f"hour {row['hour']}: quantiles decrease: {q}")
    return problems


def baseline_hours(p1):
    return [
        {"hour": h, "q": [round(p1 * math.exp(z * 0.15 * math.sqrt(h - 1)), 2) for z in Z]}
        for h in range(2, 25)
    ]


def main() -> int:
    entry = load_main()
    answers = json.loads((HERE / "practice_answers.json").read_text())
    out_dir = Path("/tmp/gta6-validate-out")
    out_dir.mkdir(exist_ok=True)
    for f in out_dir.glob("*.out"):
        f.unlink()

    # 1. Run main.py on the practice cases, twice (determinism check).
    env = dict(os.environ, MAKEFUN_INPUT_DIR=str(KIT / "input"),
               MAKEFUN_OUTPUT_DIR=str(out_dir))
    t0 = time.time()
    r1 = subprocess.run([sys.executable, str(WORK / "main.py")], env=env,
                        capture_output=True, text=True)
    run1_s = time.time() - t0
    if r1.returncode != 0:
        print("main.py FAILED:", r1.stderr)
        return 1
    first = {p.name: p.read_bytes() for p in sorted(out_dir.glob("*.out"))}
    t0 = time.time()
    r2 = subprocess.run([sys.executable, str(WORK / "main.py")], env=env,
                        capture_output=True, text=True)
    run2_s = time.time() - t0
    second = {p.name: p.read_bytes() for p in sorted(out_dir.glob("*.out"))}
    deterministic = first == second
    print(f"runs: {run1_s:.2f}s and {run2_s:.2f}s for 4 cases "
          f"(limit 8s/case); byte-identical reruns: {deterministic}")

    # 2. Validate + score each practice case.
    print("\npractice cases (pinball loss, lower is better):")
    model_losses, base_losses = [], []
    for pid in ["p00", "p01", "p02", "p03"]:
        case = json.loads((KIT / "input" / f"{pid}.in").read_text())
        out_path = out_dir / f"{pid}.out"
        problems = validate_output(out_path)
        data = json.loads(out_path.read_text())
        actual = {int(h): v for h, v in answers[pid].items()}
        model_loss = score_case(actual, data["hours"])
        base_loss = score_case(actual, baseline_hours(case["known"][0]["viewers"]))
        model_losses.append(model_loss)
        base_losses.append(base_loss)
        status = "OK" if not problems else "PROBLEMS: " + "; ".join(problems)
        print(f"  {pid} {case['game']['name'][:34]:34s} valid={status} "
              f"model={model_loss:.4f} baseline={base_loss:.4f}")
    print(f"  MEAN  model={sum(model_losses)/4:.4f} "
          f"baseline={sum(base_losses)/4:.4f} "
          f"(house baseline is ~0.387 per RULES.txt)")

    # 3. Leave-one-out over the comparables, using main.py's own functions.
    print("\nleave-one-out over comparables (model vs baseline):")
    allcomps = {}
    for pid in ["p00", "p01", "p02", "p03"]:
        case = json.loads((KIT / "input" / f"{pid}.in").read_text())
        for comp in case["comparables"]:
            allcomps[comp["name"]] = comp
    series = {}
    for name, comp in allcomps.items():
        if "before release" in name.lower():
            continue
        s = entry.day_series(comp)
        if 1 in s and len(s) >= entry.MIN_HOURS:
            series[name] = s
    for min_ratio in (0.15, 0.30):
        sub = {k: v for k, v in series.items()
               if v[1] / max(v.values()) >= min_ratio}
        tb = tm = 0.0
        for name, s in sub.items():
            train = [v for k, v in sub.items() if k != name]
            curve = entry.fit_curve(train)
            p1 = s[1]
            zlist = entry.z_values(entry.LEVELS)
            qs = {h: entry.hour_quantiles(p1, h, curve, zlist) for h in range(2, 25)}
            hours = [{"hour": h, "q": qs[h]} for h in range(2, 25)]
            tm += score_case(s, hours)
            tb += score_case(s, baseline_hours(p1))
        n = len(sub)
        print(f"  h1/peak >= {min_ratio}: n={n} model={tm/n:.4f} baseline={tb/n:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
