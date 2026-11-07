"""Health check of the collected inference simulations (submit_inference.sge).  Run by hand after collect_inference.py; it only
reads the collect's output.

    cd ~/lowrank_bandit_experiments/inference
    python3 check_inference.py inference_results > check_inference.txt

It prints:
  1. the tasks: complete, partial or missing, and the software versions;
  2. per table (block, covariate law), read one row at a time: the rows, those with a missing or non-finite mu, V,
     T or V_pqv, those at a solver cap (N7), those with |T| > 5 (and the five largest), and the collection time per
     replicate by node family;
  3. per block and choice of n0 at the block's largest n, averaged over the targets: |mean T|, sd T, the oracle
     coverage and V_hat / V_pqv; and the target whose coverage is farthest from 0.95;
  4. the two-track collection's regret along n, and (when the collect was given the regret study's curves) its ratio
     to the single-track Algorithm 1 at the same n;
  5. the collect's table of verdicts (inference_verdict.md: every choice of n0 under the same standard).
It judges nothing.  Only the Python standard library is used; a table of a million rows takes about ten seconds.
"""
import csv
import math
import os
import re
import statistics
import sys

ROW_COLUMNS = ("rep", "arm", "target", "mu", "V", "T", "V_pqv", "nuis_caps", "col_caps", "t_collect", "node")


def number(text):
    """The float in a table cell, or None when the cell is empty or not a finite number."""
    try:
        x = float(text)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def node_family(node):
    return re.sub(r"\d+$", "", node.split(".")[0])


def scan(path):
    """One merged table, one line at a time: counts, the largest |T| and the collection time of each replicate."""
    out = dict(rows=0, broken=0, capped={}, large={}, top=[], reps={}, cells={})
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        grid = "N" if "n" not in header else "n"                   # I3's grid is N (post-exploration rounds)
        col = {c: header.index(c) for c in ROW_COLUMNS + (grid,) if c in header}
        for line in f:
            v = line.rstrip("\n").split("\t")
            out["rows"] += 1
            arm, n = v[col["arm"]], int(float(v[col[grid]]))
            out["cells"][(arm, n)] = out["cells"].get((arm, n), 0) + 1
            if any(number(v[col[c]]) is None for c in ("mu", "V", "T", "V_pqv")):
                out["broken"] += 1
            if number(v[col["nuis_caps"]]) or number(v[col["col_caps"]]):
                out["capped"][arm] = out["capped"].get(arm, 0) + 1
            T = number(v[col["T"]])
            if T is not None and abs(T) > 5:
                out["large"][arm] = out["large"].get(arm, 0) + 1
                out["top"].append((abs(T), arm, n, v[col["target"]], v[col["rep"]], T))
                if len(out["top"]) > 1000:
                    out["top"] = sorted(out["top"], reverse=True)[:5]
            rep = v[col["rep"]]
            if rep not in out["reps"] and "t_collect" in col:
                out["reps"][rep] = (number(v[col["t_collect"]]), node_family(v[col["node"]]) if "node" in col else "?")
    out["top"] = sorted(out["top"], reverse=True)[:5]
    return out


def check(run):
    collected = os.path.join(run, "collected")
    print("=" * 100)
    print("RUN", run)

    # 1. tasks
    status = list(csv.DictReader(open(os.path.join(collected, "status.tsv")), delimiter="\t"))
    counts = {}
    for s in status:
        counts[s["status"]] = counts.get(s["status"], 0) + 1
    print("\n1. tasks:", len(status), counts)
    for s in [s for s in status if s["status"] != "ok"][:10]:
        print("   ", s["key"], s["status"], "%s of %s replicates" % (s["replicates"], s["planned"]))
    if os.path.exists(os.path.join(collected, "environment.tsv")):
        for e in csv.DictReader(open(os.path.join(collected, "environment.tsv")), delimiter="\t"):
            print("    made with python %s, numpy %s, scipy %s" % (e["python"], e["numpy"], e["scipy"]))

    # 2. the merged tables, one row at a time
    tables = sorted(f for f in os.listdir(collected) if re.match(r"I\w+_\w+\.tsv$", f))
    print("\n2. the rows of each table (block_law.tsv)")
    total = 0.0
    for name in tables:
        z = scan(os.path.join(collected, name))
        per_cell = sorted(set(z["cells"].values()))
        print("  %s: %d rows (%d choices of n0 x %d n; rows per choice and n: %s); missing or non-finite mu, V, T, V_pqv: %d"
              % (name, z["rows"], len({a for a, n in z["cells"]}), len({n for a, n in z["cells"]}), per_cell, z["broken"]))
        print("      at a solver cap (N7): %s" % (", ".join("%s %d" % kv for kv in sorted(z["capped"].items())) or "none"))
        print("      |T| > 5: %s" % (", ".join("%s %d" % kv for kv in sorted(z["large"].items())) or "none"))
        for x, arm, n, target, rep, T in z["top"]:
            print("        largest: %-10s n %6d %-10s replicate %s  T %8.2f" % (arm, n, target, rep, T))
        times = {}
        for t, fam in z["reps"].values():
            if t is not None:
                times.setdefault(fam, []).append(t)
                total += t
        print("      collection time per replicate (s), by node family: %s"
              % ("; ".join("%s median %.0f, max %.0f (%d)" % (f, statistics.median(t), max(t), len(t))
                           for f, t in sorted(times.items())) or "not recorded"))
    print("    total collection time %.1f CPU-hours (Steps 2-4 not included)" % (total / 3600))

    # 3. per block and choice of n0 at the largest n
    summary = list(csv.DictReader(open(os.path.join(collected, "summary.tsv")), delimiter="\t"))
    print("\n3. at each table's largest n, averaged over its targets: |mean T|, sd T, oracle coverage, V_hat / V_pqv;")
    print("   and the target whose coverage is farthest from 0.95")
    for block, regime in sorted({(s["block"], s["regime"]) for s in summary}):
        rows = [s for s in summary if s["block"] == block and s["regime"] == regime]
        end = max(int(float(s["n"])) for s in rows)
        B = max(int(float(s["B"])) for s in rows)
        print("  %s %s, n = %d, B = %d" % (block, regime, end, B))
        for arm in dict.fromkeys(s["arm"] for s in rows):
            z = [s for s in rows if s["arm"] == arm and int(float(s["n"])) == end]
            vals = {c: [number(s[c]) for s in z if number(s[c]) is not None] for c in ("mean_T", "sd_T", "cover", "Vhat_over_Vpqv")}
            far = max((s for s in z if number(s["cover"]) is not None), key=lambda s: abs(float(s["cover"]) - 0.95), default=None)
            mean = lambda c, f=lambda x: x: statistics.mean([f(x) for x in vals[c]]) if vals[c] else float("nan")
            print("      %-10s |mean T| %5.2f  sd T %5.2f  cover %5.3f  V_hat/V_pqv %5.3f   farthest: %s %s"
                  % (arm, mean("mean_T", abs), mean("sd_T"), mean("cover"), mean("Vhat_over_Vpqv"),
                     far["target"] if far else "-", "%.3f" % float(far["cover"]) if far else ""))

    # 4. regret of the two-track collection
    print("\n4. the two-track collection's cumulative expected regret (first, middle and last n)")
    regret = list(csv.DictReader(open(os.path.join(collected, "regret.tsv")), delimiter="\t"))
    for block, regime in sorted({(g["block"], g["regime"]) for g in regret}):
        g = sorted((g for g in regret if g["block"] == block and g["regime"] == regime), key=lambda g: int(float(g["n"])))
        pick = [g[0], g[len(g) // 2], g[-1]] if len(g) > 2 else g
        print("  %-6s %-8s " % (block, regime) + "; ".join("n %d: %.1f (per round %.4f)" % (
            int(float(x["n"])), float(x["regret"]), float(x["regret"]) / int(float(x["rounds"]))) for x in pick))
    path = os.path.join(collected, "regret_vs_single.tsv")
    if os.path.exists(path):
        print("  two-track / single-track regret at the same n, 95 % interval (last n of each case):")
        comp = list(csv.DictReader(open(path), delimiter="\t"))
        for block, regime in sorted({(c["block"], c["regime"]) for c in comp}):
            c = max((c for c in comp if c["block"] == block and c["regime"] == regime), key=lambda c: int(float(c["n"])))
            print("    %-6s %-8s n %6d  %s: ratio %.3f [%.3f, %.3f]%s" % (
                block, regime, int(float(c["n"])), c["case"], float(c["ratio"]), float(c["ci_lo"]), float(c["ci_hi"]),
                "; after exploration %.3f [%.3f, %.3f]" % (float(c["ratio_post"]), float(c["post_lo"]), float(c["post_hi"]))
                if c.get("ratio_post") else ""))
    else:
        print("  (no regret_vs_single.tsv: the collect was run without --regret-curves)")

    # 5. the verdict table of the collect
    print("\n5. the collect's verdicts by choice of n0 (inference_verdict.md)")
    inside = False
    for line in open(os.path.join(collected, "inference_verdict.md")):
        if line.startswith("## Verdict:"):
            print("  " + line.strip())
        if line.startswith("## Verdicts by choice of n0"):
            inside = True
            continue
        if inside and line.startswith("## "):
            break
        if inside and line.strip():
            print("  " + line.rstrip())
    print()


for run in sys.argv[1:]:
    check(run)
