"""Health check of a collected MIND regret run (the down4 block, submit_mind.sge; the main block, submit_mind_main.sge).  Run by hand after
collect_mind.py; it only reads the collect's output.

    cd ~/lowrank_bandit_experiments/realdata
    python3 check_mind_regret.py mind_regret > check_mind_regret.txt
    python3 check_mind_regret.py mind_regret_main > check_mind_regret_main.txt

It prints:
  1. the cells: how many are ok, failed, missing or stale, the software versions and the compute nodes;
  2. result rows with a missing or non-finite number;
  3. streams far from the others: regret above 5 times the median of their configuration;
  4. learning: regret per round over the last tenth of the horizon divided by regret per round over the first tenth
     after exploration, for our method, the two-track collection and each competitor's best configuration (well
     below 1 when it learns);
  5. our method: final estimation error, fitted rank, and the penalty constants chosen by cross-validation;
  6. ESTR's stage 1 (50 starts): how far the first start and the best of the first five end above the best start;
  7. each competitor's best configuration, marked when it sits at an edge of its grid;
  8. the time per cell, by arm and by node family.
It judges nothing: the comparisons stated in advance are in collected/verdict_mind.md (written by the collect), and the
collect has already checked the regret identity.  Only the Python standard library is used.
"""
import csv
import json
import math
import os
import re
import statistics
import sys


def read_tsv(run, name):
    with open(os.path.join(run, "collected", name)) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def number(text):
    """The float in a table cell, or None when the cell is empty or not a finite number."""
    try:
        x = float(text)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def value_at(curve, t):
    """Mean cumulative regret at the last grid point not after t (curve: sorted list of (t, mean))."""
    best = curve[0][1]
    for s, m in curve:
        if s <= t:
            best = m
    return best


def arm_of(key):
    return re.search(r"_b\d+_(.+)$", key).group(1)


def node_family(node):
    return re.sub(r"\d+$", "", node.split(".")[0])


def check(run):
    print("=" * 100)
    print("RUN", run)

    # 1. cells
    status = read_tsv(run, "status.tsv")
    counts = {}
    for s in status:
        counts[s["status"]] = counts.get(s["status"], 0) + 1
    print("\n1. cells:", len(status), counts)
    for s in [s for s in status if s["status"] != "ok"][:10]:
        print("   ", s["key"], s["status"], s["error"])
    versions, nodes = {}, {}
    for s in status:
        if s["status"] == "ok":
            v = "python %s, numpy %s, scipy %s" % (s["python"], s["numpy"], s["scipy"])
            versions[v] = versions.get(v, 0) + 1
            nodes[node_family(s["node"])] = nodes.get(node_family(s["node"]), 0) + 1
    for v, c in sorted(versions.items()):
        print("    made with %s: %d cells" % (v, c))
    print("    node families:", ", ".join("%s %d" % (k, c) for k, c in sorted(nodes.items())))

    # 2. missing or non-finite numbers
    rows = read_tsv(run, "replicates.tsv")
    broken = []
    for r in rows:
        fields = ["regret", "post_regret", "expected"]
        if r["method"] in ("Proposed", "Matched"):
            fields += ["theta_error_n", "rank_n"]
        for f in fields:
            if number(r[f]) is None:
                broken.append((r["block"], r["replication"], r["config_id"], f, r[f]))
    print("\n2. result rows: %d; missing or non-finite numbers: %d" % (len(rows), len(broken)))
    for b in broken[:10]:
        print("   ", b)

    # 3. streams far from the others
    groups = {}
    for r in rows:
        groups.setdefault((r["block"], r["config_id"]), []).append((int(r["replication"]), float(r["regret"])))
    far = []
    for (block, config), values in groups.items():
        med = statistics.median([v for b, v in values])
        for b, v in values:
            if med > 0 and v > 5 * med:
                far.append((block, config, b, v, med))
    print("\n3. streams with regret above 5 x the median of their configuration: %d of %d" % (len(far), len(rows)))
    far.sort(key=lambda x: -x[3] / x[4])
    for block, config, b, v, med in far[:15]:
        print("    %-6s %-26s b%d  regret %9.1f  median %8.1f  (x %.1f)" % (block, config, b, v, med, v / med))

    # 4. learning
    summary = read_tsv(run, "summary.tsv")
    best = read_tsv(run, "best.tsv")
    tau = {(s["block"], s["config_id"]): int(float(s["tau"] or 0)) for s in summary}
    curves = {}
    for c in read_tsv(run, "regret_curves.tsv"):
        curves.setdefault((c["block"], c["config_id"]), []).append((int(c["t"]), float(c["mean"])))
    shown = {(s["block"], s["config_id"]) for s in summary if s["config_id"] in ("A_t2", "two_track")}   # ours
    for b in best:                      # best.tsv also lists ESTR's best multiplier at each T1 ("ESTR@k"): skipped
        if "@" not in b["method"]:
            shown.add((b["block"], b["config_id"]))
    print("\n4. learning: (regret per round, last tenth of the horizon) / (regret per round, first tenth after exploration)")
    print("    %-6s %-26s %8s %10s %10s %7s" % ("block", "configuration", "tau", "early", "late", "ratio"))
    for block, config in sorted(shown):
        curve = sorted(curves[(block, config)])
        n = curve[-1][0]
        t0 = tau[(block, config)]
        t1 = t0 + n // 10
        early = (value_at(curve, t1) - value_at(curve, t0)) / max(t1 - t0, 1)
        late = (value_at(curve, n) - value_at(curve, n - n // 10)) / (n // 10)
        ratio = late / early if early > 0 else float("nan")
        flag = "   <- hardly learning" if ratio > 0.8 else ""
        print("    %-6s %-26s %8d %10.4f %10.4f %7.2f%s" % (block, config, t0, early, late, ratio, flag))

    # 5. our method
    print("\n5. our method (Proposed): final estimation error, fitted rank, penalty constants chosen by CV")
    print("    %-6s %-6s %5s %10s %-24s %8s %8s %s" % ("block", "arm", "B", "error", "fitted rank: count", "c = 1", "c = 1/64", "most often"))
    ours = {}
    for r in rows:
        if r["method"] == "Proposed":
            ours.setdefault((r["block"], r["config_id"]), []).append(r)
    for (block, config), rs in sorted(ours.items()):
        errors = [float(r["theta_error_n"]) for r in rs]
        ranks, constants = {}, []
        for r in rs:
            ranks[r["rank_n"]] = ranks.get(r["rank_n"], 0) + 1
            for s, c in json.loads(r["cv"] or "[]"):
                constants.append(c)
        if constants:
            at_top = sum(1 for c in constants if c == 1.0) / len(constants)
            at_bottom = sum(1 for c in constants if c == 1 / 64) / len(constants)
            common = max(set(constants), key=constants.count)
            cv_text = "%8.2f %8.2f 1/%g" % (at_top, at_bottom, round(1 / common))
        else:
            cv_text = "   (no CV checkpoint)"
        rank_text = ", ".join("%s: %d" % (k, v) for k, v in sorted(ranks.items(), key=lambda kv: float(kv[0])))
        print("    %-6s %-6s %5d %10.4f %-24s %s" % (block, config, len(rs), statistics.mean(errors), rank_text, cv_text))

    # 6. ESTR's stage 1 (the five multipliers of one T1 share one stage 1: count it once per stream)
    starts = {}
    for r in rows:
        if r["method"].startswith("ESTR") and number(r["bm_gap1"]) is not None:
            starts[(r["block"], r["method"], int(float(r["tau"])), r["replication"])] = (float(r["bm_gap1"]), float(r["bm_gap5"]))
    table = {}
    for (block, method, t1, b), v in starts.items():
        table.setdefault((block, method, t1), []).append(v)
    print("\n6. ESTR's stage 1 (50 starts): objective of the first start, and of the best of the first five, over the best")
    print("   start, minus 1: median, and the share of streams above 0.01")
    for (block, method, t1), vs in sorted(table.items()):
        g1 = [v[0] for v in vs]
        g5 = [v[1] for v in vs]
        print("    %-6s %-9s T1 %5d  B %3d  first start: median %.2g, > 0.01: %.2f   best of five: median %.2g, > 0.01: %.2f"
              % (block, method, t1, len(vs), statistics.median(g1), sum(1 for g in g1 if g > 0.01) / len(g1),
                 statistics.median(g5), sum(1 for g in g5 if g > 0.01) / len(g5)))

    # 7. best configurations at an edge of the grid
    print("\n7. each competitor's best configuration (in hindsight); an edge of its grid is marked")
    grid = {}
    for s in summary:
        grid.setdefault((s["block"], s["method"]), []).append(s)
    for b in sorted(best, key=lambda b: (b["block"], b["method"])):
        if "@" in b["method"]:
            continue
        configs = grid[(b["block"], b["method"])]
        row = next(s for s in configs if s["config_id"] == b["config_id"])
        edges = []
        for field, name in (("multiplier", "multiplier"), ("tau", "T1 (greedy comparators: tau)")):
            values = sorted({float(s[field]) for s in configs if number(s[field]) is not None})
            if len(values) > 1 and number(row[field]) is not None:
                if float(row[field]) == values[0]:
                    edges.append("%s at the lower edge" % name)
                elif float(row[field]) == values[-1]:
                    edges.append("%s at the upper edge" % name)
        print("    %-6s %-9s %-26s regret %9.2f  %s" % (b["block"], b["method"], b["config_id"], float(b["regret"]), "; ".join(edges)))

    # 8. time per cell, by arm and by node family
    times, families = {}, {}
    for s in status:
        if s["status"] == "ok" and number(s["elapsed"]) is not None:
            arm = re.sub(r"_k\d$", "", arm_of(s["key"]))            # the five T1 of ESTR (or the matched arm) together
            times.setdefault(arm, []).append(float(s["elapsed"]))
            families.setdefault((arm, node_family(s["node"])), []).append(float(s["elapsed"]))
    print("\n8. time per cell (seconds), by arm (the five T1 together), and by node family")
    total = 0.0
    for arm, ts in sorted(times.items()):
        total += sum(ts)
        by_node = "; ".join("%s %d (%d)" % (f, statistics.median(v), len(v)) for (a, f), v in sorted(families.items()) if a == arm)
        print("    %-10s cells %5d  median %8.0f  max %8.0f    by node family, median (cells): %s"
              % (arm, len(ts), statistics.median(ts), max(ts), by_node))
    print("    total %.1f CPU-hours" % (total / 3600))
    print()


for run in sys.argv[1:]:
    check(run)
