"""Health check of collected regret runs.  Run by hand after collect_regret.py; it only reads the collect's output.

    cd ~/lowrank_bandit_experiments/regret
    python3 check_regret.py regret_gaussian_approx regret_binomial_approx > check_approx.txt

For every run directory it prints:
  1. the cells: how many are ok, failed, missing or stale, the software versions they were made with and the nodes;
  2. result rows with a missing or non-finite number;
  3. replications far from the others: regret above 5 times the median of their (case, configuration);
  4. learning: regret per round over the last tenth of the horizon divided by regret per round over the first tenth
     after exploration, for our method and each competitor's best configuration (well below 1 when it learns);
  5. our method: final estimation error, fitted rank, and the penalty constants chosen by cross-validation;
  6. ESTR's stage 1: the share of recoveries whose subspace misses the truth (bm_sin > 0.5), per case and T1;
  7. each competitor's best configuration, marked when it sits at an edge of its grid;
  8. the time per cell, by arm and by node family;
  9. (tier scale) its cells at n = 10,000 against the main grid's cells of the same case (same seeds), when
     regret_gaussian/ sits beside the run.
It judges nothing: the predictions are judged by verdict_regret.py.  Only the Python standard library is used.
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


def node_family(node):
    """node017.cluster.example -> node (the nodes of one family share their hardware)."""
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
    bad = [s for s in status if s["status"] != "ok"]
    for s in bad[:10]:
        print("   ", s["key"], s["status"], s["error"])
    for e in read_tsv(run, "environment.tsv"):
        print("    made with python %s, numpy %s, scipy %s: %s cells" % (e["python"], e["numpy"], e["scipy"], e["cells"]))
    nodes = {}
    for s in status:
        if s["status"] == "ok":
            nodes[node_family(s["node"])] = nodes.get(node_family(s["node"]), 0) + 1
    print("    node families:", ", ".join("%s %d" % (k, c) for k, c in sorted(nodes.items())))

    # 2. missing or non-finite numbers
    rows = read_tsv(run, "replicates.tsv")
    broken = []
    for r in rows:
        fields = ["regret", "post_regret"]
        if r["method"] in ("Proposed", "Matched"):
            fields += ["err_n", "rank_n"]
        for f in fields:
            if number(r[f]) is None:
                broken.append((r["case_id"], r["replication"], r["config_id"], f, r[f]))
    print("\n2. result rows: %d; missing or non-finite numbers: %d" % (len(rows), len(broken)))
    for b in broken[:10]:
        print("   ", b)

    # 3. replications far from the others
    groups = {}
    for r in rows:
        groups.setdefault((r["case_id"], r["config_id"]), []).append((int(r["replication"]), float(r["regret"])))
    far = []
    for (case, config), values in groups.items():
        med = statistics.median([v for b, v in values])
        for b, v in values:
            if med > 0 and v > 5 * med:
                far.append((case, config, b, v, med))
    print("\n3. replications with regret above 5 x the median of their (case, configuration): %d of %d" % (len(far), len(rows)))
    far.sort(key=lambda x: -x[3] / x[4])
    for case, config, b, v, med in far[:15]:
        print("    %-30s %-26s b%d  regret %9.1f  median %8.1f  (x %.1f)" % (case, config, b, v, med, v / med))

    # 4. learning
    summary = read_tsv(run, "summary.tsv")
    best = read_tsv(run, "best.tsv")
    tau = {(s["case_id"], s["config_id"]): int(float(s["tau"] or 0)) for s in summary}
    curves = {}
    for c in read_tsv(run, "regret_curves.tsv"):
        curves.setdefault((c["case_id"], c["config_id"]), []).append((int(c["t"]), float(c["mean"])))
    shown = set()
    for s in summary:                   # our method at the reported c_tau = 1/20
        if s["config_id"] == "A_t2":
            shown.add((s["case_id"], "A_t2"))
    for b in best:                      # best.tsv also lists ESTR's best multiplier at each T1 ("ESTR@k"): skipped
        if b["method"] != "Matched" and "@" not in b["method"]:
            shown.add((b["case_id"], b["config_id"]))
    print("\n4. learning: (regret per round, last tenth of the horizon) / (regret per round, first tenth after exploration)")
    print("    %-30s %-26s %8s %10s %10s %7s" % ("case", "configuration", "tau", "early", "late", "ratio"))
    for case, config in sorted(shown):
        curve = sorted(curves[(case, config)])
        n = curve[-1][0]
        t0 = tau[(case, config)]
        t1 = t0 + n // 10
        early = (value_at(curve, t1) - value_at(curve, t0)) / max(t1 - t0, 1)
        late = (value_at(curve, n) - value_at(curve, n - n // 10)) / (n // 10)
        ratio = late / early if early > 0 else float("nan")
        flag = "   <- hardly learning" if ratio > 0.8 else ""
        print("    %-30s %-26s %8d %10.4f %10.4f %7.2f%s" % (case, config, t0, early, late, ratio, flag))

    # 5. our method
    print("\n5. our method (Proposed): final estimation error, fitted rank, penalty constants chosen by CV")
    print("    %-30s %-6s %5s %10s %-22s %8s %8s %s" % ("case", "arm", "B", "error", "fitted rank: count", "c = 1", "c = 1/64", "most often"))
    ours = {}
    for r in rows:
        if r["method"] == "Proposed":
            ours.setdefault((r["case_id"], r["config_id"]), []).append(r)
    for (case, config), rs in sorted(ours.items()):
        errors = [float(r["err_n"]) for r in rs]
        ranks = {}
        constants = []
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
        rank_text = ", ".join("%s: %d" % (k, v) for k, v in sorted(ranks.items()))
        print("    %-30s %-6s %5d %10.4f %-22s %s" % (case, config, len(rs), statistics.mean(errors), rank_text, cv_text))

    # 6. ESTR's stage 1 (the five multipliers of one T1 share one recovery: count it once)
    recoveries = {}
    for r in rows:
        if r["method"].startswith("ESTR") and number(r["bm_sin"]) is not None:
            recoveries[(r["case_id"], r["method"], int(float(r["tau"])), r["replication"])] = float(r["bm_sin"])
    table = {}
    for (case, method, t1, b), v in recoveries.items():
        table.setdefault((case, method, t1), []).append(v)
    print("\n6. ESTR's stage 1: subspace error bm_sin per (case, T1): median, and the share above 0.5")
    for (case, method, t1), vs in sorted(table.items()):
        share = sum(1 for v in vs if v > 0.5) / len(vs)
        print("    %-30s %-8s T1 %5d  B %3d  median %.2f  share > 0.5: %.2f" % (case, method, t1, len(vs), statistics.median(vs), share))

    # 7. best configurations at an edge of the grid (multiplier index m00 / m04, T1 index t00 / t04)
    print("\n7. each competitor's best configuration (in hindsight); an edge of the grid is marked")
    for b in sorted(best, key=lambda b: (b["case_id"], b["method"])):
        if b["method"] in ("Proposed", "Matched") or "@" in b["method"]:
            continue
        edges = []
        m = re.search(r"_m(\d\d)", b["config_id"])
        t = re.search(r"_t(\d\d)", b["config_id"])
        if m and m.group(1) in ("00", "04"):
            edges.append("multiplier at the %s edge" % ("lower" if m.group(1) == "00" else "upper"))
        if t and t.group(1) in ("00", "04"):
            edges.append("T1 at the %s edge" % ("lower" if t.group(1) == "00" else "upper"))
        print("    %-30s %-8s %-26s regret %9.2f  %s" % (b["case_id"], b["method"], b["config_id"], float(b["regret"]), "; ".join(edges)))

    # 8. time per cell, by arm and by node family (a node full of our tasks runs a binary cell up to 3 times slower)
    times, families = {}, {}
    for s in status:
        found = re.search(r"_b\d+_(.+)$", s["key"])
        if s["status"] == "ok" and found and number(s["elapsed"]) is not None:
            arm = re.sub(r"_k\d$", "", found.group(1))              # the five T1 of ESTR (or the matched arm) together
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

    # 9. the tier scale: its cells at n = 10,000 are the main grid's cells of the same case (same seeds), if the main
    # run sits beside it (regret_gaussian/)
    main = os.path.join(os.path.dirname(os.path.abspath(run)), "regret_gaussian", "cells")
    mine = sorted(f for f in os.listdir(os.path.join(run, "cells")) if "_n010000_" in f and f.endswith(".json"))
    if mine and os.path.isdir(main):
        equal, differ, absent, worst = 0, 0, 0, 0.0
        for f in mine:
            g = os.path.join(main, f.replace("_n010000", ""))
            if not os.path.exists(g):
                absent += 1
                continue
            a = {r["config_id"]: r["regret"] for r in json.load(open(os.path.join(run, "cells", f)))["rows"]}
            b = {r["config_id"]: r["regret"] for r in json.load(open(g))["rows"]}
            rel = max(abs(a[k] - b.get(k, float("inf"))) / max(1.0, abs(a[k])) for k in a)
            worst = max(worst, rel)
            equal += rel <= 1e-6
            differ += rel > 1e-6
        print("\n9. tier scale: its cells at n = 10,000 against the main grid's (same seeds): %d equal (1e-6), %d differ, "
              "%d not in the main run; largest relative difference %.2g (a difference: different node hardware can "
              "flip an argmax; selftest.py explains)" % (equal, differ, absent, worst))
    print()


for run in sys.argv[1:]:
    check(run)
