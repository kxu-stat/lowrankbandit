"""Health check of the collected MIND inference run (submit_mind_inference_eval.sge).  Run by hand after collect_mind_inference.py; it only
reads the collect's output.

    cd ~/lowrank_bandit_experiments/realdata
    python3 check_mind_inference.py mind_inference_evaluation > check_mind_inference.txt

For each world (main, down4) it prints:
  1. the streams: ok, failed, missing or stale, and the time per stream;
  2. the rows: how many, and how many have a missing or non-finite number; the rows at a solver cap (N7);
  3. the rows with |T| > 5, per choice of n0 and n, and the five largest;
  4. the variance estimate: V_hat / V_pqv at the largest n, per choice of n0 (median, 5 % and 95 % quantiles);
  5. the collection's regret along n;
  6. per choice of n0 and n, averaged over the 32 targets: |mean T|, sd T and the oracle coverage; and the targets
     whose oracle coverage at the largest n is below 0.90;
and then the reported n of both worlds (collected/reported_n.md).
It judges nothing: the verification standard is applied by the collect (collected/WORLD/verdict_mind_inference.md).
Only the Python standard library is used; the rows are read one at a time (a world has about 864,000).
"""
import csv
import gzip
import math
import os
import re
import statistics
import sys

WORLDS = ("main", "down4")


def number(text):
    """The float in a table cell, or None when the cell is empty or not a finite number."""
    try:
        x = float(text)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def quantile(values, p):
    values = sorted(values)
    return values[min(int(p * len(values)), len(values) - 1)]


def check(run):
    collected = os.path.join(run, "collected")
    status = list(csv.DictReader(open(os.path.join(collected, "status.tsv")), delimiter="\t"))
    for world in WORLDS:
        print("=" * 100)
        print("RUN %s, world %s" % (run, world))

        # 1. streams
        mine = [s for s in status if re.search(r"_%s_b\d+$" % world, s["key"])]
        counts = {}
        for s in mine:
            counts[s["status"]] = counts.get(s["status"], 0) + 1
        times = [float(s["elapsed"]) for s in mine if s["status"] == "ok" and number(s["elapsed"]) is not None]
        print("\n1. streams:", len(mine), counts)
        for s in [s for s in mine if s["status"] != "ok"][:10]:
            print("   ", s["key"], s["status"], s["error"])
        if times:
            print("    time per stream: median %.0f s, max %.0f s; total %.1f CPU-hours"
                  % (statistics.median(times), max(times), sum(times) / 3600))

        # 2-4. the rows, one at a time
        path = os.path.join(collected, world, "rows.tsv.gz")
        if not os.path.exists(path):
            print("\n    no %s: has the collect run?" % path)
            continue
        n_rows, broken, capped, capped_streams = 0, [], {}, set()
        large, top, ratios = {}, [], {}
        cells = {}
        with gzip.open(path, "rt") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                n_rows += 1
                n = int(float(r["n"]))
                cells[(r["arm"], n)] = cells.get((r["arm"], n), 0) + 1
                values = {k: number(r[k]) for k in ("mu", "V", "T", "V_pqv", "mu_star")}
                bad = [k for k, v in values.items() if v is None]
                if bad and len(broken) < 10:
                    broken.append((r["arm"], n, r["target"], r["rep"], ",".join(bad)))
                elif bad:
                    broken.append(None)
                if int(float(r["nuis_caps"] or 0)) > 0 or int(float(r["col_caps"] or 0)) > 0:
                    capped[(r["arm"], n)] = capped.get((r["arm"], n), 0) + 1
                if int(float(r["col_caps"] or 0)) > 0:
                    capped_streams.add(r["rep"])
                T = values["T"]
                if T is not None and abs(T) > 5:
                    large[(r["arm"], n)] = large.get((r["arm"], n), 0) + 1
                    top.append((abs(T), r["arm"], n, r["target"], r["rep"], T,
                                values["V"] / values["V_pqv"] if values["V"] and values["V_pqv"] else float("nan")))
                if values["V"] is not None and values["V_pqv"]:
                    ratios.setdefault((r["arm"], n), []).append(values["V"] / values["V_pqv"])
        top = sorted(top, reverse=True)[:5]
        summary = list(csv.DictReader(open(os.path.join(collected, world, "summary.tsv")), delimiter="\t"))
        arms = list(dict.fromkeys(s["arm"] for s in summary))         # the order of simulation.N0_CHOICES
        grid = sorted({n for a, n in cells})
        end = grid[-1]
        print("\n2. rows: %d (%d choices of n0 x %d n; per choice and n: %s); missing or non-finite mu, V, T, V_pqv or mu*: %d"
              % (n_rows, len(arms), len(grid), sorted(set(cells.values())), len(broken)))
        for b in [b for b in broken if b][:10]:
            print("   ", b)
        print("    rows at a solver cap (N7): %d; streams whose collection hit a cap: %d"
              % (sum(capped.values()), len(capped_streams)))
        for (a, n), c in sorted(capped.items())[:10]:
            print("      %-10s n %6d: %d rows" % (a, n, c))
        print("\n3. rows with |T| > 5: %d" % sum(large.values()))
        for (a, n), c in sorted(large.items()):
            print("      %-10s n %6d: %d of %d rows" % (a, n, c, cells[(a, n)]))
        for x, a, n, target, rep, T, ratio in top:
            print("      largest: %-10s n %6d %-20s stream %s  T %8.2f  V_hat/V_pqv %.2f" % (a, n, target, rep, T, ratio))
        print("\n4. V_hat / V_pqv at n = %d (every target and stream): median [5 %%, 95 %%]" % end)
        for a in arms:
            v = ratios.get((a, end), [])
            if v:
                print("      %-10s %.3f [%.3f, %.3f]" % (a, statistics.median(v), quantile(v, 0.05), quantile(v, 0.95)))

        # 5. regret of the collection
        print("\n5. the collection's cumulative expected regret (both tracks), over the streams")
        for g in csv.DictReader(open(os.path.join(collected, world, "regret.tsv")), delimiter="\t"):
            print("      n %6d  B %4d  regret %9.1f (se %.1f)  per round %.4f"
                  % (int(float(g["n"])), int(float(g["B"])), float(g["regret"]), float(g["se"]), float(g["regret"]) / float(g["n"])))

        # 6. the 32 targets averaged, per choice of n0 and n
        print("\n6. averaged over the 32 targets: |mean T|, sd T, oracle coverage (each n: the same streams stopped at n)")
        print("      %-10s " % "n0" + " ".join("%22s" % ("n = %d" % n) for n in grid))
        low = []
        for a in arms:
            text = []
            for n in grid:
                z = [s for s in summary if s["arm"] == a and int(float(s["n"])) == n]
                m = [abs(number(s["mean_T"])) for s in z if number(s["mean_T"]) is not None]
                sd = [number(s["sd_T"]) for s in z if number(s["sd_T"]) is not None]
                cv = [number(s["cover"]) for s in z if number(s["cover"]) is not None]
                text.append("%6.2f %6.2f %6.3f" % (statistics.mean(m) if m else float("nan"),
                                                   statistics.mean(sd) if sd else float("nan"),
                                                   statistics.mean(cv) if cv else float("nan")))
                if n == end:
                    low += [(a, s["target"], number(s["cover"])) for s in z
                            if number(s["cover"]) is not None and number(s["cover"]) < 0.90]
            print("      %-10s " % a + " ".join("%22s" % t for t in text))
        print("    targets with oracle coverage below 0.90 at n = %d: %d" % (end, len(low)))
        for a, target, c in low[:15]:
            print("      %-10s %-22s %.3f" % (a, target, c))
        print()

    # the reported n
    print("=" * 100)
    print("the reported n (collected/reported_n.md)")
    for line in open(os.path.join(collected, "reported_n.md")):
        if line.startswith("| world") or line.startswith("| main world") or line.startswith("| down-sampled"):
            print("   ", line.rstrip())
    print()


for run in sys.argv[1:]:
    check(run)
