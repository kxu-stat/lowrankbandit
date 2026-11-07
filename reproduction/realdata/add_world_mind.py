#!/usr/bin/env python3
"""Add the constants of a down-sampled click model to a MIND setup file.  Run once, by hand, before any run:

    python3 add_world_mind.py SETUP_IN PREPARED_NPZ W OUT_NPZ

The world's truth is the setup's with the intercept raised by log(1 / W) (mind.downsampled: the click model of logs
that keep every click and each non-click with probability W).  Every constant the methods are given is recomputed from
that truth by the rules of setup_mind.py, so that the regret study can run in that world: Algorithm 1's balls, the
competitors' declared norms (logistic and 2y - 1, mind.declared and mind.declared_estr) and the main-effects-only
comparator's ball.  The keys are written with the prefix "w<1/W>_" (W = 1/4: "w4_glm_S_F_cmx", ...); the setup's own
keys are kept unchanged.  As a check, the main world's constants are recomputed by the same function and must equal the
file's.  Writes OUT_NPZ and OUT_check.txt.
"""
import os, sys, time, hashlib
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
import numpy as np
from scipy.special import expit
import mind

WORLD_KEYS = ("radius_Theta", "radii_main", "main_only_ball", "glm_S_F_cmx", "glm_S2_cmx", "glm_S_F_c", "glm_S2_c",
              "pm1_S_F_cmx", "pm1_S2_cmx", "pm1_S_F_c", "pm1_S2_c", "glm_S_F_estr", "glm_S2_estr", "pm1_S_F_estr",
              "pm1_S2_estr")


def constants(ff, gamma, Theta, D):
    """The constants the methods are given, from the truth (gamma, Theta), as setup_mind.py computes them."""
    N = mind.N_COV
    starts = np.cumsum((0,) + mind.group_sizes("cmx", D))
    out = dict(radius_Theta=2 * np.linalg.norm(Theta),
               radii_main=np.array([2 * np.linalg.norm(gamma[a:b]) for a, b in zip(starts[:-1], starts[1:])]))
    p_star = expit(ff["C"] @ gamma + np.einsum("ij,ij->i", ff["M"] @ Theta, ff["X"]))
    g_c, _, _ = mind.fit_logistic(dict(ff, C=ff["C"][:, :N], y=p_star), 0, (np.zeros(N), None))
    out["main_only_ball"] = 2 * np.linalg.norm(g_c)
    for v, g in (("cmx", gamma), ("c", gamma[:N])):
        out["glm_S_F_%s" % v], out["glm_S2_%s" % v] = mind.declared(g, Theta, v, D)
        ff_v = dict(ff, C=ff["C"] if v == "cmx" else ff["C"][:, :N], y=p_star)
        out["pm1_S_F_%s" % v], out["pm1_S2_%s" % v] = mind.declared(*mind.linear_fit(ff_v), v, D)
    out["glm_S_F_estr"], out["glm_S2_estr"] = mind.declared_estr(gamma, Theta, D)
    out["pm1_S_F_estr"], out["pm1_S2_estr"] = mind.declared_estr(*mind.linear_fit(dict(ff, y=p_star)), D)
    return out, float(p_star.mean())


def main():
    setup_in, prep_path, w, out_path = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
    t0, lines = time.time(), []

    def out(text):
        print(text, flush=True)
        lines.append(text)

    setup = dict(np.load(setup_in))
    D = int(setup["D"])
    data = mind.load(prep_path, D, setup)
    ff = mind.full_feedback(data, "cmx")
    out("setup %s (package %s), D = %d; %d labeled pairs" % (setup_in, setup["package"], D, ff["n_pairs"]))
    main_c, p_main = constants(ff, setup["truth_gamma"], setup["truth_Theta"], D)
    worst = max(float(np.max(np.abs(np.asarray(main_c[k]) - np.asarray(setup[k])))) for k in WORLD_KEYS)
    out("check, main world: largest difference from the file's constants %.2e (mean click probability %.4f)"
        % (worst, p_main))
    assert worst < 1e-8, "the main world's constants are not reproduced"
    world = mind.downsampled(setup, w)
    tag = "w%d_" % round(1 / w)
    wc, p_w = constants(ff, world["truth_gamma"], world["truth_Theta"], D)
    save = dict(setup)
    for k in WORLD_KEYS:
        save[tag + k] = np.asarray(wc[k])
        out("  %-22s main %-28s %s" % (tag + k, np.round(np.asarray(setup[k]), 4), np.round(np.asarray(wc[k]), 4)))
    save[tag + "intercept_shift"] = np.log(1 / w)
    save["package"] = mind.SETUP_VERSION                                 # the format of the file written here
    out("world W = %g: intercept %+.3f -> %+.3f, mean click probability %.4f (main %.4f)  (%.0f s)"
        % (w, setup["truth_gamma"][0], world["truth_gamma"][0], p_w, p_main, time.time() - t0))
    np.savez(out_path, **save)
    sha = hashlib.sha256(open(out_path, "rb").read()).hexdigest()
    out("wrote %s, sha256 %s" % (out_path, sha))
    open(out_path.replace(".npz", "_check.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
