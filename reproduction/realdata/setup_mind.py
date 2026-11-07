#!/usr/bin/env python3
"""Build, once and before any run, everything the MIND study takes from the data: the kept users and their vectors,
the covariates' standardization, the truth of the semi-synthetic world, the constants the methods are given, and the
targets of the inference.  Run by hand; the cluster then needs only the prepared file and the setup file this writes.

    python3 setup_mind.py PREPARED_NPZ MIND_SMALL_NPZ GLOVE_NPZ D OUT_NPZ

MIND_SMALL_NPZ holds the users' histories and GLOVE_NPZ the 300-d title encodings, the files the prepared file was made
from.  Writes OUT_NPZ and OUT_check.txt (every number below).

One data set throughout, as in statistics papers that build a semi-synthetic study on real data: all kept users.
  users         kept: at least HMIN = 10 encoded history articles.
  user vectors  the mean of the user's last <= 50 encoded history articles (GloVe 6B 300-d title averages), projected
                on the arm-weighted PCA of the displayed articles (D components), re-whitened on the kept users'
                eligible (K >= 5) rounds and scaled to length sqrt(D) (so E[x x'] is about I, as E[m m'] = I for the
                whitened article vectors: the paper's normalization, mind.py).
  covariates    intercept, same-category share, log impression size; the last two standardized over the displayed
                pairs of the eligible impressions (mind.covariate_stats).
  rank          declared r = RANK = 2.  The eigenvalue-ratio estimate (Lam and Yao 2012, eq. 2.8; argmax over
                k <= D / 2 of lambda_k / lambda_{k+1}, lambda the squared singular values of the unrestricted fit's
                Theta) is printed beside it.
  truth         the rank-r maximum-likelihood logistic fit of cmx on the full feedback (all labeled pairs of the
                eligible K >= 5 impressions, unweighted): the truth of the semi-synthetic world.
  constants     taken from the truth, as the simulations take them from theirs: Algorithm 1's balls (twice the truth's
                norms per block, design units); the logistic competitors' declared norms (the truth's, in the model
                each fits and in the units of the data it sees: mind.declared for OFUL, mind.declared_estr for ESTR,
                the norms of ESTR's augmented truth, which carries the m and x main effects); the 2y - 1
                competitors' declared norms (the least-squares fit of 2 p* - 1, the model they fit, arranged the same
                way); the main-effects-only comparator's ball (twice the norm of the logistic fit of c to p*).
  inference     the planning value kappa_D = p_D (1 - p_D) of the design-stage split, p_D the logged click rate of the
                displayed pairs of the eligible K = 10 impressions (a planning value of the information per round in
                the paper's normalization, the rows' own; inference/algorithms.design_split); and the targets: for
                the 8 categories with the most displayed pairs, linear functionals of theta* built from the category's
                average article and the average user and the category's average reader (`targets`).
All coefficients are in design units (mind.py): the rows z = (c, m, x, vec(m x')) with every block's second moments
about I.
"""
import os, sys, time, hashlib
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")          # one thread: the fits do not depend on the machine's thread count
import numpy as np
from scipy.special import expit
import mind

HMIN = 10                     # kept users: at least 10 encoded history articles
RANK = 2                      # the declared rank of the truth (and of ESTR)
HISTORY = 50                  # the last <= 50 encoded history articles
CATEGORIES = 8                # the categories with targets
TARGET_KINDS = ("or", "orx", "ort", "lvl")


def user_vectors(prep, hist, glove, data, D):
    """The kept users' user vectors, re-whitened on the kept users' eligible (K >= 5) rounds, length sqrt(D) (rows of
    users without an encoded history are 0)."""
    enc, has = glove["enc"].astype(np.float64), glove["n_found"] > 0
    c_news, c_imp, imp_size = prep["c_news"], prep["c_imp"], prep["imp_size"].astype(int)
    shown = np.zeros(len(enc), bool)
    shown[c_news] = True
    fit = shown & has
    wa = np.bincount(c_news, weights=1.0 / imp_size[c_imp], minlength=len(enc))[fit]
    wa /= wa.sum()                                              # arm weights: each impression has total weight 1
    mu = wa @ enc[fit]
    _, _, Vt = np.linalg.svd(np.sqrt(wa)[:, None] * (enc[fit] - mu), full_matrices=False)
    h_ptr, h_news = hist["h_ptr"], hist["h_news"]
    Z = np.zeros((len(hist["users"]), D))
    for u in range(len(hist["users"])):
        h = h_news[max(h_ptr[u], h_ptr[u + 1] - HISTORY): h_ptr[u + 1]]
        h = h[has[h]]
        if len(h):
            Z[u] = (enc[h].mean(0) - mu) @ Vt[:D].T
    rounds = data["imp_user"][data["eligible_K5"]]
    w, Q = np.linalg.eigh(np.cov(Z[rounds].T))
    X = (Z - Z[rounds].mean(0)) @ Q @ np.diag(w ** -0.5) @ Q.T
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return np.where(nrm > 0, np.sqrt(D) * X / np.maximum(nrm, 1e-300), 0.0)


def z_rows(data, pairs, m, x):
    """The design row z = (c, m, x, vec(m x')) (vec column-major, as inference/algorithms) for the covariates c
    averaged over `pairs` and given m and x."""
    c = mind.covariates(data, pairs, "c").mean(0)
    return np.concatenate([c, m, x, np.outer(m, x).reshape(-1, order="F")])


def targets(data, keep):
    """The inference targets mu* = <A, theta*>.  For each of the CATEGORIES categories with the most displayed pairs
    among the eligible K = 10 impressions: m_k the category's average displayed article; x_all the average user of
    those impressions; x_k the average reader of k (a kept user with at least 3 logged clicks, at least half of them
    on articles of k); and the loadings
        or_k   z(c_k, m_k, x_k) - z(c_k^all, m_k, x_all): the log odds ratio of a click on the category's average
               article for its average reader against the average user (c the covariates averaged over the
               category's displayed pairs, for its readers and for everyone);
        orx_k  the same without the c block: the part carried by the user vectors (x and Theta blocks);
        ort_k  its interaction part only: m_k' Theta (x_k - x_all);
        lvl_k  z(0, m_k, x_k): the click log-odds of the category's average article for its average reader, without
               the c block (m, x and Theta blocks).
    Returns (names, loadings, information per category)."""
    imps = np.flatnonzero(data["eligible_K10"])
    pairs = np.concatenate([data["first"][i] + np.arange(data["imp_size"][i]) for i in imps])
    cat = data["news_cat"].astype(int)
    clicked = np.flatnonzero(data["c_lab"] == 1)
    profile = np.zeros((len(keep), cat.max() + 1))
    np.add.at(profile, (data["imp_user"][data["c_imp"][clicked]], cat[data["c_news"][clicked]]), 1.0)
    total = profile.sum(1)
    users = data["imp_user"][imps]
    x_all = data["X"][users].mean(0)
    count = np.bincount(cat[data["c_news"][pairs]], minlength=cat.max() + 1)
    names, As, info = [], [], {}
    for k in np.argsort(-count)[:CATEGORIES]:
        pk = pairs[cat[data["c_news"][pairs]] == k]
        reader = keep & (total >= 3) & (profile[:, k] >= 0.5 * total)
        mk = data["M"][data["c_news"][pk]].mean(0)
        xk = data["X"][users[reader[users]]].mean(0)
        z_read = z_rows(data, pk[reader[data["imp_user"][data["c_imp"][pk]]]], mk, xk)
        z_all = z_rows(data, pk, mk, x_all)
        A_or = z_read - z_all
        A_orx = A_or.copy(); A_orx[:mind.N_COV] = 0.0
        A_ort = A_orx.copy(); A_ort[:mind.N_COV + 2 * data["D"]] = 0.0
        A_lvl = z_read.copy(); A_lvl[:mind.N_COV] = 0.0
        name = CAT_NAMES[k] if k < len(CAT_NAMES) else "category%d" % k
        for kind, A in zip(TARGET_KINDS, (A_or, A_orx, A_ort, A_lvl)):
            names.append("%s_%s" % (kind, name)); As.append(A)
        info[name] = dict(share=float(count[k] / count.sum()), readers=int(reader.sum()))
    return names, np.array(As), info


CAT_NAMES = ["autos", "entertainment", "finance", "foodanddrink", "health", "kids", "lifestyle", "middleeast", "movies",
             "music", "news", "northamerica", "sports", "travel", "tv", "video", "weather"]   # the prepared category codes


def main():
    prep_path, hist_path, glove_path, D, out_path = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
    t0 = time.time()
    lines = []

    def out(text):
        print(text, flush=True)
        lines.append(text)

    sha = lambda path: hashlib.sha256(open(path, "rb").read()).hexdigest()
    data_sha = sha(prep_path)
    out("setup (package %s), D = %d; prepared %s (sha256 %s...), histories %s (%s...), encodings %s (%s...)"
        % (mind.SETUP_VERSION, D, prep_path, data_sha[:16], hist_path, sha(hist_path)[:16], glove_path, sha(glove_path)[:16]))
    prep = np.load(prep_path)
    data = mind.load(prep_path, D)
    keep = prep["hist_n"].astype(int) >= HMIN
    for K in (5, 10):
        data["eligible_K%d" % K] = data["eligible_K%d" % K] & keep[data["imp_user"]]
    data["X"] = user_vectors(prep, np.load(hist_path, allow_pickle=True), np.load(glove_path, allow_pickle=True), data, D)
    data["cov_stats"] = mind.covariate_stats(data)
    out("kept users (history >= %d): %d of %d; eligible impressions K = 5: %d, K = 10: %d; covariate standardization %s"
        % (HMIN, keep.sum(), len(keep), data["eligible_K5"].sum(), data["eligible_K10"].sum(), data["cov_stats"]))

    ff = mind.full_feedback(data, "cmx")
    g0, T0, _ = mind.fit_logistic(ff)
    lam = np.linalg.svd(T0, compute_uv=False) ** 2
    ratios = lam[:D // 2] / lam[1:D // 2 + 1]
    out("unrestricted fit of cmx (%d labeled pairs, %d clicks, %d impressions): eigenvalue ratios lambda_k / "
        "lambda_k+1, k = 1..%d: %s -> ER rank %d; declared rank %d  (%.0f s)" % (
            ff["n_pairs"], ff["y"].sum(), ff["n_impressions"], D // 2, np.round(ratios, 3), np.argmax(ratios) + 1, RANK,
            time.time() - t0))
    gamma, Theta, _ = mind.fit_logistic(ff, RANK, (g0, T0))
    sv = np.linalg.svd(Theta, compute_uv=False)
    out("truth (rank %d, design units): intercept %.3f, ||Theta||_F %.4f, singular values %s"
        % (RANK, gamma[0], np.linalg.norm(Theta), np.round(sv[:RANK], 4)))

    # the constants the methods are given, from the truth
    starts = np.cumsum((0,) + mind.group_sizes("cmx", D))
    radii_main = np.array([2 * np.linalg.norm(gamma[a:b]) for a, b in zip(starts[:-1], starts[1:])])
    p_star = expit(ff["C"] @ gamma + np.einsum("ij,ij->i", ff["M"] @ Theta, ff["X"]))      # the truth on the fit's rows
    ff_c = dict(ff, C=ff["C"][:, :mind.N_COV], y=p_star)
    g_c, _, _ = mind.fit_logistic(ff_c, 0, (np.zeros(mind.N_COV), None))
    save = dict(D=D, keep=keep, X=data["X"], hmin=HMIN, data_sha=data_sha, package=mind.SETUP_VERSION, rank=RANK,
                er_ratios=ratios, truth_gamma=gamma, truth_Theta=Theta, radius_Theta=2 * np.linalg.norm(Theta),
                radii_main=radii_main, main_only_ball=2 * np.linalg.norm(g_c),
                cov_stats_samecat=np.array(data["cov_stats"]["samecat"]), cov_stats_logK=np.array(data["cov_stats"]["logK"]))
    for v, g in (("cmx", gamma), ("c", gamma[:mind.N_COV])):
        save["glm_S_F_%s" % v], save["glm_S2_%s" % v] = mind.declared(g, Theta, v, D)
        ff_v = dict(ff, C=ff["C"] if v == "cmx" else ff["C"][:, :mind.N_COV], y=p_star)
        save["pm1_S_F_%s" % v], save["pm1_S2_%s" % v] = mind.declared(*mind.linear_fit(ff_v), v, D)
        out("declared norms (S_F, S2) for the %-3s design: logistic (%.3f, %.3f), 2y - 1 (%.3f, %.3f)"
            % (v, save["glm_S_F_%s" % v], save["glm_S2_%s" % v], save["pm1_S_F_%s" % v], save["pm1_S2_%s" % v]))
    # ESTR's declared norms, of its augmented truth (mind.estr_matrix): the logistic truth, and the least-squares
    # fit of 2 p* - 1 in the cmx model (the linear model the authors' 2y - 1 ESTR fits on its augmented design)
    save["glm_S_F_estr"], save["glm_S2_estr"] = mind.declared_estr(gamma, Theta, D)
    save["pm1_S_F_estr"], save["pm1_S2_estr"] = mind.declared_estr(*mind.linear_fit(dict(ff, y=p_star)), D)
    sv_aug = np.linalg.svd(mind.estr_matrix(gamma, Theta, D), compute_uv=False)
    out("declared norms (S_F, S2) of ESTR's augmented truth: logistic (%.3f, %.3f), 2y - 1 (%.3f, %.3f); singular "
        "values of the logistic one %s (ESTR is given rank %d)" % (save["glm_S_F_estr"], save["glm_S2_estr"],
                                                                  save["pm1_S_F_estr"], save["pm1_S2_estr"],
                                                                  np.round(sv_aug[:RANK + 2], 3), RANK + 1))
    out("Algorithm 1's balls (design units): Theta %.4f, main %s; main-only ball %.3f  (%.0f s)"
        % (save["radius_Theta"], np.round(radii_main, 4), save["main_only_ball"], time.time() - t0))

    # the inference: planning value and targets
    imps = np.flatnonzero(data["eligible_K10"])
    pairs = np.concatenate([data["first"][i] + np.arange(data["imp_size"][i]) for i in imps])
    p_D = float(data["c_lab"][pairs].mean())
    names, As, info = targets(data, keep)
    theta = np.r_[gamma, Theta.reshape(-1, order="F")]
    save.update(p_D=p_D, kappa_D=p_D * (1 - p_D), target_names=np.array(names), target_A=As,
                target_info=np.array([str(info)]))
    out("inference: logged click rate of the %d displayed pairs of the %d eligible K = 10 impressions p_D = %.4f, "
        "kappa_D = %.4f" % (len(pairs), len(imps), p_D, p_D * (1 - p_D)))
    for name, A in zip(names, As):
        out("  target %-22s mu* = %+.4f" % (name, A @ theta))
    out("  categories: %s" % info)

    np.savez(out_path, **save)
    out("wrote %s, sha256 %s (%.0f s)" % (out_path, sha(out_path), time.time() - t0))
    open(out_path.replace(".npz", "_check.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
