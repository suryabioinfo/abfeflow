"""Non-equilibrium free-energy estimators (pmx-compatible conventions).

Forward work  W_f = int_0^1 dH/dl dl     along A->B
Reverse work  W_r = int_1^0 dH/dl dl     along B->A   (so -W_r is comparable with W_f)
All energies in kJ/mol unless stated.
"""
import numpy as np
from scipy import stats

KB = 0.0083144626


def integrate(dhdl_file, forward=True):
    d = np.loadtxt(dhdl_file, comments=("#", "@"), usecols=(1,))
    lam = np.linspace(0.0, 1.0, len(d)) if forward else np.linspace(1.0, 0.0, len(d))
    return float(np.trapezoid(d, lam)) if hasattr(np, "trapezoid") else float(np.trapz(d, lam))


def bar(wf, wr, T):
    """Bennett acceptance ratio for non-equilibrium work (Shirts et al. 2003). Returns dG (kJ/mol)."""
    wf, wr = np.asarray(wf, float), np.asarray(wr, float)
    kT = KB * T; M = kT * np.log(len(wf) / len(wr))
    f = lambda x: 1.0 / (1.0 + np.exp(np.clip(x / kT, -500, 500)))
    g = lambda dg: f(M + wf - dg).sum() - f(-M + wr + dg).sum()
    lo, hi = min(wf.min(), -wr.max()) - 100, max(wf.max(), -wr.min()) + 100
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g(mid) > 0: hi = mid
        else: lo = mid
    return 0.5 * (lo + hi)


def cgi(wf, wr):
    """Crooks Gaussian intersection."""
    m1, s1 = np.mean(wf), np.std(wf, ddof=1)
    m2, s2 = np.mean(-np.asarray(wr)), np.std(wr, ddof=1)
    if not (s1 > 1e-9 and s2 > 1e-9):
        return float("nan")
    if abs(s1 - s2) < 1e-9:
        return 0.5 * (m1 + m2)
    a = 1 / (2 * s1 ** 2) - 1 / (2 * s2 ** 2)
    b = m2 / s2 ** 2 - m1 / s1 ** 2
    c = m1 ** 2 / (2 * s1 ** 2) - m2 ** 2 / (2 * s2 ** 2) - np.log(s2 / s1)
    r = np.roots([a, b, c]).real
    lo, hi = sorted((m1, m2))
    inside = [x for x in r if lo - 1e-9 <= x <= hi + 1e-9]
    return float(inside[0] if inside else r[np.argmin(np.abs(r - 0.5 * (m1 + m2)))])


def jarzynski(w, T, forward=True):
    """forward: dG = -kT ln<exp(-W_f/kT)>;  reverse: dG(A->B) = +kT ln<exp(-W_r/kT)>."""
    kT = KB * T; x = -np.asarray(w, float) / kT
    mx = x.max(); lme = mx + np.log(np.mean(np.exp(x - mx)))
    return float(-kT * lme if forward else kT * lme)


def ks_p(w):
    w = np.asarray(w, float)
    return float(stats.kstest((w - w.mean()) / w.std(ddof=1), "norm").pvalue) if len(w) > 3 else float("nan")


def overlap(wf, wr, bins=40):
    """Histogram overlap coefficient (0..1) between W_f and -W_r."""
    a, b = np.asarray(wf), -np.asarray(wr)
    edges = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), bins + 1)
    ha, _ = np.histogram(a, edges, density=True); hb, _ = np.histogram(b, edges, density=True)
    return float(np.sum(np.minimum(ha, hb)) * (edges[1] - edges[0]))


def boot(fn, wf, wr, n=200, seed=1):
    rng = np.random.default_rng(seed); wf, wr = np.asarray(wf), np.asarray(wr)
    v = np.array([fn(rng.choice(wf, len(wf)), rng.choice(wr, len(wr))) for _ in range(n)], float)
    v = v[np.isfinite(v)]
    return float(np.std(v, ddof=1)) if len(v) > 2 else float("nan")


def leg(wf, wr, T, nboots=200):
    """All estimators for one leg (kJ/mol)."""
    b = bar(wf, wr, T)
    return dict(n_f=len(wf), n_r=len(wr), bar=b, bar_se=boot(lambda x, y: bar(x, y, T), wf, wr, nboots),
                cgi=cgi(wf, wr), cgi_se=boot(cgi, wf, wr, nboots),
                jarz_f=jarzynski(wf, T, True), jarz_r=jarzynski(wr, T, False),
                wf_mean=float(np.mean(wf)), wf_sd=float(np.std(wf, ddof=1)),
                wr_mean=float(np.mean(-np.asarray(wr))), wr_sd=float(np.std(wr, ddof=1)),
                ks_p_f=ks_p(wf), ks_p_r=ks_p(wr), overlap=overlap(wf, wr),
                dissipation_f=float(np.mean(wf) - b), dissipation_r=float(np.mean(wr) + b))
