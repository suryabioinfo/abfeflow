"""Collect works, run estimators per leg / replica / time block, close the cycle, write results.json."""
import glob, json, os, re
import numpy as np
from . import estimators as E
from .stages import state, log

KCAL = 4.184


def _works(cfg, base, direction, tag=""):
    """[(name, W)] for finished transitions; cached in 04_analysis."""
    d = cfg.path(base, f"trans{direction}{tag}")
    files = [p for p in glob.glob(os.path.join(d, "*", "dhdl.xvg")) if os.path.exists(os.path.join(os.path.dirname(p), "done"))]
    cache = cfg.path("04_analysis", f"works_{base.split('_')[1]}_{direction}{tag}.csv")
    if os.path.exists(cache):
        rows = [l.split(",") for l in open(cache).read().splitlines()[1:]]
        if len(rows) == len(files):
            return [(n, float(w)) for n, w in rows]
    out = sorted(((os.path.basename(os.path.dirname(p)), E.integrate(p, direction == "A2B")) for p in files),
                 key=lambda t: _natkey(t[0]))
    with open(cache, "w") as fh:
        fh.write("frame,work_kJmol\n" + "".join(f"{n},{w:.6f}\n" for n, w in out))
    return out


def _natkey(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _times_A(cfg, names):
    meta = state(cfg).get("frames", {}).get("complexA")
    if meta:
        return np.array([meta.get(n, np.nan) for n in names])
    # legacy layout: end-anchored even spacing over the state-A window (matches the original shell scripts)
    t0, t1 = [float(x) * 1000 for x in cfg["sampling"]["stateA_window_ns"]]
    n = int(cfg["sampling"]["n_frames_A"]); dt = (t1 - t0) / n; e = np.floor(t1 / dt) * dt
    return np.array([e - (n - 1 - int(re.findall(r"\d+", nm)[-1])) * dt for nm in names])


def _dG_r(cfg):
    j = cfg.path("01_restraints", "boresch.json")
    if os.path.exists(j):
        return float(json.load(open(j))["dG_r_kJ"])
    itp = open(cfg.path("01_restraints", "boresch_restraints.itp")).read()
    return float(re.search(r"=\s*\+([\d.]+)\s*kJ/mol", itp).group(1))


def run(cfg, tag=""):
    st = state(cfg)
    T = float(st.get("T") or 298.0)
    nb = int(cfg["analysis"]["nboots"])
    os.makedirs(cfg.path("04_analysis"), exist_ok=True)
    out = dict(T=T, units="kJ/mol (kcal/mol fields suffixed _kcal)", tag=tag)
    W = {}
    for leg, base in (("water", "03_water"), ("complex", "02_complex")):
        f, r = _works(cfg, base, "A2B", tag), _works(cfg, base, "B2A", tag)
        if not f or not r:
            raise SystemExit(f"no finished {leg} transitions{tag}")
        W[leg] = (f, r)
        out[leg] = E.leg([w for _, w in f], [w for _, w in r], T, nb)
        log(cfg, f"analysis: {leg}: N={len(f)}/{len(r)}  BAR {out[leg]['bar']/KCAL:.2f} ± {out[leg]['bar_se']/KCAL:.2f}  "
                 f"CGI {out[leg]['cgi']/KCAL:.2f}  overlap {out[leg]['overlap']:.2f} kcal/mol")
    # ---- complex substructure: per state-B replica (all A frames) and per state-A time block (all B frames)
    fA, rB = W["complex"]
    wA = np.array([w for _, w in fA]); tA = _times_A(cfg, [n for n, _ in fA])
    reps = sorted({n.split("_")[0] for n, _ in rB if n.startswith("rep")})
    out["complex_by_B_replica"] = {}
    for rp in reps:
        wr = [w for n, w in rB if n.startswith(rp + "_")]
        out["complex_by_B_replica"][rp] = dict(n=len(wr), bar=E.bar(wA, wr, T), bar_se=E.boot(lambda x, y: E.bar(x, y, T), wA, wr, nb),
                                               wr_mean=float(np.mean(-np.asarray(wr))))
    k = int(cfg["analysis"]["stateA_blocks"]); order = np.argsort(tA); blocks = np.array_split(order, k)
    wrall = np.array([w for _, w in rB])
    out["complex_by_A_block"] = [dict(t_ns=[float(tA[b].min() / 1000), float(tA[b].max() / 1000)], n=len(b),
                                      bar=E.bar(wA[b], wrall, T), wf_mean=float(wA[b].mean()),
                                      wf_se=float(wA[b].std(ddof=1) / np.sqrt(len(b))) if len(b) > 1 else 0.0) for b in blocks]
    # ---- convergence: BAR with growing numbers of transitions (state A in time order, state B round-robin over replicas)
    conv = {}
    for leg in ("water", "complex"):
        f, r = W[leg]
        wf = np.array([w for _, w in f]); wr = np.array([w for _, w in r])
        if leg == "complex":
            wf = wA[order]
            byrep = [[w for n, w in r if n.startswith(rp + "_")] for rp in reps] or [list(wr)]
            wr = np.array([x for tup in zip(*byrep) for x in tup] + [x for lst in byrep for x in lst[min(map(len, byrep)):]])
        fr = np.linspace(0.2, 1.0, 9)
        conv[leg] = []
        for q in fr:
            nf = min(len(wf), max(5, int(round(q * len(wf))))); nr = min(len(wr), max(5, int(round(q * len(wr)))))
            if conv[leg] and conv[leg][-1]["n_f"] == nf and conv[leg][-1]["n_r"] == nr:
                continue                                   # same subset as the previous point
            conv[leg].append(dict(frac=float(q), n_f=nf, n_r=nr, bar=E.bar(wf[:nf], wr[:nr], T),
                                  se=E.boot(lambda x, y: E.bar(x, y, T), wf[:nf], wr[:nr], 100)))
    out["convergence"] = conv
    # ---- cycle
    dGr = _dG_r(cfg)
    rep_bar = [v["bar"] for v in out["complex_by_B_replica"].values()]
    se_rep = float(np.std(rep_bar, ddof=1) / np.sqrt(len(rep_bar))) if len(rep_bar) > 1 else 0.0
    blk = [b["bar"] for b in out["complex_by_A_block"]]
    se_blk = float(np.std(blk, ddof=1) / np.sqrt(len(blk))) if len(blk) > 1 else 0.0
    se_c = max(out["complex"]["bar_se"], se_rep)
    dG = out["water"]["bar"] - out["complex"]["bar"] + dGr
    err = float(np.hypot(out["water"]["bar_se"], se_c))
    err_cons = float(np.sqrt(out["water"]["bar_se"] ** 2 + se_c ** 2 + se_blk ** 2))
    out["cycle"] = dict(dG_water=out["water"]["bar"], dG_complex=out["complex"]["bar"], dG_restr=dGr, dG_bind=dG,
                        err=err, err_conservative=err_cons, se_complex_used=se_c, se_complex_replicas=se_rep,
                        se_complex_Ablocks=se_blk, dG_hydration=-out["water"]["bar"])
    out["cycle_kcal"] = {k: v / KCAL for k, v in out["cycle"].items()}
    ref = cfg.get("reference", {})
    out["reference"] = ref
    json.dump(out, open(cfg.path("04_analysis", f"results{tag}.json"), "w"), indent=2, default=float)
    c = out["cycle_kcal"]
    msg = (f"analysis: dG_bind = {c['dG_bind']:.2f} ± {c['err']:.2f} kcal/mol (conservative ± {c['err_conservative']:.2f}); "
           f"dG_w {c['dG_water']:.2f}, dG_c {c['dG_complex']:.2f}, dG_r +{c['dG_restr']:.2f}")
    if ref.get("dG_exp") is not None:
        msg += f"; experiment {ref['dG_exp']:.2f}"
    log(cfg, msg)
    _warnings(cfg, out)
    return out


def _warnings(cfg, out):
    W = []
    for leg in ("water", "complex"):
        L = out[leg]
        if L["overlap"] < 0.1:
            W.append(f"{leg}: forward/reverse work overlap {L['overlap']:.2f} < 0.1 - longer or more transitions needed")
        if abs(L["bar"] - L["cgi"]) / KCAL > 0.5:
            W.append(f"{leg}: BAR and CGI differ by {abs(L['bar']-L['cgi'])/KCAL:.2f} kcal/mol (non-Gaussian works)")
    # spreads are flagged only when they exceed 1 kcal/mol AND are statistically significant (> 2 combined SE)
    rv = list(out["complex_by_B_replica"].values())
    if len(rv) > 1:
        hi, lo = max(rv, key=lambda v: v["bar"]), min(rv, key=lambda v: v["bar"])
        d, se = hi["bar"] - lo["bar"], float(np.hypot(hi["bar_se"], lo["bar_se"]))
        if d / KCAL > 1.0 and d > 2 * se:
            W.append(f"complex: state-B replicas span {d/KCAL:.2f} kcal/mol (> 2 SE = {2*se/KCAL:.2f}) - state B is not converged")
    ab = out["complex_by_A_block"]
    if len(ab) > 1:
        hi, lo = max(ab, key=lambda b: b["wf_mean"]), min(ab, key=lambda b: b["wf_mean"])
        d, se = hi["wf_mean"] - lo["wf_mean"], float(np.hypot(hi.get("wf_se", 0), lo.get("wf_se", 0)))
        if d / KCAL > 1.5 and d > 2 * se:
            W.append(f"complex: mean forward work differs by {d/KCAL:.2f} kcal/mol (> 2 SE = {2*se/KCAL:.2f}) between state-A "
                     "time blocks - the bound state samples slowly interconverting substates")
    href = (cfg.get("reference") or {}).get("dG_hyd_ref")
    if href is not None and abs(-out["water"]["bar"] / KCAL - href) > 1.5:
        W.append(f"water: hydration free energy {-out['water']['bar']/KCAL:.2f} kcal/mol differs from the reference "
                 f"{href:.2f} by more than 1.5 kcal/mol - check ligand charges / protocol on a known molecule")
    out["warnings"] = W
    json.dump(out, open(cfg.path("04_analysis", f"results{out['tag']}.json"), "w"), indent=2, default=float)
    for w in W:
        log(cfg, "WARNING: " + w)
