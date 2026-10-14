"""Publication figures, tables (CSV / Markdown / LaTeX) and a methods paragraph from results.json."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KCAL = 4.184
# reference palette (validated categorical order) + ink / grid tokens
BLUE, ORANGE, AQUA = "#0e7afd", "#f75819", "#00b374"
SERIES = [BLUE, ORANGE, AQUA, "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e7e6e2", "#5f5e5a"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})


def _save(fig, stem):
    try: fig.tight_layout()
    except Exception: pass
    fig.savefig(stem + ".png"); fig.savefig(stem + ".pdf"); plt.close(fig)


def _gauss(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))


# ---------------------------------------------------------------------------------------------- figures
def fig_cycle(res, ref, stem):
    c = res["cycle_kcal"]
    # each bar is the signed CONTRIBUTION to dG_bind; tick labels give the leg's own value
    steps = [(f"water leg\nΔG$_{{water}}$ = {c['dG_water']:.2f}", c["dG_water"]),
             (f"complex leg\nΔG$_{{complex}}$ = {c['dG_complex']:.2f}\n(bar: −ΔG$_{{complex}}$)", -c["dG_complex"]),
             (f"restraints\nΔG$_{{restr}}$ = {c['dG_restr']:.2f}", c["dG_restr"])]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    run, levels = 0.0, [0.0]
    for i, (lab, v) in enumerate(steps):
        ax.bar(i, v, bottom=run, width=0.55, color=BLUE if v < 0 else ORANGE, edgecolor="white", linewidth=1.5)
        inside = abs(v) > 0.18 * (abs(c["dG_complex"]) + 1)
        ax.text(i if inside else i + 0.31, run + v / 2, f"{abs(v):.2f}", va="center", ha="center" if inside else "left",
                fontsize=8.5, color="white" if inside else INK, fontweight="bold" if inside else "normal")
        ax.plot([i + 0.28, i + 0.72], [run + v] * 2, color=INK2, lw=0.8, ls=":")
        run += v; levels.append(run)
    i = len(steps)
    ax.bar(i, c["dG_bind"], width=0.55, color=NEUTRAL, edgecolor="white", linewidth=1.5)
    ax.errorbar(i, c["dG_bind"], yerr=c["err"], color=INK, capsize=4, lw=1.2)
    ax.text(i + 0.31, c["dG_bind"] / 2, f"{c['dG_bind']:.2f}\n± {c['err']:.2f}", va="center", ha="left", fontsize=8.5, color=INK)
    lo, hi = min(levels + [c["dG_bind"] - c["err"], ref.get("dG_exp") or 0]), max(levels)
    pad = 0.08 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad); ax.set_xlim(-0.5, len(steps) + 0.9)
    if ref.get("dG_exp") is not None:
        ax.axhline(ref["dG_exp"], color=INK, lw=1.0, ls="--")
        ax.text(len(steps) + 0.88, ref["dG_exp"] - pad * 0.15, f"{ref.get('label_exp','Experiment')}: {ref['dG_exp']:.2f}",
                va="top", ha="right", fontsize=8, color=INK2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(len(steps) + 1), [l for l, _ in steps] + [f"result\nΔG°$_{{bind}}$ = {c['dG_bind']:.2f}"])
    ax.tick_params(axis="x", labelsize=8)
    ax.set_ylabel("Free energy (kcal/mol)")
    ax.set_title("ΔG°$_{bind}$ = ΔG$_{water}$ − ΔG$_{complex}$ + ΔG$_{restr}$ = "
                 f"{c['dG_water']:.2f} − {c['dG_complex']:.2f} + {c['dG_restr']:.2f} = {c['dG_bind']:.2f} kcal/mol",
                 loc="left", pad=10, fontsize=9.5)
    _save(fig, stem)


def fig_works(works, res, T, stem):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for ax, leg, title in zip(axs, ("water", "complex"), ("Water leg", "Complex leg")):
        wf = np.array(works[leg][0]) / KCAL; wr = -np.array(works[leg][1]) / KCAL
        lo, hi = min(wf.min(), wr.min()), max(wf.max(), wr.max()); pad = 0.1 * (hi - lo)
        bins = np.linspace(lo - pad, hi + pad, 30); x = np.linspace(lo - pad, hi + pad, 300)
        for w, col, lab in ((wf, BLUE, "forward  $W_{A→B}$"), (wr, ORANGE, "reverse  $−W_{B→A}$")):
            ax.hist(w, bins=bins, density=True, color=col, alpha=0.35, edgecolor="white", linewidth=0.8)
            ax.hist(w, bins=bins, density=True, histtype="step", color=col, linewidth=1.2, label=lab)
            ax.plot(x, _gauss(x, w.mean(), w.std(ddof=1)), color=col, ls="--", lw=1.0)
        L = res[leg]
        ax.axvline(L["bar"] / KCAL, color=INK, lw=1.2)
        ax.text(0.02, 0.97, f"BAR ΔG = {L['bar']/KCAL:.2f} ± {L['bar_se']/KCAL:.2f}\nN = {L['n_f']} / {L['n_r']},  overlap {L['overlap']:.2f}",
                transform=ax.transAxes, va="top", fontsize=8, color=INK)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Work (kcal/mol)"); ax.set_ylabel("Density")
    axs[0].legend(loc="upper left", bbox_to_anchor=(0, -0.22), ncol=2)
    _save(fig, stem)


def fig_convergence(res, stem):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, leg, title in zip(axs, ("water", "complex"), ("Water leg", "Complex leg")):
        cv = res["convergence"][leg]
        n = np.array([c["n_f"] + c["n_r"] for c in cv]); y = np.array([c["bar"] for c in cv]) / KCAL
        e = np.array([c["se"] for c in cv]) / KCAL
        ax.fill_between(n, y - e, y + e, color=BLUE, alpha=0.18, linewidth=0)
        ax.plot(n, y, color=BLUE, marker="o", ms=4)
        ax.axhline(y[-1], color=INK2, lw=0.8, ls="--")
        ax.set_title(title, loc="left"); ax.set_xlabel("Transitions used (forward + reverse)"); ax.set_ylabel("ΔG BAR (kcal/mol)")
    _save(fig, stem)


def fig_complex_diag(works, res, tA, stem):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw=dict(width_ratios=[1.6, 1]))
    wf = np.array(works["complex"][0]) / KCAL
    o = np.argsort(tA); t = np.array(tA)[o] / 1000; w = wf[o]
    axs[0].scatter(t, w, s=12, color=BLUE, edgecolor="white", linewidth=0.5, zorder=3)
    if len(w) >= 7:
        k = 7; rm = np.array([np.median(w[max(0, i - k // 2):i + k // 2 + 1]) for i in range(len(w))])
        axs[0].plot(t, rm, color=INK2, lw=1.2, label="running median (7)")
        axs[0].legend(loc="lower left")
    axs[0].set_xlabel("State-A frame time (ns)"); axs[0].set_ylabel("Forward work (kcal/mol)")
    axs[0].set_title("State A: forward work vs. frame time", loc="left")
    reps = res["complex_by_B_replica"]; names = list(reps)
    rB = works["complex_r_by_rep"]
    rng = np.random.default_rng(0)
    for i, rp in enumerate(names):
        y = -np.array(rB[rp]) / KCAL
        axs[1].scatter(i + rng.uniform(-0.18, 0.18, len(y)), y, s=12, color=SERIES[i % 8], edgecolor="white", linewidth=0.5, zorder=3)
        axs[1].plot([i - 0.28, i + 0.28], [y.mean()] * 2, color=INK, lw=1.4)
    axs[1].axhline(res["complex"]["bar"] / KCAL, color=INK2, lw=0.8, ls="--")
    axs[1].set_xticks(range(len(names)), [f"{n}\n{reps[n]['bar']/KCAL:.2f}" for n in names])
    axs[1].set_ylabel("−Reverse work (kcal/mol)"); axs[1].set_title("State B replicas (BAR below)", loc="left")
    _save(fig, stem)


def fig_comparison(res, ref, stem):
    c = res["cycle_kcal"]
    rows = [("ABFE", c["dG_bind"], c["err"], BLUE)]
    for k, v in (ref.get("other") or {}).items():
        rows.append((k, float(v), None, INK2))
    if ref.get("dG_exp") is not None:
        rows.append((ref.get("label_exp", "Experiment"), ref["dG_exp"], ref.get("dG_exp_err"), INK))
    fig, ax = plt.subplots(figsize=(4.6, 0.55 * len(rows) + 0.9))
    for i, (lab, v, e, col) in enumerate(rows[::-1]):
        ax.errorbar(v, i, xerr=e, fmt="o", ms=7, color=col, capsize=4, lw=1.4)
        ax.text(v, i + 0.22, f"{v:.2f}" + (f" ± {e:.2f}" if e else ""), ha="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
    ax.set_ylim(-0.6, len(rows) - 0.3); ax.set_xlabel("ΔG°$_{bind}$ (kcal/mol)")
    if c.get("err_conservative"):
        i = len(rows) - 1
        ax.plot([c["dG_bind"] - c["err_conservative"], c["dG_bind"] + c["err_conservative"]], [i, i], color=BLUE, lw=0.8, alpha=0.6)
    ax.set_title("Binding free energy", loc="left")
    _save(fig, stem)


def restraint_figure(dat, info, stem):
    d = np.loadtxt(dat)
    labs = ["r (nm)", "θ$_A$ (°)", "θ$_B$ (°)", "φ$_A$ (°)", "φ$_B$ (°)", "φ$_C$ (°)"]
    refs = [info["r0_nm"], info["thA_deg"], info["thB_deg"], info["phiA_deg"], info["phiB_deg"], info["phiC_deg"]]
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 3.6), sharex=True)
    for k, ax in enumerate(axs.flat):
        y = d[:, k + 1]
        if k >= 3:                                   # dihedrals: unwrap around the reference (no false ±180° jumps)
            y = refs[k] + (y - refs[k] + 180.0) % 360.0 - 180.0
        ax.plot(d[:, 0], y, color=BLUE, lw=0.9)
        ax.axhline(refs[k], color=INK, lw=0.9, ls="--")
        ax.set_title(labs[k], loc="left")
    for ax in axs[1]:
        ax.set_xlabel("time (ns)")
    fig.suptitle("Boresch coordinates in state A: P " + "/".join(info["P_label"]) + "   L " + "/".join(info["L_label"]),
                 fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, stem)


# ---------------------------------------------------------------------------------------------- tables
def _tables(res, st, cfg, outdir):
    c = res["cycle_kcal"]; ref = cfg.get("reference", {})
    cyc = [("ΔG_water (decoupling in water)", c["dG_water"], res["water"]["bar_se"] / KCAL),
           ("ΔG_complex (decoupling + restraints)", c["dG_complex"], c["se_complex_used"]),
           ("ΔG_restr (analytic, standard state)", c["dG_restr"], 0.0),
           ("ΔG°_bind = ΔG_water − ΔG_complex + ΔG_restr", c["dG_bind"], c["err"]),
           ("ΔG°_bind, conservative error (incl. state-A blocks)", c["dG_bind"], c["err_conservative"])]
    if ref.get("dG_exp") is not None:
        cyc.append((f"{ref.get('label_exp','Experiment')}", ref["dG_exp"], ref.get("dG_exp_err") or 0.0))
    legs = []
    for leg in ("water", "complex"):
        L = res[leg]
        legs.append((leg, L["n_f"], L["n_r"], L["bar"] / KCAL, L["bar_se"] / KCAL, L["cgi"] / KCAL, L["cgi_se"] / KCAL,
                     L["jarz_f"] / KCAL, L["jarz_r"] / KCAL, L["wf_mean"] / KCAL, L["wf_sd"] / KCAL, L["wr_mean"] / KCAL,
                     L["wr_sd"] / KCAL, L["overlap"], L["ks_p_f"], L["ks_p_r"]))
    reps = [(k, v["n"], v["bar"] / KCAL, v["bar_se"] / KCAL) for k, v in res["complex_by_B_replica"].items()]
    blks = [(f"{b['t_ns'][0]:.0f}-{b['t_ns'][1]:.0f} ns", b["n"], b["bar"] / KCAL, b["wf_mean"] / KCAL) for b in res["complex_by_A_block"]]
    H = dict(cycle=("Term", "kcal/mol", "±"),
             legs=("Leg", "N_f", "N_r", "BAR", "±", "CGI", "±", "JARZ_f", "JARZ_r", "<W_f>", "sd", "<-W_r>", "sd", "overlap", "KS p_f", "KS p_r"),
             replicas=("State-B replica", "N_r", "ΔG_complex BAR", "±"),
             blocks=("State-A block", "N_f", "ΔG_complex BAR", "<W_f>"))
    data = dict(cycle=cyc, legs=legs, replicas=reps, blocks=blks)
    fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)
    md = []
    for name, rows in data.items():
        h = H[name]
        with open(os.path.join(outdir, f"table_{name}.csv"), "w") as fh:
            fh.write(",".join(h) + "\n" + "".join(",".join(fmt(x) for x in r) + "\n" for r in rows))
        t = "| " + " | ".join(h) + " |\n|" + "---|" * len(h) + "\n" + "".join("| " + " | ".join(fmt(x) for x in r) + " |\n" for r in rows)
        md.append((name, t))
        tex = ("\\begin{tabular}{l" + "r" * (len(h) - 1) + "}\n\\toprule\n" + " & ".join(h).replace("_", "\\_").replace("<", "$<$").replace(">", "$>$")
               .replace("Δ", "$\\Delta$").replace("°", "$^\\circ$").replace("±", "$\\pm$").replace("−", "$-$") + " \\\\\n\\midrule\n"
               + "".join(" & ".join(fmt(x) for x in r).replace("_", "\\_").replace("Δ", "$\\Delta$").replace("°", "$^\\circ$")
                         .replace("−", "$-$") + " \\\\\n" for r in rows) + "\\bottomrule\n\\end{tabular}\n")
        open(os.path.join(outdir, f"table_{name}.tex"), "w").write(tex)
    return dict(md)


def _methods(res, st, cfg):
    S = cfg["sampling"]; r = st["restraint"]; c = res["cycle_kcal"]
    sc = {"gapsys": "the soft-core potential of Gapsys et al. (linearisation points 0.85 for Lennard-Jones, 0.3 for Coulomb)",
          "beutler": "Beutler soft-core (α = 0.5, power 1, σ = 0.3 nm) applied to both Lennard-Jones and Coulomb terms"}[cfg["softcore"]]
    return (
        f"Absolute binding free energies were computed with a non-equilibrium (fast-growth) alchemical protocol in GROMACS"
        f"{'' if st.get('gmx_version', 'unknown') == 'unknown' else ' ' + st['gmx_version']}. Non-bonded, constraint and barostat settings were identical to the production simulation. "
        f"The ligand was decoupled from its environment (couple-intramol = no) along a single λ coordinate on which Coulomb and "
        f"Lennard-Jones interactions vanish simultaneously, using {sc}. In the complex, six Boresch restraints "
        f"(one distance, two angles, three dihedrals; k = {r['k_dist']:g} kJ mol⁻¹ nm⁻² and {r['k_ang']:g} kJ mol⁻¹ rad⁻²) were switched on "
        f"along the same λ. Anchor atoms were selected automatically from the unrestrained bound-state trajectory "
        f"({S['stateA_window_ns'][0]}–{S['stateA_window_ns'][1]} ns) as the rigid protein backbone atoms "
        f"({', '.join(r['P_label'])}) and ligand atoms ({', '.join(r['L_label'])}) minimising the mean restraint energy of the "
        f"unrestrained ensemble; the analytic standard-state correction was ΔG_restr = +{c['dG_restr']:.2f} kcal/mol (T = {res['T']:g} K). "
        f"Equilibrium sampling of the decoupled, restrained state comprised {S['stateB_replicas']} independent replicas of "
        f"{S['stateB_length_ns']} ns; the water leg used {S['water_eq_ns']} ns at each end state. "
        f"{res['complex']['n_f']} forward and {res['complex']['n_r']} reverse transitions of {S['transition_ps']['complex']} ps were run "
        f"in the complex and {res['water']['n_f']}/{res['water']['n_r']} transitions of {S['transition_ps']['water']} ps in water, each "
        f"started from an equilibrium snapshot with velocities drawn from the Maxwell–Boltzmann distribution. Work values were "
        f"obtained by integrating dH/dλ, and free energies were estimated with the Bennett acceptance ratio (BAR); Crooks Gaussian "
        f"intersection and Jarzynski estimators are reported for comparison. Uncertainties are bootstrap standard errors "
        f"({cfg['analysis']['nboots']} resamples); for the complex leg the larger of the bootstrap error and the standard error over "
        f"state-B replicas was used. ΔG°bind = ΔG_water − ΔG_complex + ΔG_restr = {c['dG_bind']:.2f} ± {c['err']:.2f} kcal/mol.")


def run(cfg, tag=""):
    from .stages import state
    from .analysis import _works, _times_A
    st = state(cfg)
    res = json.load(open(cfg.path("04_analysis", f"results{tag}.json")))
    out = cfg.path("05_report"); os.makedirs(out, exist_ok=True)
    ref = cfg.get("reference", {})
    works = {}
    for leg, base in (("water", "03_water"), ("complex", "02_complex")):
        f, r = _works(cfg, base, "A2B", tag), _works(cfg, base, "B2A", tag)
        works[leg] = ([w for _, w in f], [w for _, w in r])
        if leg == "complex":
            works["complex_r_by_rep"] = {rp: [w for n, w in r if n.startswith(rp + "_")] for rp in res["complex_by_B_replica"]}
            tA = _times_A(cfg, [n for n, _ in f])
    fig_cycle(res, ref, os.path.join(out, "fig1_cycle"))
    fig_works(works, res, res["T"], os.path.join(out, "fig2_work_distributions"))
    fig_convergence(res, os.path.join(out, "fig3_convergence"))
    fig_complex_diag(works, res, tA, os.path.join(out, "fig4_complex_diagnostics"))
    fig_comparison(res, ref, os.path.join(out, "fig5_comparison"))
    p = cfg.path("01_restraints", "boresch_timeseries.dat")
    if os.path.exists(p) and "restraint" in st:
        restraint_figure(p, st["restraint"], os.path.join(out, "figS1_restraints"))
    md = _tables(res, st, cfg, out)
    methods = _methods(res, st, cfg) if "restraint" in st else "(methods text needs `abfeflow setup` state)"
    open(os.path.join(out, "methods.md"), "w").write(methods + "\n")
    c = res["cycle_kcal"]
    rep = [f"# ABFE report: {st.get('ligand', 'ligand')}\n",
           f"**ΔG°bind = {c['dG_bind']:.2f} ± {c['err']:.2f} kcal/mol** (conservative ± {c['err_conservative']:.2f})"
           + (f"; {ref.get('label_exp','experiment')} {ref['dG_exp']:.2f}" if ref.get("dG_exp") is not None else "") + "\n",
           "## Warnings\n" + ("\n".join(f"- {w}" for w in res.get("warnings", [])) or "- none") + "\n",
           "## Figures\n",
           "![](fig1_cycle.png)\n\n*Figure 1. Thermodynamic-cycle decomposition. Bars are the contributions to ΔG°bind; the grey bar is the "
           "result with its standard error; the dashed line is the experimental value.*\n",
           "![](fig2_work_distributions.png)\n\n*Figure 2. Forward (A→B, decoupling) and sign-inverted reverse (B→A) work distributions with "
           "Gaussian fits (dashed). By the Crooks theorem the distributions cross at ΔG; the solid line is the BAR estimate.*\n",
           "![](fig3_convergence.png)\n\n*Figure 3. BAR estimate as a function of the number of transitions used (shaded: bootstrap SE).*\n",
           "![](fig4_complex_diagnostics.png)\n\n*Figure 4. Complex leg diagnostics: forward work against the time of the state-A "
           "snapshot (detects slowly interconverting bound substates) and reverse work by state-B replica.*\n",
           "![](fig5_comparison.png)\n\n*Figure 5. Computed binding free energy compared with experiment and other estimates "
           "(thin line: conservative error).*\n",
           "## Tables\n", "### Thermodynamic cycle (kcal/mol)\n" + md["cycle"],
           "### Estimators per leg (kcal/mol)\n" + md["legs"],
           "### Complex leg by state-B replica\n" + md["replicas"], "### Complex leg by state-A time block\n" + md["blocks"],
           "## Methods (draft)\n" + methods + "\n"]
    open(os.path.join(out, "report.md"), "w").write("\n".join(rep))
    print(f"report: written to {out}")
