"""Workflow stages. Each stage is restart-safe and ends with a guard that stops on a broken state."""
import glob, json, os, shutil, threading, time
import numpy as np
from .gmx import Gmx, GmxError
from .topology import Topology
from .mdp import MdpWriter
from . import restraints as R

DIRS = ["00_inputs", "01_restraints", "02_complex/topology", "02_complex/eqB", "03_water/topology", "03_water/equil",
        "04_analysis", "05_report", "mdp"]


def log(cfg, msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(msg, flush=True)
    with open(cfg.path("abfeflow.log"), "a") as fh:
        fh.write(line + "\n")


def state(cfg):
    p = cfg.path("abfe_state.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def save_state(cfg, st):
    json.dump(st, open(cfg.path("abfe_state.json"), "w"), indent=2, default=float)


def _need(st, key, stage):
    if key not in st:
        raise SystemExit(f"run `abfeflow {stage}` first")


# --------------------------------------------------------------------------------------------- setup
def setup(cfg):
    for d in DIRS:
        os.makedirs(cfg.path(d), exist_ok=True)
    g = Gmx(cfg); sysc = cfg["system"]; L = lambda m: log(cfg, m)
    for k in ("production_tpr", "production_xtc", "production_mdp", "topology"):
        if not os.path.exists(sysc[k]):
            raise SystemExit(f"missing {k}: {sysc[k]}")
    shutil.copy(sysc["production_mdp"], cfg.path("00_inputs", "production.mdp"))
    ref = cfg.path("00_inputs", "ref.gro")
    if sysc.get("reference_gro"):
        shutil.copy(sysc["reference_gro"], ref)
    elif not os.path.exists(ref):
        g.run(["editconf", "-f", sysc["production_tpr"], "-o", ref], cfg.path("00_inputs"), "editconf.log")
    top = Topology(sysc["topology"], sysc["ligand_moltype"])
    q = top.lig_charge
    if abs(q) > 1e-3:
        raise SystemExit(f"ligand net charge {q:+.3f}: charged ligands need a finite-size / co-alchemical-ion "
                         "correction that this single-lambda protocol does not apply")
    water = sysc["water_moltype"]
    pos, neg = top.ions(water)
    nw = top.count(water)
    conc = (min(sum(top.count(n) for n in pos), sum(top.count(n) for n in neg)) / (nw / 55.5)) if nw and pos and neg else 0.0
    # water sites from the reference structure
    sites = 3
    for line in open(ref).read().splitlines()[2:-1]:
        if line[5:10].strip() == water:
            rid = line[:5]; sites = sum(1 for l in open(ref).read().splitlines()[2:-1] if l[:5] == rid and l[5:10].strip() == water); break
    mw = MdpWriter(cfg, sysc["production_mdp"])
    T = mw.T
    L(f"setup: ligand {top.ligand} ({len(top.lig_atoms)} atoms, q={q:+.3f}), T={T:g} K, water {water} x{nw} ({sites}-site), "
      f"ions +{pos} -{neg} -> {conc:.3f} M")
    heavy = [i for i, a in enumerate(top.lig_atoms) if top.heavy(a)]
    rdir = cfg.path("01_restraints")
    if cfg["restraints"]["mode"] == "manual":
        info = R.manual(cfg, ref, sysc["production_xtc"], top.lig_atoms, T, rdir, log=L)
    else:
        info = R.select(cfg, ref, sysc["production_xtc"], top.lig_atoms, top.lig_bonds, heavy, top.lig_resname, T, rdir, log=L)
    files = mw.write_all(cfg.path("mdp"))
    top.write_complex(cfg.path("02_complex", "topology", "topol_abfe.top"), os.path.join(rdir, "boresch_restraints.itp"))
    top.write_water(cfg.path("03_water", "topology", "topol_water.top"))
    st = state(cfg)
    st.pop("checked", None)                             # setup changes inputs -> check must be rerun
    st.update(dict(T=T, ligand=top.ligand, lig_resname=top.lig_resname, lig_natoms=len(top.lig_atoms),
                   lig_first=info["L"][0] - 0, water=water, water_sites=sites, ion_pos=pos, ion_neg=neg, salt_M=conc,
                   restraint=info, mdps=files, gmx_version=g.version(),
                   lig_global=[i for i in range(_lig_offset(ref, top.lig_resname) + 1,
                                                _lig_offset(ref, top.lig_resname) + 1 + len(top.lig_atoms))]))
    save_state(cfg, st)
    plot_restraints(cfg)
    L("setup: done -> next `abfeflow check`")


def _lig_offset(ref, resname):
    for i, line in enumerate(open(ref).read().splitlines()[2:-1]):
        if line[5:10].strip() == resname:
            return i
    raise SystemExit(f"ligand residue {resname} not in {ref}")


def plot_restraints(cfg):
    from .report import restraint_figure
    p = cfg.path("01_restraints", "boresch_timeseries.dat")
    if os.path.exists(p):
        restraint_figure(p, json.load(open(cfg.path("01_restraints", "boresch.json"))), cfg.path("01_restraints", "boresch_timeseries"))


# --------------------------------------------------------------------------------------------- log parsing
def log_energies(path):
    """First energy block of a GROMACS .log -> {term: value}."""
    lines = open(path).read().splitlines()
    for i, l in enumerate(lines):
        if l.strip() == "Energies (kJ/mol)":
            out, j = {}, i + 1
            while j + 1 < len(lines) and lines[j].strip():
                names, vals = lines[j], lines[j + 1]
                for k in range(0, len(names), 15):
                    n, v = names[k:k + 15].strip(), vals[k:k + 15].strip()
                    if n:
                        try: out[n] = float(v)
                        except ValueError: pass
                j += 2
            return out
    return {}


def _gro_natoms(path):
    try:
        with open(path) as fh:
            fh.readline(); n = int(fh.readline().split()[0])
        return n
    except Exception:
        return 0


def _dump(cfg, t_ps, out, cwd, natoms=None):
    """Extract one frame; verified (atom count) and retried once - never continue from an empty structure."""
    g = Gmx(cfg); dst = os.path.join(cwd, out)
    for attempt in (1, 2):
        if os.path.exists(dst):
            os.remove(dst)
        g.run(["trjconv", "-s", cfg["system"]["production_tpr"], "-f", cfg["system"]["production_xtc"], "-dump", t_ps,
               "-pbc", "mol", "-center", "-o", out], cwd, f"trjconv_dump{attempt}.log",
              stdin=f"{cfg['system']['center_group']}\nSystem\n")
        n = _gro_natoms(dst)
        if n and (natoms is None or n == natoms):
            return n
    raise SystemExit(f"frame extraction at t={t_ps} ps produced an invalid {dst} ({n} atoms); see {cwd}/trjconv_dump2.log")


# --------------------------------------------------------------------------------------------- check
def check(cfg):
    st = state(cfg); _need(st, "restraint", "setup")
    g = Gmx(cfg); d = cfg.path("02_complex", "check"); os.makedirs(d, exist_ok=True)
    top = cfg.path("02_complex", "topology", "topol_abfe.top")
    t0 = float(cfg["sampling"]["stateA_window_ns"][0]) * 1000
    _dump(cfg, t0, "frame.gro", d)
    base = open(cfg.path("mdp", "complex_check.mdp")).read()
    E = {}
    for lam in (0, 1):
        m = os.path.join(d, f"check_l{lam}.mdp")
        open(m, "w").write(base.replace("init-lambda             = 0", f"init-lambda             = {lam}"))
        g.grompp(m, "frame.gro", top, f"l{lam}.tpr", d, log=f"grompp_l{lam}.log")
        g.run(["mdrun", "-deffnm", f"l{lam}", "-nsteps", 0] + g.mdrun_opts(4), d, f"mdrun_l{lam}.out")
        E[lam] = log_energies(os.path.join(d, f"l{lam}.log"))
    h0, h1 = E[0].get("Harmonic Pot.", 0.0), E[1].get("Harmonic Pot.", 0.0)
    i0, i1 = E[0].get("Improper Dih.", 0.0), E[1].get("Improper Dih.", 0.0)
    log(cfg, f"check: restraint terms  lambda=0: harmonic {h0:.3f}, improper {i0:.3f}   lambda=1: harmonic {h1:.3f}, improper {i1:.3f} kJ/mol")
    ok = abs(h0) < 1e-6 and (h1 > 0 or i1 > i0)
    if not ok:
        raise SystemExit("check FAILED: restraint energy must be 0 at lambda=0 and > 0 at lambda=1 "
                         "(is boresch_restraints.itp included at the end of topol_abfe.top?)")
    st["checked"] = True; save_state(cfg, st)
    log(cfg, "check: passed (grompp -maxwarn %d at both end states, restraints switch with lambda)" % g.maxwarn)


# --------------------------------------------------------------------------------------------- water leg
def water(cfg, pinoffset=None, threads=None):
    st = state(cfg); _need(st, "restraint", "setup")
    g = Gmx(cfg); M = cfg.path("mdp"); T = cfg.path("03_water", "topology"); E = cfg.path("03_water", "equil")
    threads = threads or cfg["parallel"]["water_eq_threads"]
    if pinoffset is None:
        pinoffset = max(0, int(cfg["hardware"]["cores"]) - threads)
    top = os.path.join(T, "topol.top")
    if not os.path.exists(os.path.join(T, "sys.gro")):
        shutil.copy(os.path.join(T, "topol_water.top"), top)
        ref = open(cfg.path("00_inputs", "ref.gro")).read().splitlines()
        lig = [l for l in ref[2:-1] if l[5:10].strip() == st["lig_resname"]]
        open(os.path.join(T, "lig.gro"), "w").write("ligand\n%d\n%s\n%s\n" % (len(lig), "\n".join(lig), ref[-1]))
        box = {3: "spc216.gro", 4: "tip4p.gro", 5: "tip5p.gro"}.get(st["water_sites"], "spc216.gro")
        g.run(["editconf", "-f", "lig.gro", "-o", "box.gro", "-bt", "dodecahedron", "-d", cfg["sampling"]["water_box_nm"], "-c"], T, "editconf.log")
        g.run(["solvate", "-cp", "box.gro", "-cs", box, "-p", "topol.top", "-o", "solv.gro"], T, "solvate.log")
        g.grompp(os.path.join(M, "water_em.mdp"), "solv.gro", "topol.top", "ions.tpr", T, log="grompp_ions.log")
        if st["salt_M"] > 0:
            g.run(["genion", "-s", "ions.tpr", "-p", "topol.top", "-o", "sys.gro", "-conc", f"{st['salt_M']:.4f}", "-neutral",
                   "-pname", st["ion_pos"][0], "-nname", st["ion_neg"][0]], T, "genion.log", stdin=f"{st['water']}\n")
        else:
            shutil.copy(os.path.join(T, "solv.gro"), os.path.join(T, "sys.gro"))
        log(cfg, "water: box built (" + ", ".join(l.strip() for l in open(top).read().split("[ molecules ]")[1].splitlines()
                                                     if l.strip() and not l.startswith(";")) + ")")
    steps = [("em", "water_em.mdp", os.path.join(T, "sys.gro"), None),
             ("nvt", "water_nvt.mdp", "em.gro", None), ("npt", "water_npt.mdp", "nvt.gro", "nvt.cpt"),
             ("eqA", "water_eqA.mdp", "npt.gro", "npt.cpt"),
             ("eqB_nvt", "water_eqB_nvt.mdp", "npt.gro", None), ("eqB", "water_eqB.mdp", "eqB_nvt.gro", "eqB_nvt.cpt")]
    for name, mdp, conf, cpt in steps:
        if os.path.exists(os.path.join(E, f"{name}.gro")):
            continue
        g.grompp(os.path.join(M, mdp), conf, top, f"{name}.tpr", E, log=f"grompp_{name}.log", cpt=cpt)
        g.mdrun(name, E, threads, pinoffset, log=f"{name}.out", minimize=(name == "em"))
        log(cfg, f"water: {name} done")
    log(cfg, "water: equilibrium end states done")


# --------------------------------------------------------------------------------------------- complex state B
def _p1l1(conf_gro, P1, L1):
    P1, L1 = int(P1), int(L1)
    lines = open(conf_gro).read().splitlines()
    xyz = lambda i: np.array([float(lines[i + 1][20:28]), float(lines[i + 1][28:36]), float(lines[i + 1][36:44])])
    b = [float(x) for x in lines[-1].split()]
    box = np.zeros((3, 3)); box[0, 0], box[1, 1], box[2, 2] = b[:3]
    if len(b) == 9:
        box[0, 1], box[0, 2], box[1, 0], box[1, 2], box[2, 0], box[2, 1] = b[3:]
    d = xyz(L1) - xyz(P1)
    for k in (2, 1, 0):                                   # triclinic minimum image (GROMACS box convention)
        d -= box[k] * np.round(d[k] / box[k, k])
    return float(np.linalg.norm(d))


def _replica(cfg, rep, t_ps, pinoffset, threads, st, errors):
    g = Gmx(cfg); M = cfg.path("mdp"); top = cfg.path("02_complex", "topology", "topol_abfe.top")
    D = cfg.path("02_complex", "eqB", f"rep{rep}"); os.makedirs(D, exist_ok=True)
    P1, L1, r0 = int(st["restraint"]["P"][0]), int(st["restraint"]["L"][0]), float(st["restraint"]["r0_nm"])
    try:
        if os.path.exists(os.path.join(D, "md.gro")):
            return
        if _gro_natoms(os.path.join(D, "start.gro")) == 0:
            raise SystemExit(f"rep{rep}: start.gro missing or empty")
        if not os.path.exists(os.path.join(D, "nvt.gro")):
            g.grompp(os.path.join(M, "complex_eqB_nvt.mdp"), "start.gro", top, "nvt.tpr", D, log="grompp_nvt.log")
            g.mdrun("nvt", D, threads, pinoffset, log="nvt.out")
        e = log_energies(os.path.join(D, "nvt.log"))
        d = _p1l1(os.path.join(D, "nvt.gro"), P1, L1)
        if "Harmonic Pot." not in e or abs(d - r0) > 0.2:
            raise SystemExit(f"rep{rep}: restraint guard failed after NVT (P1-L1 = {d:.3f} nm, reference {r0:.3f})")
        log(cfg, f"stateB rep{rep}: NVT ok, P1-L1 = {d:.3f} nm (ref {r0:.3f})")
        if not os.path.exists(os.path.join(D, "npt.gro")):
            g.grompp(os.path.join(M, "complex_eqB_npt.mdp"), "nvt.gro", top, "npt.tpr", D, log="grompp_npt.log", cpt="nvt.cpt")
            g.mdrun("npt", D, threads, pinoffset, log="npt.out")
        if not os.path.exists(os.path.join(D, "md.tpr")):
            g.grompp(os.path.join(M, "complex_eqB_md.mdp"), "npt.gro", top, "md.tpr", D, log="grompp_md.log", cpt="npt.cpt")
        extra = ["-cpi", "md.cpt"] if os.path.exists(os.path.join(D, "md.cpt")) else []
        g.mdrun("md", D, threads, pinoffset, extra=extra, log="md.out")
        d = _p1l1(os.path.join(D, "md.gro"), P1, L1)
        log(cfg, f"stateB rep{rep}: finished, final P1-L1 = {d:.3f} nm (ref {r0:.3f})")
        if abs(d - r0) > 0.2:
            raise SystemExit(f"rep{rep}: ligand left its restrained position (P1-L1 = {d:.3f} nm)")
    except BaseException as ex:                          # any failure must stop the stage
        errors.append(f"rep{rep}: {ex}"); log(cfg, f"stateB rep{rep}: FAILED: {ex}")


def stateb(cfg, reps=None):
    st = state(cfg); _need(st, "checked", "check")
    S = cfg["sampling"]; n = int(S["stateB_replicas"]); th = int(cfg["parallel"]["stateB_threads"])
    t0, t1 = [float(x) for x in S["stateA_window_ns"]]
    seeds = S.get("stateB_seed_times_ns") or [t0 + (t1 - t0) * (k + 0.5) / n for k in range(n)]
    reps = reps or list(range(1, n + 1))
    natoms = _gro_natoms(cfg.path("00_inputs", "ref.gro"))
    for rep in reps:                                    # starting frames: one at a time, verified, before any run
        D = cfg.path("02_complex", "eqB", f"rep{rep}"); os.makedirs(D, exist_ok=True)
        if _gro_natoms(os.path.join(D, "start.gro")) != natoms:
            _dump(cfg, seeds[rep - 1] * 1000, "start.gro", D, natoms)
            log(cfg, f"stateB rep{rep}: start frame t = {seeds[rep-1]:g} ns extracted ({natoms} atoms)")
    errors, threads = [], []
    for k, rep in enumerate(reps):
        th_ = threading.Thread(target=_replica, args=(cfg, rep, seeds[rep - 1] * 1000, k * th, th, st, errors))
        th_.start(); threads.append(th_); time.sleep(2)
    for t in threads:
        t.join()
    if errors:
        raise SystemExit("stateB: " + " | ".join(errors))
    log(cfg, "stateB: all replicas done")


# --------------------------------------------------------------------------------------------- frames
def _times(xtc, top_gro):
    import MDAnalysis as mda
    u = mda.Universe(top_gro, xtc)
    return np.array([ts.time for ts in u.trajectory])


def _pick(times, t0, t1, n):
    """End-anchored, evenly spaced frame selection inside [t0, t1] (ps). Returns 0-based frame indices."""
    w = np.where((times >= t0) & (times <= t1))[0]
    if len(w) < n:
        raise SystemExit(f"only {len(w)} frames in [{t0}, {t1}] ps, {n} requested")
    step = max(1, len(w) // n)
    idx = w[::-1][::step][:n][::-1]
    return idx


def _extract(cfg, tpr, xtc, top_gro, t0, t1, n, outdir, prefix, center):
    g = Gmx(cfg); os.makedirs(outdir, exist_ok=True)
    times = _times(xtc, top_gro)
    idx = _pick(times, t0, t1, n)
    ndx = os.path.join(outdir, f"{prefix}frames.ndx")
    with open(ndx, "w") as fh:
        fh.write("[ frames ]\n" + "\n".join(str(i + 1) for i in idx) + "\n")
    g.run(["trjconv", "-s", tpr, "-f", xtc, "-fr", ndx, "-pbc", "mol", "-center", "-sep", "-o", f"{prefix}frame_.gro"],
          outdir, f"{prefix}trjconv.log", stdin=f"{center}\nSystem\n")
    got = len(glob.glob(os.path.join(outdir, f"{prefix}frame_*.gro")))
    if got != n:
        raise SystemExit(f"frame extraction in {outdir}: expected {n}, got {got}")
    return {f"{prefix}frame_{k}": float(times[i]) for k, i in enumerate(idx)}


def frames(cfg, which=("complexA", "complexB", "water")):
    st = state(cfg); S = cfg["sampling"]; meta = st.get("frames", {})
    ref = cfg.path("00_inputs", "ref.gro"); c = cfg["system"]["center_group"]
    if "complexA" in which:
        t0, t1 = [float(x) * 1000 for x in S["stateA_window_ns"]]
        meta["complexA"] = _extract(cfg, cfg["system"]["production_tpr"], cfg["system"]["production_xtc"], ref, t0, t1,
                                    int(S["n_frames_A"]), cfg.path("02_complex", "framesA"), "", c)
    if "complexB" in which:
        meta["complexB"] = {}
        for rep in range(1, int(S["stateB_replicas"]) + 1):
            D = cfg.path("02_complex", "eqB", f"rep{rep}")
            tm = _times(os.path.join(D, "md.xtc"), os.path.join(D, "start.gro"))
            meta["complexB"].update(_extract(cfg, os.path.join(D, "md.tpr"), os.path.join(D, "md.xtc"), os.path.join(D, "start.gro"),
                                             tm[0] + float(S["stateB_discard_ns"]) * 1000, tm[-1], int(S["frames_per_replica"]),
                                             cfg.path("02_complex", "framesB"), f"rep{rep}_", c))
    if "water" in which:
        E = cfg.path("03_water", "equil")
        for s in ("A", "B"):
            tm = _times(os.path.join(E, f"eq{s}.xtc"), os.path.join(E, "npt.gro"))
            meta[f"water{s}"] = _extract(cfg, os.path.join(E, f"eq{s}.tpr"), os.path.join(E, f"eq{s}.xtc"), os.path.join(E, "npt.gro"),
                                         tm[0] + float(S["water_discard_ns"]) * 1000, tm[-1], int(S["n_frames_water"]),
                                         cfg.path("03_water", f"frames{s}"), "", st["lig_resname"])
    st["frames"] = meta; save_state(cfg, st)
    log(cfg, "frames: " + ", ".join(f"{k}: {len(v)}" for k, v in meta.items()))


# --------------------------------------------------------------------------------------------- transitions
def _first_last(dhdl):
    rows = [l for l in open(dhdl) if l.strip() and l[0] not in "#@"]
    return (float(rows[0].split()[1]), float(rows[-1].split()[1])) if rows else (None, None)


def transitions(cfg, leg, direction, tag="", stride=1, jobs=None, threads=None, pin0=0):
    st = state(cfg); g = Gmx(cfg)
    base = cfg.path("02_complex" if leg == "complex" else "03_water")
    top = cfg.path("02_complex", "topology", "topol_abfe.top") if leg == "complex" else cfg.path("03_water", "topology", "topol.top")
    mdp = cfg.path("mdp", f"trans_{leg}_{direction}{tag}.mdp")
    if not os.path.exists(mdp):
        raise SystemExit(f"missing {mdp}")
    src = os.path.join(base, f"frames{direction[0]}")
    fr = sorted(glob.glob(os.path.join(src, "*.gro")), key=lambda p: (len(p), p))[::stride]
    if not fr:
        raise SystemExit(f"no frames in {src}; run `abfeflow frames`")
    out = os.path.join(base, f"trans{direction}{tag}"); os.makedirs(out, exist_ok=True)
    P = cfg["parallel"]
    jobs = jobs or (P["complex_jobs"] if leg == "complex" else P["water_jobs"])
    threads = threads or (P["complex_threads"] if leg == "complex" else P["water_threads"])
    todo = [f for f in fr if not os.path.exists(os.path.join(out, os.path.basename(f)[:-4], "done"))]
    log(cfg, f"transitions {leg} {direction}{tag}: {len(fr)} frames, {len(todo)} to run, {jobs} jobs x {threads} threads")
    fails, lock = [], threading.Lock()

    def worker(w):
        for f in todo[w::jobs]:
            n = os.path.basename(f)[:-4]; D = os.path.join(out, n); os.makedirs(D, exist_ok=True)
            try:
                g.grompp(mdp, f, top, "tr.tpr", D)
                g.run(["mdrun", "-deffnm", "tr", "-dhdl", "dhdl.xvg"] + g.mdrun_opts(threads, pin0 + w * threads), D, "mdrun.out")
                open(os.path.join(D, "done"), "w").close()
            except GmxError as ex:
                with lock: fails.append(n)
            for junk in ("tr.trr", "tr.cpt", "tr_prev.cpt", "tr.xtc"):
                if os.path.exists(os.path.join(D, junk)): os.remove(os.path.join(D, junk))

    ts = [threading.Thread(target=worker, args=(w,)) for w in range(jobs)]
    [t.start() for t in ts]; [t.join() for t in ts]
    done = len(glob.glob(os.path.join(out, "*", "done")))
    log(cfg, f"transitions {leg} {direction}{tag}: {done}/{len(fr)} finished" + (f", FAILED: {sorted(fails)}" if fails else ""))
    # guard: reverse runs must start where forward runs end (same lambda=1 ensemble)
    fwd, rev = os.path.join(base, f"transA2B{tag}"), os.path.join(base, f"transB2A{tag}")
    ends = [_first_last(p)[1] for p in glob.glob(os.path.join(fwd, "*", "dhdl.xvg")) if os.path.exists(os.path.join(os.path.dirname(p), "done"))]
    starts = [_first_last(p)[0] for p in glob.glob(os.path.join(rev, "*", "dhdl.xvg")) if os.path.exists(os.path.join(os.path.dirname(p), "done"))]
    if ends and starts:
        a, b = np.median(ends), np.median(starts)
        msg = f"guard: dH/dl at lambda=1  forward end median {a:.0f}, reverse start median {b:.0f} kJ/mol"
        log(cfg, msg)
        if abs(b - a) > 1000:
            log(cfg, "WARNING: reverse runs start far from the forward end state - state B is not the restrained decoupled "
                     "ensemble (restraints missing during state-B sampling?). Do not analyse before fixing.")
    if fails:
        raise SystemExit(f"{len(fails)} transitions failed; inspect their mdrun.out, fix, and rerun (finished ones are skipped)")


# --------------------------------------------------------------------------------------------- status
def status(cfg):
    st = state(cfg)
    rows = [("setup", "restraint" in st), ("check", st.get("checked", False))]
    rows.append(("water equilibrium", os.path.exists(cfg.path("03_water", "equil", "eqB.gro"))))
    n = int(cfg["sampling"]["stateB_replicas"])
    rows.append((f"state B ({n} replicas)", all(os.path.exists(cfg.path("02_complex", "eqB", f"rep{r}", "md.gro")) for r in range(1, n + 1))))
    rows.append(("frames", "frames" in st))
    for leg, base in (("complex", "02_complex"), ("water", "03_water")):
        for d in ("A2B", "B2A"):
            tot = len(glob.glob(cfg.path(base, f"frames{d[0]}", "*.gro")))
            done = len(glob.glob(cfg.path(base, f"trans{d}", "*", "done")))
            rows.append((f"{leg} {d}", f"{done}/{tot}"))
    rows.append(("analysis", os.path.exists(cfg.path("04_analysis", "results.json"))))
    for k, v in rows:
        print(f"  {k:24s} {v if isinstance(v, str) else ('done' if v else '-')}")


# --------------------------------------------------------------------------------------------- adopt (legacy projects)
def adopt(cfg):
    """Register a project that was run with the original shell scripts (same directory layout), without touching its files."""
    import re
    from .mdp import parse, temperature
    st = state(cfg)
    itp = cfg.path("01_restraints", "boresch_restraints.itp")
    if not os.path.exists(itp):
        raise SystemExit(f"{itp} not found - is project_dir the existing ABFE directory?")
    txt = open(itp).read()
    m = re.search(r";\s*P1\s+(\S+)\s+P2\s+(\S+)\s+P3\s+(\S+)\s+L1\s+(\S+)\s+L2\s+(\S+)\s+L3\s+(\S+)", txt)
    rows = [l.split() for l in txt.splitlines() if l.strip() and l.strip()[0].isdigit()]
    b, a1, a2, d1, d2, d3 = rows[:6]
    dG = float(re.search(r"=\s*\+([\d.]+)\s*kJ/mol", txt).group(1))
    mdps = [p for p in (cfg["system"]["production_mdp"], cfg.path("00_inputs", "md_production.mdp")) if p and os.path.exists(p)]
    T = temperature(parse(mdps[0])) if mdps else 298.0
    st.update(dict(T=T, ligand=cfg["system"]["ligand_moltype"], gmx_version=Gmx(cfg).version(), adopted=True,
                   restraint=dict(P=[int(b[0]), int(a1[0]), int(d1[0])], L=[int(b[1]), int(a2[1]), int(d3[3])],
                                  P_label=list(m.groups()[:3]) if m else ["P1", "P2", "P3"],
                                  L_label=list(m.groups()[3:]) if m else ["L1", "L2", "L3"],
                                  r0_nm=float(b[3]), thA_deg=float(a1[4]), thB_deg=float(a2[4]), phiA_deg=float(d1[5]),
                                  phiB_deg=float(d2[5]), phiC_deg=float(d3[5]), k_dist=float(b[6]), k_ang=float(a1[7]),
                                  T=T, dG_r_kJ=dG, dG_r_kcal=dG / 4.184)))
    save_state(cfg, st)
    log(cfg, f"adopt: registered existing project (T={T:g} K, dG_r=+{dG/4.184:.2f} kcal/mol, anchors "
             f"{st['restraint']['P_label']} / {st['restraint']['L_label']})")
