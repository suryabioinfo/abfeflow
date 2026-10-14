"""Automatic Boresch restraint selection from the unrestrained bound-state trajectory.

Protein anchors P1-P3: backbone atoms (default CA) within `cutoff_nm` of the ligand, rigid
(RMSF after superposition <= rmsf_max_A) and outside user-excluded flexible regions.
Ligand anchors L1-L3: three consecutive atoms of a ring (from the ligand bond graph);
if the ligand has no ring, the three most rigid bonded heavy atoms.
Ranking: <U> = 1/2 sum K var(x), the mean restraint energy the unrestrained ensemble would
feel; low <U> means the restraints barely perturb the bound state.
"""
import itertools, json, os
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import minimize_vectors

KB = 0.0083144626
V0 = 1.66054                                     # nm^3, 1 M


def dG_restraint(r0, thA, thB, kr, ka, T):
    """Analytic cost (kJ/mol, positive) of imposing the six Boresch restraints on a free ligand at 1 M."""
    kT = KB * T
    arg = 8 * np.pi ** 2 * V0 * np.sqrt(kr * ka ** 5) / (r0 ** 2 * np.sin(thA) * np.sin(thB) * (2 * np.pi * kT) ** 3)
    return kT * np.log(arg)


def _ang(p1, p2, p3):
    v1, v2 = p1 - p2, p3 - p2
    c = (v1 * v2).sum(-1) / np.linalg.norm(v1, axis=-1) / np.linalg.norm(v2, axis=-1)
    return np.arccos(np.clip(c, -1, 1))


def _dih(p1, p2, p3, p4):
    b0, b1, b2 = p1 - p2, p3 - p2, p4 - p3
    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1n).sum(-1, keepdims=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdims=True) * b1n
    return np.arctan2((np.cross(b1n, v) * w).sum(-1), (v * w).sum(-1))


def _circ(x):
    m = np.arctan2(np.sin(x).mean(), np.cos(x).mean())
    return m, np.angle(np.exp(1j * (x - m))).std()


def _kabsch_rmsf(X, iters=3):
    Y = X.copy()
    for _ in range(iters):
        ref = Y.mean(0); ref -= ref.mean(0); out = []
        for x in X:
            xc = x - x.mean(0)
            V, S, Wt = np.linalg.svd(xc.T @ ref)
            D = np.diag([1, 1, np.sign(np.linalg.det(V @ Wt))])
            out.append(xc @ (V @ D @ Wt))
        Y = np.array(out)
    return np.sqrt(((Y - Y.mean(0)) ** 2).sum(-1).mean(0)), Y


def _parse_ranges(items):
    out = set()
    for it in items or []:
        it = str(it)
        if "-" in it:
            a, b = it.split("-"); out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(it))
    return out


def ligand_triples(lig_atoms, lig_bonds, heavy_idx):
    """Ordered (L1,L2,L3) local 0-based triples along bonds, preferring ring atoms (2-core of the heavy graph)."""
    nb = {i: set() for i in heavy_idx}
    for a, b in lig_bonds:
        a, b = a - 1, b - 1
        if a in nb and b in nb:
            nb[a].add(b); nb[b].add(a)
    core = {i: set(v) for i, v in nb.items()}
    changed = True
    while changed:                                # strip leaves -> atoms on cycles (and linkers between rings)
        changed = False
        for i in [i for i, v in core.items() if len(v) < 2]:
            for j in core.pop(i):
                core[j].discard(i)
            changed = True
    graph, ring = (core, True) if len(core) >= 3 else (nb, False)
    trip = [(i, j, k) for j in graph for i in graph[j] for k in graph[j] if i != k]
    return trip, ring


def select(cfg, ref_gro, xtc, lig_atoms, lig_bonds, lig_heavy, lig_resname, T, outdir, log=print):
    R = cfg["restraints"]; S = cfg["sampling"]
    kr, ka = float(R["k_dist"]), float(R["k_ang"])
    u = mda.Universe(ref_gro, xtc)
    lig = u.select_atoms(f"resname {lig_resname}")
    if len(lig) != len(lig_atoms) or any(a.name != b["name"] for a, b in zip(lig, lig_atoms)):
        raise SystemExit(f"ligand atoms in {ref_gro} (resname {lig_resname}) do not match the ligand topology")
    water = cfg["system"]["water_moltype"]
    excl = _parse_ranges(R.get("exclude_residues"))
    bb = R.get("backbone_atom", "CA")
    cand = [a for a in u.select_atoms(f"name {bb} and not resname {lig_resname} {water} SOL WAT HOH TIP3")
            if {"N", "C"} <= set(a.residue.atoms.names) and a.resid not in excl]
    t0, t1 = [float(x) * 1000 for x in S["stateA_window_ns"]]
    frames = [ts.frame for ts in u.trajectory if t0 <= ts.time <= t1]
    if len(frames) < 20:
        raise SystemExit(f"only {len(frames)} trajectory frames inside stateA_window_ns; need >= 20")
    log(f"restraints: {len(frames)} frames in {S['stateA_window_ns']} ns, {len(cand)} candidate {bb} atoms before filters")

    cidx = np.array([a.index for a in cand])
    Xl, Xc = [], []
    for f in frames:
        ts = u.trajectory[f]
        box = ts.dimensions
        pl = lig.positions
        cen = pl.mean(0)
        pl = cen + minimize_vectors(pl - cen, box)
        cen = pl.mean(0)
        pc = cen + minimize_vectors(u.atoms.positions[cidx] - cen, box)
        Xl.append(pl / 10); Xc.append(pc / 10)
    Xl, Xc = np.array(Xl), np.array(Xc)
    cen = Xl.mean(1)
    near = np.linalg.norm(Xc - cen[:, None], axis=2).mean(0) < float(R["cutoff_nm"])
    cand = [c for c, k in zip(cand, near) if k]; Xc = Xc[:, near]
    if len(cand) < 3:
        raise SystemExit("fewer than 3 candidate protein anchors near the ligand; increase restraints.cutoff_nm")
    rmsf, _ = _kabsch_rmsf(Xc)
    rigid = rmsf * 10 <= float(R["rmsf_max_A"])
    log(f"restraints: {len(cand)} anchors within {R['cutoff_nm']} nm, {rigid.sum()} with RMSF <= {R['rmsf_max_A']} A")
    cand = [c for c, k in zip(cand, rigid) if k]; Xc = Xc[:, rigid]; rmsf = rmsf[rigid]

    # pose stability: ligand heavy-atom RMSD after superposing each frame on the anchor candidates
    heavy = np.array(lig_heavy)
    pose = []
    ref_c = Xc.mean(0); ref_c0 = ref_c - ref_c.mean(0)
    for x, l in zip(Xc, Xl):
        xc0 = x - x.mean(0); V, _, Wt = np.linalg.svd(xc0.T @ ref_c0)
        Rm = V @ np.diag([1, 1, np.sign(np.linalg.det(V @ Wt))]) @ Wt
        pose.append((l[heavy] - x.mean(0)) @ Rm)
    pose = np.array(pose); mean_pose = pose.mean(0)
    pose_rmsd = np.sqrt(((pose - mean_pose) ** 2).sum(-1).mean(-1)) * 10
    multi = np.percentile(pose_rmsd, 95) > 2.0
    if multi:
        log(f"WARNING: ligand pose RMSD 95th percentile = {np.percentile(pose_rmsd, 95):.2f} A (> 2 A): possible multiple binding modes")

    trip, ring = ligand_triples(lig_atoms, lig_bonds, list(heavy))
    log(f"restraints: {len(trip)} ligand anchor triples ({'ring atoms' if ring else 'no ring: bonded heavy atoms'})")

    res = []
    nC = len(cand)
    for p1, p2, p3 in itertools.permutations(range(nC), 3):
        P1, P2, P3 = Xc[:, p1], Xc[:, p2], Xc[:, p3]
        if not (0.4 <= np.linalg.norm(P1 - P2, axis=-1).mean() <= 1.2): continue
        if not (0.4 <= np.linalg.norm(P2 - P3, axis=-1).mean() <= 1.2): continue
        if not (30 <= np.degrees(_ang(P3, P2, P1).mean()) <= 150): continue
        for l1, l2, l3 in trip:
            L1, L2, L3 = Xl[:, l1], Xl[:, l2], Xl[:, l3]
            r = np.linalg.norm(P1 - L1, axis=-1)
            if not (0.4 <= r.mean() <= 1.2): continue
            tA, tB = _ang(P2, P1, L1), _ang(P1, L1, L2)
            dA, dB = np.degrees(tA), np.degrees(tB)
            if min(np.percentile(dA, 1), np.percentile(dB, 1)) < 30: continue
            if max(np.percentile(dA, 99), np.percentile(dB, 99)) > 150: continue
            fA, fB, fC = _dih(P3, P2, P1, L1), _dih(P2, P1, L1, L2), _dih(P1, L1, L2, L3)
            (mA, sA), (mB, sB), (mC, sC) = _circ(fA), _circ(fB), _circ(fC)
            U = 0.5 * (kr * r.var() + ka * (tA.var() + tB.var() + sA ** 2 + sB ** 2 + sC ** 2))
            res.append(dict(U=U, p=(p1, p2, p3), l=(l1, l2, l3), r0=r.mean(), sr=r.std(), tA=tA.mean(), stA=tA.std(),
                            tB=tB.mean(), stB=tB.std(), fA=mA, sfA=sA, fB=mB, sfB=sB, fC=mC, sfC=sC,
                            series=(r, dA, dB, np.degrees(fA), np.degrees(fB), np.degrees(fC))))
    if not res:
        raise SystemExit("no anchor set passed the geometric filters; relax cutoff_nm / rmsf_max_A or give manual anchors")
    res.sort(key=lambda d: d["U"])
    best = res[0]
    lig0 = lig[0].index
    P = [cand[i] for i in best["p"]]
    Lg = [int(lig0) + int(i) + 1 for i in best["l"]]
    Pg = [int(a.index) + 1 for a in P]
    dG = dG_restraint(best["r0"], best["tA"], best["tB"], kr, ka, T)
    info = dict(P=Pg, L=Lg, P_label=[f"{a.resname}{a.resid}:{a.name}" for a in P],
                L_label=[lig_atoms[i]["name"] for i in best["l"]], r0_nm=best["r0"],
                thA_deg=np.degrees(best["tA"]), thB_deg=np.degrees(best["tB"]),
                phiA_deg=np.degrees(best["fA"]), phiB_deg=np.degrees(best["fB"]), phiC_deg=np.degrees(best["fC"]),
                sd=dict(r_A=best["sr"] * 10, thA=np.degrees(best["stA"]), thB=np.degrees(best["stB"]),
                        phiA=np.degrees(best["sfA"]), phiB=np.degrees(best["sfB"]), phiC=np.degrees(best["sfC"])),
                k_dist=kr, k_ang=ka, T=T, dG_r_kJ=dG, dG_r_kcal=dG / 4.184, mean_U_kJ=best["U"], n_valid_sets=len(res),
                n_frames=len(frames), ring_anchors=ring, pose_rmsd_p95_A=float(np.percentile(pose_rmsd, 95)),
                multiple_modes_warning=bool(multi),
                top5=[dict(U=d["U"], P=[f"{cand[i].resname}{cand[i].resid}" for i in d["p"]],
                           L=[lig_atoms[i]["name"] for i in d["l"]]) for d in res[:5]])
    os.makedirs(outdir, exist_ok=True)
    write_itp(os.path.join(outdir, "boresch_restraints.itp"), info)
    with open(os.path.join(outdir, "boresch.json"), "w") as fh:
        json.dump(info, fh, indent=2, default=float)
    t = np.array([u.trajectory[f].time for f in frames]) / 1000
    np.savetxt(os.path.join(outdir, "boresch_timeseries.dat"), np.column_stack([t] + list(best["series"])),
               header="time_ns r_nm thetaA thetaB phiA phiB phiC (deg)", fmt="%.4f")
    log(f"restraints: chosen P={info['P_label']} L={info['L_label']}  r0={best['r0']:.3f} nm  "
        f"dG_r=+{dG/4.184:.2f} kcal/mol  (<U>={best['U']:.2f} kJ/mol, {len(res)} valid sets)")
    return info


def write_itp(path, i):
    P, L, kr, ka = i["P"], i["L"], i["k_dist"], i["k_ang"]
    with open(path, "w") as fh:
        fh.write(f"""; Boresch restraints written by abfeflow
; P1 {i['P_label'][0]}  P2 {i['P_label'][1]}  P3 {i['P_label'][2]}   L1 {i['L_label'][0]}  L2 {i['L_label'][1]}  L3 {i['L_label'][2]}
; force constants are 0 in state A and full in state B (switched by the same lambda as the decoupling)
; dG_r (analytic, T={i['T']:g} K) = +{i['dG_r_kJ']:.3f} kJ/mol = +{i['dG_r_kcal']:.3f} kcal/mol
[ intermolecular_interactions ]
[ bonds ]
{P[0]:>7} {L[0]:>7}   6  {i['r0_nm']:.4f}  0.0  {i['r0_nm']:.4f}  {kr}
[ angles ]
{P[1]:>7} {P[0]:>7} {L[0]:>7}   1  {i['thA_deg']:.3f}  0.0  {i['thA_deg']:.3f}  {ka}
{P[0]:>7} {L[0]:>7} {L[1]:>7}   1  {i['thB_deg']:.3f}  0.0  {i['thB_deg']:.3f}  {ka}
[ dihedrals ]
{P[2]:>7} {P[1]:>7} {P[0]:>7} {L[0]:>7}   2  {i['phiA_deg']:.3f}  0.0  {i['phiA_deg']:.3f}  {ka}
{P[1]:>7} {P[0]:>7} {L[0]:>7} {L[1]:>7}   2  {i['phiB_deg']:.3f}  0.0  {i['phiB_deg']:.3f}  {ka}
{P[0]:>7} {L[0]:>7} {L[1]:>7} {L[2]:>7}   2  {i['phiC_deg']:.3f}  0.0  {i['phiC_deg']:.3f}  {ka}
""")


def manual(cfg, ref_gro, xtc, lig_atoms, T, outdir, log=print):
    """User-specified anchors: reference values are still measured from the trajectory."""
    R = cfg["restraints"]; S = cfg["sampling"]
    Pg, Lg = R["manual"]["P"], R["manual"]["L"]
    u = mda.Universe(ref_gro, xtc)
    t0, t1 = [float(x) * 1000 for x in S["stateA_window_ns"]]
    X = []
    for ts in u.trajectory:
        if t0 <= ts.time <= t1:
            p = u.atoms.positions[[i - 1 for i in Pg + Lg]]
            p = p[0] + minimize_vectors(p - p[0], ts.dimensions)
            X.append(p / 10)
    X = np.array(X); P1, P2, P3, L1, L2, L3 = [X[:, k] for k in range(6)]
    r = np.linalg.norm(P1 - L1, axis=-1); tA, tB = _ang(P2, P1, L1), _ang(P1, L1, L2)
    (mA, sA), (mB, sB), (mC, sC) = _circ(_dih(P3, P2, P1, L1)), _circ(_dih(P2, P1, L1, L2)), _circ(_dih(P1, L1, L2, L3))
    kr, ka = float(R["k_dist"]), float(R["k_ang"])
    dG = dG_restraint(r.mean(), tA.mean(), tB.mean(), kr, ka, T)
    lab = lambda i: f"{u.atoms[i-1].resname}{u.atoms[i-1].resid}:{u.atoms[i-1].name}"
    info = dict(P=Pg, L=Lg, P_label=[lab(i) for i in Pg], L_label=[u.atoms[i - 1].name for i in Lg], r0_nm=r.mean(),
                thA_deg=np.degrees(tA.mean()), thB_deg=np.degrees(tB.mean()), phiA_deg=np.degrees(mA),
                phiB_deg=np.degrees(mB), phiC_deg=np.degrees(mC),
                sd=dict(r_A=r.std() * 10, thA=np.degrees(tA.std()), thB=np.degrees(tB.std()), phiA=np.degrees(sA),
                        phiB=np.degrees(sB), phiC=np.degrees(sC)),
                k_dist=kr, k_ang=ka, T=T, dG_r_kJ=dG, dG_r_kcal=dG / 4.184, n_frames=len(X), manual=True)
    os.makedirs(outdir, exist_ok=True)
    write_itp(os.path.join(outdir, "boresch_restraints.itp"), info)
    json.dump(info, open(os.path.join(outdir, "boresch.json"), "w"), indent=2, default=float)
    log(f"restraints (manual): r0={r.mean():.3f} nm dG_r=+{dG/4.184:.2f} kcal/mol")
    return info
