"""Generate every free-energy .mdp from the production .mdp so the two can never disagree."""
import os

INHERIT = ["dt", "constraints", "constraint-algorithm", "lincs-iter", "lincs-order", "cutoff-scheme", "nstlist",
           "pbc", "coulombtype", "coulomb-modifier", "rcoulomb", "epsilon-r", "fourierspacing", "pme-order",
           "ewald-rtol", "vdwtype", "vdw-modifier", "rvdw", "rvdw-switch", "dispcorr", "verlet-buffer-tolerance",
           "periodic-molecules", "comm-mode", "nstcomm"]
PRESSURE = ["pcoupltype", "ref-p", "compressibility"]


def _norm(k):
    return k.strip().lower().replace("_", "-")


def parse(path):
    d = {}
    for line in open(path):
        line = line.split(";")[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            d[_norm(k)] = v.strip()
    return d


def temperature(prod):
    return float(prod.get("ref-t", "300").split()[0])


SOFTCORE = {
    "gapsys": """sc-function                  = gapsys
sc-gapsys-scale-linpoint-lj  = 0.85
sc-gapsys-scale-linpoint-q   = 0.3
sc-gapsys-sigma-lj           = 0.3
""",
    "beutler": """sc-function                  = beutler
sc-alpha                     = 0.5
sc-power                     = 1
sc-r-power                   = 6
sc-sigma                     = 0.3
sc-coul                      = yes
""",
}


class MdpWriter:
    def __init__(self, cfg, prod_mdp):
        self.cfg = cfg
        self.prod = parse(prod_mdp)
        self.T = temperature(self.prod)
        self.dt = float(self.prod.get("dt", "0.002"))
        if "pcoupl" not in self.prod or self.prod["pcoupl"].lower() in ("no", ""):
            raise SystemExit("production .mdp has no pressure coupling; abfeflow assumes NPT production")

    def _common(self):
        s = "; ---- inherited from the production .mdp\n"
        for k in INHERIT:
            if k in self.prod:
                s += f"{k:24s}= {self.prod[k]}\n"
        return s

    def _ensemble(self, npt):
        tau_p = max(float(self.prod.get("tau-p", "2.0").split()[0]), 2.0)
        s = ("; ---- ensemble (sd = Langevin, recommended for decoupled states)\n"
             f"integrator              = sd\ntc-grps                 = System\ntau-t                   = 1.0\n"
             f"ref-t                   = {self.T:g}\n")
        if npt:
            s += f"pcoupl                  = C-rescale\ntau-p                   = {tau_p:g}\n"
            for k in PRESSURE:
                if k in self.prod:
                    s += f"{k:24s}= {self.prod[k]}\n"
        else:
            s += "pcoupl                  = no\n"
        return s

    def _fe(self, lam, dlam=0.0, nstdhdl=0):
        return ("; ---- alchemical decoupling of the ligand (complex: Boresch restraints on the same lambda)\n"
                f"free-energy             = yes\ncouple-moltype          = {self.cfg['system']['ligand_moltype']}\n"
                "couple-lambda0          = vdw-q\ncouple-lambda1          = none\ncouple-intramol         = no\n"
                f"init-lambda             = {lam}\ndelta-lambda            = {dlam:.10g}\nnstdhdl                 = {nstdhdl}\n"
                "dhdl-derivatives        = yes\nseparate-dhdl-file      = yes\ndhdl-print-energy       = no\n"
                + SOFTCORE[self.cfg["softcore"]])

    def run(self, title, ps, lam, *, npt=True, genvel=False, cont=False, xtc_ps=10.0, dlam=0.0, nstdhdl=0, nstcalc=100):
        nsteps = int(round(ps / self.dt))
        nxtc = int(round(xtc_ps / self.dt)) if xtc_ps else 0
        s = (f"title                   = {title}\nnsteps                  = {nsteps}\n"
             "nstxout                 = 0\nnstvout                 = 0\nnstfout                 = 0\n"
             f"nstcalcenergy           = {nstcalc}\nnstenergy               = {max(nstcalc, 5000 // nstcalc * nstcalc)}\n"
             f"nstlog                  = {max(nstcalc, 5000 // nstcalc * nstcalc)}\nnstxout-compressed      = {nxtc}\n"
             f"continuation            = {'yes' if cont else 'no'}\ngen-vel                 = {'yes' if genvel else 'no'}\n")
        if genvel:
            s += f"gen-temp                = {self.T:g}\ngen-seed                = -1\n"
        return s + self._common() + self._ensemble(npt) + self._fe(lam, dlam, nstdhdl)

    def em(self):
        return ("title = water leg energy minimisation\nintegrator = steep\nemtol = 500\nemstep = 0.01\nnsteps = 50000\n"
                "nstcalcenergy = 100\n" + self._common() + self._fe(0))

    def write_all(self, outdir):
        os.makedirs(outdir, exist_ok=True)
        S = self.cfg["sampling"]
        tc, tw = float(S["transition_ps"]["complex"]), float(S["transition_ps"]["water"])
        nd = int(S["nstdhdl"]); x = float(S.get("xtc_ps", 10))
        files = {
            "water_em.mdp": self.em(),
            "water_nvt.mdp": self.run("water NVT (lambda 0)", S["water_nvt_ps"], 0, npt=False, genvel=True, xtc_ps=x),
            "water_npt.mdp": self.run("water NPT (lambda 0)", S["water_npt_ps"], 0, cont=True, xtc_ps=x),
            "water_eqA.mdp": self.run("water state A (coupled)", S["water_eq_ns"] * 1000, 0, cont=True, xtc_ps=x),
            "water_eqB_nvt.mdp": self.run("water state B NVT (decoupled)", S["water_nvt_ps"], 1, npt=False, genvel=True, xtc_ps=x),
            "water_eqB.mdp": self.run("water state B (decoupled)", S["water_eq_ns"] * 1000, 1, cont=True, xtc_ps=x),
            "complex_eqB_nvt.mdp": self.run("complex state B NVT (decoupled + restrained)", S["stateB_nvt_ps"], 1, npt=False, genvel=True, xtc_ps=x),
            "complex_eqB_npt.mdp": self.run("complex state B NPT", S["stateB_npt_ps"], 1, cont=True, xtc_ps=x),
            "complex_eqB_md.mdp": self.run("complex state B production", S["stateB_length_ns"] * 1000, 1, cont=True, xtc_ps=x),
            "complex_check.mdp": self.run("zero-step check", 0, 0, npt=True, genvel=True, nstcalc=1, xtc_ps=0),
        }
        for leg, ps in (("complex", tc), ("water", tw)):
            n = int(round(ps / self.dt))
            for d, lam, sign in (("A2B", 0, 1), ("B2A", 1, -1)):
                files[f"trans_{leg}_{d}.mdp"] = self.run(f"{leg} transition {d}, {ps:g} ps", ps, lam, genvel=True, xtc_ps=0,
                                                       dlam=sign / n, nstdhdl=nd, nstcalc=nd)
            if leg == "complex":
                n2 = 2 * n
                for d, lam, sign in (("A2B", 0, 1), ("B2A", 1, -1)):
                    files[f"trans_complex_{d}_long.mdp"] = self.run(f"complex transition {d}, {2*ps:g} ps (length check)", 2 * ps,
                                                                  lam, genvel=True, xtc_ps=0, dlam=sign / n2, nstdhdl=nd, nstcalc=nd)
        for name, text in files.items():
            with open(os.path.join(outdir, name), "w") as fh:
                fh.write(text)
        return sorted(files)
