"""Read a GROMACS topology (following local #includes) and write the two ABFE topologies."""
import os, re

INC = re.compile(r'^\s*#include\s+"([^"]+)"')


def _resolve(inc, base):
    p = inc if os.path.isabs(inc) else os.path.join(base, inc)
    return os.path.abspath(p) if os.path.exists(p) else None


def absolutize(text, base):
    """Rewrite relative #include paths that exist next to the topology into absolute paths
    (force-field includes such as amber19sb.ff/... are left for GROMACS to resolve)."""
    out = []
    for line in text.splitlines():
        m = INC.match(line)
        if m:
            p = _resolve(m.group(1), base)
            if p:
                line = line.replace(m.group(1), p)
        out.append(line)
    return "\n".join(out) + "\n"


def _walk(path, seen=None):
    """Yield (file, line) over a topology and every include that exists locally."""
    seen = seen or set()
    path = os.path.abspath(path)
    if path in seen:
        return
    seen.add(path)
    base = os.path.dirname(path)
    for line in open(path):
        m = INC.match(line)
        if m:
            p = _resolve(m.group(1), base)
            if p:
                yield from _walk(p, seen)
                continue
        yield path, line


class Topology:
    def __init__(self, top, ligand):
        self.top = os.path.abspath(top)
        self.ligand = ligand
        self.molecules = []          # [(name, count)] in order
        self.moltypes = {}           # name -> {'atoms': [...], 'bonds': [...], 'file': path}
        cur, sec = None, None
        for f, raw in _walk(self.top):
            line = raw.split(";")[0].strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("["):
                sec = line.strip("[] ").lower()
                if sec == "moleculetype":
                    cur = "__pending__"
                continue
            if sec == "moleculetype" and cur == "__pending__":
                cur = line.split()[0]
                self.moltypes[cur] = {"atoms": [], "bonds": [], "file": f}
                continue
            if sec == "atoms" and cur in self.moltypes:
                p = line.split()
                self.moltypes[cur]["atoms"].append(dict(nr=int(p[0]), type=p[1], resnr=int(p[2]), resname=p[3],
                                                        name=p[4], charge=float(p[6]) if len(p) > 6 else 0.0,
                                                        mass=float(p[7]) if len(p) > 7 else None))
            elif sec == "bonds" and cur in self.moltypes:
                p = line.split()
                self.moltypes[cur]["bonds"].append((int(p[0]), int(p[1])))
            elif sec == "molecules" and f == self.top:     # only the main file defines the system
                p = line.split()
                self.molecules.append((p[0], int(p[1])))
        if ligand not in self.moltypes:
            raise SystemExit(f"ligand moleculetype '{ligand}' not found in {top} or its local includes")
        if (ligand, 1) not in self.molecules:
            raise SystemExit(f"[ molecules ] must contain exactly one '{ligand}'")

    # ---- ligand properties
    @property
    def lig_atoms(self):
        return self.moltypes[self.ligand]["atoms"]

    @property
    def lig_bonds(self):
        return self.moltypes[self.ligand]["bonds"]

    @property
    def lig_resname(self):
        return self.lig_atoms[0]["resname"]

    @property
    def lig_charge(self):
        return sum(a["charge"] for a in self.lig_atoms)

    def heavy(self, a):
        return (a["mass"] or 0) > 1.5 if a["mass"] is not None else not a["name"].upper().startswith("H")

    # ---- solvent/ions
    def ions(self, water):
        lig = self.ligand
        names = [n for n, c in self.molecules if n not in (lig, water) and self.atoms_per(n) in (1, None)
                 and not n.lower().startswith("protein")]
        pos, neg = [], []
        for n in names:
            q = self.mol_charge(n)
            if q is None:
                q = 1 if n.upper() in ("NA", "K", "NA+", "K+", "SOD", "POT") else -1 if n.upper() in ("CL", "CL-", "CLA") else 0
            (pos if q > 0 else neg if q < 0 else []).append(n)
        return pos, neg

    def atoms_per(self, name):
        return len(self.moltypes[name]["atoms"]) if name in self.moltypes else None

    def mol_charge(self, name):
        if name not in self.moltypes:
            return None
        return sum(a["charge"] for a in self.moltypes[name]["atoms"])

    def count(self, name):
        return sum(c for n, c in self.molecules if n == name)

    # ---- writers
    def _body_without_system(self):
        text = open(self.top).read()
        cut = re.search(r"^\s*\[\s*system\s*\]", text, re.M)
        body = text[:cut.start()] if cut else text
        return absolutize(body, os.path.dirname(self.top))

    def write_complex(self, out, restraint_itp):
        text = absolutize(open(self.top).read(), os.path.dirname(self.top))
        text += ("\n; ---- abfeflow: Boresch restraints (force constants 0 in state A, full in state B)\n"
                 f'#include "{os.path.abspath(restraint_itp)}"\n')
        with open(out, "w") as fh:
            fh.write(text)

    def write_water(self, out):
        text = self._body_without_system()
        text += f"\n[ system ]\n{self.ligand} in water (abfeflow water leg)\n\n[ molecules ]\n{self.ligand}  1\n"
        with open(out, "w") as fh:
            fh.write(text)
