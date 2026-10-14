"""Thin, logged wrapper around the GROMACS command line."""
import os, subprocess, shlex


class GmxError(RuntimeError):
    pass


def fatal_text(path, tail=12):
    """The GROMACS 'Fatal error' block if present (gmx_mpi prints it before the MPI abort text), else the last lines."""
    lines = open(path).read().splitlines()
    for i, l in enumerate(lines):
        if l.strip().startswith(("Fatal error", "Inconsistency in user input", "Error in user input",
                                 "Invalid input value", "Feature not implemented", "Software inconsistency")):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("For more information") and j < i + 15:
                j += 1
            return "\n".join(lines[max(0, i - 2):j])
    return "\n".join(lines[-tail:])


class Gmx:
    def __init__(self, cfg):
        self.exe = cfg["gromacs"]["gmx"]
        self.gpu = cfg["gromacs"]["gpu"]
        self.pme_gpu = cfg["gromacs"].get("pme_on_gpu", True)
        self.maxwarn = int(cfg["gromacs"]["maxwarn"])
        self.pinstride = int(cfg["hardware"]["pinstride"])
        self.real_mpi = "_mpi" in os.path.basename(self.exe.split()[0]) or "mpirun" in self.exe

    def run(self, args, cwd, log, stdin=None, check=True):
        cmd = shlex.split(self.exe) + [str(a) for a in args]
        with open(os.path.join(cwd, log), "w") as fh:
            fh.write("$ " + " ".join(cmd) + "\n")
            fh.flush()
            p = subprocess.run(cmd, cwd=cwd, input=stdin, text=True, stdout=fh, stderr=subprocess.STDOUT)
        if check and p.returncode != 0:
            raise GmxError(f"{' '.join(cmd[:3])} failed in {cwd} (see {log}):\n" + fatal_text(os.path.join(cwd, log)))
        return p.returncode

    def grompp(self, mdp, conf, top, out, cwd, log="grompp.log", cpt=None, ref=None):
        a = ["grompp", "-f", mdp, "-c", conf, "-p", top, "-o", out, "-po", "mdout.mdp", "-maxwarn", self.maxwarn]
        if cpt: a += ["-t", cpt]
        if ref: a += ["-r", ref]
        return self.run(a, cwd, log)

    def mdrun_opts(self, threads, pinoffset=None, minimize=False):
        o = ["-ntomp", threads]
        if not self.real_mpi:
            o = ["-ntmpi", 1] + o                 # thread-MPI build: one rank per job
        if pinoffset is not None:
            # offsets are given in cores; with pinstride 2 (SMT siblings adjacent) core k starts at logical CPU 2k
            o += ["-pin", "on", "-pinoffset", int(pinoffset) * self.pinstride, "-pinstride", self.pinstride]
        if self.gpu:                                  # PME on GPU is not supported for energy minimisation
            o += ["-nb", "gpu", "-pme", "gpu" if (self.pme_gpu and not minimize) else "cpu"]
        return o

    def mdrun(self, deffnm, cwd, threads, pinoffset=None, extra=(), log="mdrun.out", check=True, minimize=False):
        return self.run(["mdrun", "-deffnm", deffnm] + self.mdrun_opts(threads, pinoffset, minimize) + list(extra), cwd, log, check=check)

    def version(self):
        try:
            out = subprocess.run(shlex.split(self.exe) + ["--version"], text=True, capture_output=True).stdout
            for l in out.splitlines():
                if "GROMACS version" in l:
                    return l.split(":", 1)[1].strip()
        except Exception:
            pass
        return "unknown"
