"""abfeflow command line."""
import argparse, os, sys, threading
from . import config, stages, __version__

STAGES = """\
  init          write a template config (abfe.yaml)
  setup         copy inputs, inherit .mdp settings, select Boresch restraints, write topologies and all .mdp files
  check         grompp both end states (-maxwarn 0) and verify the restraints switch with lambda
  water         water leg: build box, equilibrate, run both end states
  stateb        complex state B: decoupled + restrained replicas (guarded)
  frames        extract snapshots for all transitions
  transitions   run non-equilibrium transitions (--leg complex|water --dir A2B|B2A)
  run           everything from `check` to the last transition, in order (state B and water concurrently)
  analyse       integrate works, BAR/CGI/Jarzynski, per-replica and per-block checks, close the cycle
  report        figures (PNG+PDF), tables (CSV/MD/LaTeX), methods paragraph, report.md
  status        progress overview
  adopt         register an existing project run with the original shell scripts (then analyse/report)
"""


def main(argv=None):
    ap = argparse.ArgumentParser(prog="abfeflow", description="Non-equilibrium ABFE workflow (GROMACS + pmx-compatible analysis)",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=STAGES)
    ap.add_argument("stage", choices=["init", "setup", "check", "water", "stateb", "frames", "transitions", "run",
                                      "analyse", "analyze", "report", "status", "adopt"])
    ap.add_argument("-c", "--config", default="abfe.yaml")
    ap.add_argument("--leg", choices=["complex", "water"])
    ap.add_argument("--dir", choices=["A2B", "B2A"])
    ap.add_argument("--tag", default="", help="e.g. _long for the 2x-length convergence check")
    ap.add_argument("--stride", type=int, default=1, help="use every n-th frame (length checks)")
    ap.add_argument("--jobs", type=int); ap.add_argument("--threads", type=int)
    ap.add_argument("--pinoffset", type=int, default=0)
    ap.add_argument("--reps", type=int, nargs="*", help="state-B replicas to run (default all)")
    ap.add_argument("--version", action="version", version=__version__)
    a = ap.parse_args(argv)

    if a.stage == "init":
        config.write_template(a.config); print(f"wrote {a.config}; edit the `system` block, then run `abfeflow setup`"); return
    cfg = config.load(a.config)
    os.makedirs(cfg["project_dir"], exist_ok=True)
    if a.stage == "setup":
        stages.setup(cfg)
    elif a.stage == "check":
        stages.check(cfg)
    elif a.stage == "water":
        stages.water(cfg, pinoffset=a.pinoffset if a.pinoffset else None, threads=a.threads)
    elif a.stage == "stateb":
        stages.stateb(cfg, a.reps)
    elif a.stage == "frames":
        stages.frames(cfg)
    elif a.stage == "transitions":
        if not (a.leg and a.dir):
            ap.error("transitions needs --leg and --dir")
        stages.transitions(cfg, a.leg, a.dir, a.tag, a.stride, a.jobs, a.threads, a.pinoffset)
    elif a.stage == "run":
        st = stages.state(cfg)
        if not st.get("checked"):
            stages.check(cfg)
        err = []
        def _w():
            try: stages.water(cfg)
            except BaseException as ex: err.append(f"water: {ex}")
        tw = threading.Thread(target=_w); tw.start()
        stages.stateb(cfg); tw.join()
        if err: raise SystemExit(err[0])
        stages.frames(cfg)
        for leg in ("water", "complex"):
            for d in ("A2B", "B2A"):
                stages.transitions(cfg, leg, d)
        print("all simulations finished -> `abfeflow analyse` then `abfeflow report`")
    elif a.stage in ("analyse", "analyze"):
        from . import analysis; analysis.run(cfg, a.tag)
    elif a.stage == "report":
        from . import report; report.run(cfg, a.tag)
    elif a.stage == "status":
        stages.status(cfg)
    elif a.stage == "adopt":
        stages.adopt(cfg)


if __name__ == "__main__":
    main()
