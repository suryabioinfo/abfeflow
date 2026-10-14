"""Configuration: one YAML file describes the system; everything else has defaults."""
import copy, os, yaml

TEMPLATE = """\
# ---------------------------------------------------------------------------
# abfeflow configuration. Paths may be relative to this file.
# Only the `system` block is mandatory; everything else has defaults.
# ---------------------------------------------------------------------------
project_dir: ABFE_project            # all outputs go here

system:
  production_tpr: md.tpr             # bound-state (state A) production run
  production_xtc: md.xtc             # its trajectory (all atoms)
  production_mdp: md.mdp             # nonbonded / T / P settings are inherited from this
  topology: topol.top                # topology used for the production run
  ligand_moltype: LIG                # [ moleculetype ] name of the ligand
  reference_gro: null                # optional structure with the same atom order (default: made from the tpr)
  water_moltype: SOL
  center_group: Protein              # index group used to centre frames

gromacs:
  gmx: gmx_mpi                       # GROMACS binary (2022 or newer for Gapsys soft-core)
  gpu: true                          # -nb gpu -pme gpu
  pme_on_gpu: true                   # set false if FE + GPU PME is not supported on your build
  maxwarn: 0

hardware:
  cores: 40                          # physical cores to use (pinning offsets 0..cores-1)
  pinstride: 1                       # 2 only if logical CPUs 0 and 1 share a physical core (check lscpu -e)

sampling:
  stateA_window_ns: [200, 500]       # part of the production run treated as equilibrated state A
  n_frames_A: 100                    # forward (A->B) transitions in the complex
  stateB_replicas: 3                 # independent decoupled-state runs
  stateB_seed_times_ns: null         # default: centres of equal sub-windows of stateA_window_ns
  stateB_length_ns: 50
  stateB_discard_ns: 16              # frames are taken after this time
  frames_per_replica: 34             # reverse (B->A) transitions per replica
  water_eq_ns: 10                    # per end state
  water_discard_ns: 2
  n_frames_water: 100
  transition_ps: {complex: 500, water: 200}   # large/buried sites: complex 1000, water 500 (see README)
  nstdhdl: 10                        # dH/dlambda output interval (steps)
  water_box_nm: 1.5                  # solute-box distance for the water leg
  stateB_nvt_ps: 100                 # short equilibration lengths (advanced)
  stateB_npt_ps: 1000
  water_nvt_ps: 100
  water_npt_ps: 1000
  xtc_ps: 10                         # trajectory output interval of equilibrium runs

softcore: gapsys                     # gapsys | beutler

restraints:
  mode: auto                         # auto | manual
  manual: null                       # {P: [p1,p2,p3], L: [l1,l2,l3]} global 1-based atom numbers
  exclude_residues: []               # e.g. ["105-117"]  flexible regions never used as anchors
  backbone_atom: CA
  cutoff_nm: 1.3                     # candidate CA atoms within this distance of the ligand
  rmsf_max_A: 0.6                    # candidate CA atoms must be at least this rigid
  k_dist: 4184.0                     # kJ/mol/nm^2
  k_ang: 41.84                       # kJ/mol/rad^2

parallel:                            # concurrent jobs share the GPU, each pinned to its own cores
  stateB_threads: 12
  water_eq_threads: 4
  complex_jobs: 4
  complex_threads: 10
  water_jobs: 8
  water_threads: 5

analysis:
  nboots: 200
  stateA_blocks: 3                   # time blocks of state A used as a convergence diagnostic

reference:                           # optional, for tables and figures (kcal/mol)
  dG_exp: null
  dG_exp_err: null
  label_exp: Experiment
  other: {}                          # e.g. {LIE: -7.62}
  dG_hyd_ref: null                   # optional hydration free energy reference (kcal/mol) for a sanity check
"""

DEFAULTS = yaml.safe_load(TEMPLATE)


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


class Config(dict):
    """dict with attribute-free access helpers and resolved absolute paths."""
    @property
    def root(self):
        return self["project_dir"]

    def path(self, *parts):
        return os.path.join(self["project_dir"], *parts)


def load(path):
    path = os.path.abspath(path)
    with open(path) as fh:
        user = yaml.safe_load(fh) or {}
    cfg = Config(_merge(DEFAULTS, user))
    base = os.path.dirname(path)
    absify = lambda p: p if (p is None or os.path.isabs(p)) else os.path.normpath(os.path.join(base, p))
    cfg["project_dir"] = absify(cfg["project_dir"])
    for k in ("production_tpr", "production_xtc", "production_mdp", "topology", "reference_gro"):
        cfg["system"][k] = absify(cfg["system"][k])
    cfg["_config_file"] = path
    for k in ("production_tpr", "production_xtc", "production_mdp", "topology"):
        if cfg["system"][k] is None:
            raise SystemExit(f"config: system.{k} is required")
    return cfg


def write_template(path):
    if os.path.exists(path):
        raise SystemExit(f"{path} exists; not overwriting")
    with open(path, "w") as fh:
        fh.write(TEMPLATE)
