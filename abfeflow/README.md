# abfeflow: non-equilibrium absolute binding free energies with GROMACS

End-to-end workflow for **absolute binding free energies (ABFE)** of a neutral ligand, using
non-equilibrium (fast-growth) alchemical transitions in GROMACS. Analysis is pmx-compatible:
BAR, Crooks Gaussian intersection and Jarzynski, validated against `pmx analyse` to ±0.01 kcal/mol.

You provide the bound-state production run, its topology and its `.mdp`, plus one YAML file.
abfeflow does the rest:

```
setup ─▶ check ─▶ ┬─ stateb (complex, decoupled + restrained, N replicas) ─┐
                  └─ water  (ligand in water, both end states) ────────────┴▶ frames ─▶ transitions (4 sets) ─▶ analyse ─▶ report
```

## Install

```bash
unzip abfeflow_v0.2.0.zip -d ~/SOFT/          # -> ~/SOFT/abfeflow
conda create -n abfe python=3.11 -y && conda activate abfe     # or reuse an existing env (e.g. pmx)
pip install -e ~/SOFT/abfeflow                # pulls numpy, scipy, matplotlib, MDAnalysis, PyYAML
abfeflow --version                            # 0.2.0
```
Upgrading from 0.1.x: `pip uninstall -y abfeflow`, delete the old folder, then install as above.
Existing project folders and yaml files work unchanged.
GROMACS 2022 or newer is required for the default Gapsys soft-core. pmx itself is **not** needed;
you can still cross-check with `pmx analyse` because the dhdl files are standard.

## Quick start (new system)

```bash
mkdir my_abfe && cd my_abfe
abfeflow init                           # writes abfe.yaml - edit the `system` block
abfeflow setup                          # inputs, restraints, topologies, all .mdp files
abfeflow check                          # grompp both end states; restraints must switch with lambda
abfeflow run                            # state B + water leg (concurrently), frames, all transitions
abfeflow analyse
abfeflow report                         # 05_report/: figures, tables, methods, report.md
```
`abfeflow run` can be replaced by the individual stages (`stateb`, `water`, `frames`,
`transitions --leg complex --dir A2B`, ...). Every stage is restart-safe: rerun the same command after an
interruption and finished work is skipped. Check progress with `abfeflow status`.

## What you must provide (`abfe.yaml`, `system` block)

| key | meaning |
|---|---|
| `production_tpr`, `production_xtc` | the unrestrained bound-state run (state A); all atoms in the xtc |
| `production_mdp` | the `.mdp` of that run; cut-offs, PME, dispersion correction, constraints, T and P are inherited |
| `topology` | the `.top` used for that run (local `#include`s are followed and made absolute) |
| `ligand_moltype` | the ligand's `[ moleculetype ]` name (must appear once in `[ molecules ]`) |

Typical edits elsewhere: `sampling.stateA_window_ns` (equilibrated part of production),
`restraints.exclude_residues` (flexible loops/helices never to be used as anchors),
`gromacs.gmx`, `hardware.cores`, the `parallel` block, and `reference.dG_exp` for the report.

## What each stage does, and the guards that stop it

| stage | does | stops if |
|---|---|---|
| `setup` | copies inputs, builds `ref.gro`, inherits `.mdp` settings, picks Boresch anchors automatically, computes ΔG_restr, writes `topol_abfe.top` (restraints appended) and the water-leg topology | ligand is charged; ligand atoms in the structure don't match the topology; no anchor set passes the geometric filters |
| `check` | grompp at λ=0 and λ=1 with `-maxwarn 0`, zero-step runs | restraint energy is not 0 at λ=0 and > 0 at λ=1 |
| `stateb` | N replicas from evenly spaced production frames: NVT → NPT → production at λ=1 | after NVT, no restraint energy or P1–L1 off by > 0.2 nm; the same check at the end |
| `water` | box (same water model, same salt concentration as the complex), EM, NVT, NPT, both end states | any grompp warning |
| `frames` | evenly spaced snapshots, made whole and centred (`trjconv -fr`) | wrong number of frames |
| `transitions` | concurrent jobs share the GPU, each pinned to its own cores | reports failed frames; warns if reverse runs start far from where forward runs end (the signature of a broken state B) |
| `analyse` | integrates works, BAR / CGI / Jarzynski with bootstrap errors, per state-B replica (all A frames), per state-A time block, convergence | warns on poor overlap, BAR–CGI disagreement, replica spread > 1 kcal/mol, state-A substates |

### Automatic restraint selection
Protein anchors are backbone atoms (default Cα) within `cutoff_nm` of the ligand, rigid
(RMSF ≤ `rmsf_max_A` after superposition) and outside `exclude_residues`. Ligand anchors are three
consecutive atoms of a ring found from the ligand's bond graph; if there is no ring, three bonded heavy atoms are used.
Sets with near-linear angles are rejected. The set with the lowest mean restraint energy over the
unrestrained ensemble wins. It also checks ligand pose stability and warns about multiple binding modes.
On T4L L99A/M102Q + 2-propylphenol it reproduces the hand-selected anchors exactly. For manual anchors,
set `restraints.mode: manual` and `manual: {P: [p1,p2,p3], L: [l1,l2,l3]}`.

## Using it on a project run with the original shell scripts

The directory layout is the same, so an existing project can be analysed and reported without
re-running anything:
```bash
cd ABFE_2PP_3HTB                        # the existing project
abfeflow init -c abfe.yaml              # set project_dir: .  and the system/reference blocks
abfeflow adopt                          # reads 01_restraints/boresch_restraints.itp; changes no files
abfeflow analyse && abfeflow report
```

## Outputs

```
01_restraints/   boresch_restraints.itp, boresch.json, boresch_timeseries.{dat,png,pdf}
04_analysis/     results.json (everything), works_*.csv (one work value per transition)
05_report/       fig1_cycle        thermodynamic-cycle decomposition with experiment
                 fig2_work_distributions   forward / reverse works, Gaussian fits, BAR
                 fig3_convergence  BAR vs number of transitions
                 fig4_complex_diagnostics  forward work vs state-A time; reverse work per state-B replica
                 fig5_comparison   ABFE vs experiment and other estimates (e.g. LIE)
                 figS1_restraints  Boresch coordinates in state A
                 table_*.{csv,md,tex}, methods.md (draft methods paragraph), report.md
```

## Limitations (by design, flagged rather than silently handled)
- Neutral ligands only. Charged ligands need a finite-size or co-alchemical-ion correction.
- One ligand copy; no cofactors or metals in the decoupled region.
- Membrane systems inherit semi-isotropic coupling from the production `.mdp`, but anchor selection
  has not been tested on them.
- Local execution only (one workstation, shared GPU). The commands each stage runs are logged in each
  directory and in `abfeflow.log`, so a SLURM wrapper is straightforward.


## Choosing transition length

Run a short pilot (100 ps transitions, ~10 per direction) first and read `dissipation_f/_r` in
`04_analysis/results.json`. Dissipation falls roughly as 1/τ, and BAR needs forward/reverse overlap ≥ 0.1:
aim for ≲ 2 kcal/mol dissipation per direction. Typical outcome: water leg 200–500 ps; small cavities
(T4L) 500 ps; large or deeply buried sites (AChE gorge, ~100 k atoms) 1 ns.

## Hardware notes

* `hardware.cores` = physical cores to use. Jobs are pinned at offsets `k × threads`; keep
  `jobs × threads ≤ cores` in each `parallel` entry.
* `pinstride: 2` only if `lscpu -e` shows logical CPUs 0 and 1 on the same core (offsets are scaled for you).
* Transitions use the sd integrator, so the update step runs on the CPU; several jobs share one GPU well.

## Changelog

* **0.2.0** – consolidated release of 0.1.1–0.1.3 fixes; pin offsets scaled by `pinstride`; replica/block
  spread warnings only when statistically significant (> 2 SE); convergence curve without duplicate subsets;
  AChE 32-core example.
* 0.1.3 – EM runs with `-pme cpu`; GROMACS fatal-error text shown in the log.
* 0.1.1 – sequential, verified start-frame extraction for state-B replicas.
