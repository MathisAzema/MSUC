# MSUC.jl

Code for the paper **"Interval-based cuts for stochastic dual dynamic programming in multistage stochastic unit commitment"** (Mathis Azéma).

The unit commitment schedule is decided the day before and stays fixed, while the dispatch adapts to the observed net demand. Only the root problem is non-convex; given the schedule, each stage is a linear program, so SDDP applies. The package implements SDDP with the standard 3-bin cuts and with the stronger interval cuts of the paper.

This repository is a fork of [SDDP.jl](https://github.com/odow/SDDP.jl) (module renamed `MSUC`) extended with the unit commitment model and the interval cut generator.

## Installation

Requires Julia 1.12.6 (the version the `Manifest.toml` is resolved for) and Gurobi with a valid license (`GRB_LICENSE_FILE`).

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Where to start

The notebooks walk through the code step by step; run them from the repository root.

| Notebook | Content |
|---|---|
| [`small_example.ipynb`](small_example.ipynb) | One-unit deterministic example of the paper (Sect. 5.1): exact optimum, cut comparison between the two formulations |
| [`validation.ipynb`](validation.ipynb) | Train and simulate a policy on an SMS++ instance: bound trajectory, policy cost, root MIP time, cut diagnostics |
| [`validationIEEE.ipynb`](validationIEEE.ipynb) | Same as `validation.ipynb` on the IEEE 118-bus system |

## Repository structure

```
src/
  MSUC.jl                 module definition and includes
  functions/              the code, one folder per concern
    algorithm/              SDDP loop: train, iterations, forward/backward passes, subproblem solves
    cuts/                   3-bin and interval cut generation, duals
    model/                  policy graph, root and stage problems, schedule, state variables
    instance/               instance parsing, ARMA uncertainty process, thermal unit model
    run/                    experiment helpers: models, jobs, policy recording and reloading
    reference/              exact multistage optimum (scenario tree), used as a reference
    tools/                  sampling, simulation, training log
  struct_functions/       the types needed by the files of functions/ (same sub-folders and file names)
examples/                 short scripts checking the SDDP bounds against the exact optimum
runSMS.jl, runIEEE.jl    parallel experiment campaigns (usage in the header of each file)
Data/
  T-Ramp/                 SMS++ instances (.nc4)
  IEEE/118_syst/          IEEE 118-bus data
```

`MSUC.jl` includes all the types first, then the functions.

## Scripts

- `julia --project=. examples/toy_convergence.jl`: small stochastic instance with a computable exact optimum; checks the lower bound and the simulated policy cost around it for both formulations.
- `julia --project=. examples/compare_formulations.jl`: exact, interval and 3-bin bounds on a deterministic instance.
- `runSMS.jl` and `runIEEE.jl`: train policies for a grid of settings across several workers, then simulate them at given times or iterations. See the header of each file for the options and the output files.

Correspondence with the paper's experiments (Sect. 5): the methods `bin3`, `bin3_gamma`, `interval_dual`, `interval_minimum` and `interval_maximum` are 3-bin SDDP, 3-bin SDDP with the interval variables in the first-stage MILP, and interval-based SDDP with the dual-guided, minimum and maximum rules; `passes_per_mip` is n_FB, `sigma_list` is σ and `S_list` is the support size S.

## License

Files derived from SDDP.jl keep their Mozilla Public License 2.0 headers.
