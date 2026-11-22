# Experiments on the SMS++ instances (Data/T-Ramp/*.nc4).
#
# Entry points (one core per run, runs spread over `Nbworkers` workers):
#   run_train                trains the policy (SDDP, 3-bin or interval formulation); records bounds, timings,
#                            schedules and cuts
#   run_simulate             replays the policy at given training times and estimates its cost by simulation
#   run_simulate_iterations  same, for given iterations instead of times
#
# Usage, from the repository root (first time: `julia --project=. -e 'import Pkg; Pkg.instantiate()'`):
#
#     include("runSMS.jl")
#     size_list = [10, 50]                   # number of units: 10, 20, 50, 75, 100, 150, 200
#     id_list   = 1:5                        # instance id, Data/T-Ramp/<size>_0_<id>_w.nc4 (1:11 for size 200)
#     methods   = ["interval_dual", "bin3"]  # keys of `MSUC.METHODS` (see also interval_projection/minimum/maximum,
#                                            # bin3_gamma)
#     run_train(size_list, id_list, methods, [3600]; S_list = [3, 5], sigma_list = [0.25, 1.0])
#     run_simulate(size_list, id_list, methods, [3600], [60, 600, 3600], 1000; S_list = [3, 5], sigma_list = [0.25, 1.0])
#     run_simulate_iterations(size_list, id_list, methods, [3600], [10, 50, 100], 1000; S_list = [3, 5], sigma_list = [0.25, 1.0])
#
# `run_simulate*` can be called later, in another session: everything it needs is in each run's folder.
# Options (defaults in parentheses); each list gives one run per element of the Cartesian product:
#   T (24) horizon, S_list ([5]) ARMA innovation support size, sigma_list ([1.0]) noise scale (0 = deterministic),
#   mip_gap_list ([nothing]) root MIP gap (nothing = solver default), passes_per_mip_list ([1]) passes per root MIP.
# The seed is fixed to 1 (`MSUC.SEED`) for training and simulation. Run logic lives in `src/functions/run/`.
#
# Results: results_sms/<method>_T<T>_S<S>_sigma<s>_passes<p>_gap<g>/<instance>/time_<t>/
#   config.csv, log.csv, timers.csv   run parameters; bound/incumbent/time per iteration; per-phase call counts and times
#   phases.csv                        cumulative time per phase and iteration (nested phases: `parent/child`)
#   states.csv, schedules.csv         sampled states; root-MIP schedule (selected intervals, nonzero x) per iteration
#   used_components.csv, cuts.csv     stage cuts restricted to the x components used by at least one schedule
#   simulation_<N>.csv                per-time summary (bound, #cuts, expected cost, standard error, 95% CI)
#   simulation_<t>_<N>.csv            dispatch cost of each trajectory at time t
#   simulation_iterations_<N>.csv, simulation_it<k>_<N>.csv   same, for `run_simulate_iterations`

import Pkg
Pkg.activate(".")

using Distributed

# Keep Nbworkers * GurobiThreads <= physical cores: time is the measured quantity, so an oversubscribed
# machine would skew the results. `NBWORKERS=2 julia` overrides the number of workers.
GurobiThreads = 1
Nbworkers = parse(Int, get(ENV, "NBWORKERS", "10"))

# `nprocs() > 1` only holds with real workers (the master is counted in `nworkers()`).
if nprocs() > 1
    rmprocs(workers())
end
addprocs(Nbworkers)

@everywhere import Pkg
@everywhere Pkg.activate(".")
@everywhere using MSUC
@everywhere using Gurobi

# One Gurobi environment per worker, read locally by the jobs: `pmap` sends nothing (Distributed would
# otherwise try to copy the master's environment).
@everywhere const GRB_ENV = Gurobi.Env()
@everywhere Gurobi.GRBsetintparam(GRB_ENV, "Threads", $GurobiThreads)
@everywhere MSUC.set_run_optimizer!(() -> Gurobi.Optimizer(GRB_ENV))

"""
    run_train(size_list, id_list, methods, time_max_list; T, S_list, sigma_list, mip_gap_list,
              passes_per_mip_list)

Train one model per combination of (size, id, method, time limit, S, sigma, mip_gap, passes_per_mip),
`Nbworkers` at a time, and write each run's results to its own folder.
"""
function run_train(
    size_list,
    id_list,
    method_list,
    time_max_list;
    T = 24,
    S_list = [5],
    sigma_list = [1.0],
    mip_gap_list = [nothing],
    passes_per_mip_list = [1],
)
    combos = MSUC.run_combos(
        size_list, id_list, method_list, time_max_list;
        T, S_list, sigma_list, passes_per_mip_list, mip_gap_list,
    )
    pmap(combos; on_error = e -> (@warn("job d'entrainement en echec", exception = e); nothing)) do (instance, method, time_max, settings)
        MSUC.train_job(instance, method, time_max, settings)
        MSUC.release_memory()
    end
    return
end

"""
    run_simulate(size_list, id_list, methods, time_max_list, time_list, N; T, S_list, sigma_list,
                 mip_gap_list, passes_per_mip_list)

Simulate the policy of each trained run (same arguments as `run_train`) at the times in `time_list`
(seconds since the start of training) on `N` trajectories. If `S^T <= 4096`, all trajectories are
enumerated and the cost is exact.
"""
function run_simulate(
    size_list,
    id_list,
    method_list,
    time_max_list,
    time_list,
    N::Int;
    T = 24,
    S_list = [5],
    sigma_list = [1.0],
    mip_gap_list = [nothing],
    passes_per_mip_list = [1],
)
    combos = MSUC.run_combos(
        size_list, id_list, method_list, time_max_list;
        T, S_list, sigma_list, passes_per_mip_list, mip_gap_list,
    )
    pmap(combos; on_error = e -> (@warn("job de simulation en echec", exception = e); nothing)) do (instance, method, time_max, settings)
        folder = MSUC.results_folder(instance, method, time_max, settings)
        MSUC.simulate_job(folder, instance, method, time_list, N, settings)
        MSUC.release_memory()
    end
    return
end

"""
    run_simulate_iterations(size_list, id_list, methods, time_max_list, iteration_list, N; T, S_list,
                            sigma_list, mip_gap_list, passes_per_mip_list)

Like `run_simulate`, but for given iterations: the policy of iteration k is the schedule x^k with the
stage cuts up to iteration k-1. Iterations missing from a run are skipped with a warning.
"""
function run_simulate_iterations(
    size_list,
    id_list,
    method_list,
    time_max_list,
    iteration_list,
    N::Int;
    T = 24,
    S_list = [5],
    sigma_list = [1.0],
    mip_gap_list = [nothing],
    passes_per_mip_list = [1],
)
    combos = MSUC.run_combos(
        size_list, id_list, method_list, time_max_list;
        T, S_list, sigma_list, passes_per_mip_list, mip_gap_list,
    )
    pmap(combos; on_error = e -> (@warn("job de simulation en echec", exception = e); nothing)) do (instance, method, time_max, settings)
        folder = MSUC.results_folder(instance, method, time_max, settings)
        MSUC.simulate_iterations_job(folder, instance, method, iteration_list, N, settings)
        MSUC.release_memory()
    end
    return
end
