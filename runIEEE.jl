# Experiments on the IEEE 118-bus instance (Data/IEEE/118_syst): 54 units, DC network flow.
# Works like `runSMS.jl` (see it for the options and output files), restricted to this instance.
#
#     include("runIEEE.jl")
#     methods = ["interval_dual", "bin3"]
#     run_train(methods, [3600]; S_list = [3, 5], sigma_list = [0.25, 1.0])
#     run_simulate(methods, [3600], [60, 600, 3600], 1000; S_list = [3, 5], sigma_list = [0.25, 1.0])
#     run_simulate_iterations(methods, [3600], [10, 50, 100], 1000; S_list = [3, 5], sigma_list = [0.25, 1.0])
#
# Differences from runSMS.jl: T <= 24 (the load profile covers 24 hours), and `mip_gap_list` defaults to
# [1e-2] since the root MIP dominates the training time.
# Results: results_ieee/<method>_T<T>_S<S>_sigma<s>_passes<p>_gap<g>/118_syst/time_<t>/

import Pkg
Pkg.activate(".")

using Distributed

# Keep Nbworkers * GurobiThreads <= physical cores: time is the measured quantity, so an oversubscribed
# machine would skew the results. There is at most one run per method and setting, so 2 workers
# (`NBWORKERS=2 julia`) suffice to compare two methods.
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
    run_train(methods, time_max_list; T, S_list, sigma_list, mip_gap_list, passes_per_mip_list)

Train one model per combination of (method, time limit, S, sigma, mip_gap, passes_per_mip) on the IEEE
118-bus instance, `Nbworkers` at a time, and write each run's results to its own folder.
"""
function run_train(
    method_list,
    time_max_list;
    T = 24,
    S_list = [5],
    sigma_list = [1.0],
    mip_gap_list = [1e-2],
    passes_per_mip_list = [1],
)
    combos = MSUC.ieee_combos(
        method_list, time_max_list;
        T, S_list, sigma_list, passes_per_mip_list, mip_gap_list,
    )
    pmap(combos; on_error = e -> (@warn("job d'entrainement en echec", exception = e); nothing)) do (instance, method, time_max, settings)
        MSUC.train_job(instance, method, time_max, settings)
        MSUC.release_memory()
    end
    return
end

"""
    run_simulate(methods, time_max_list, time_list, N; T, S_list, sigma_list, mip_gap_list,
                 passes_per_mip_list)

Simulate the policy of each trained run (same arguments as `run_train`) at the times in `time_list`
(seconds since the start of training) on `N` trajectories. If `S^T <= 4096`, all trajectories are
enumerated and the cost is exact.
"""
function run_simulate(
    method_list,
    time_max_list,
    time_list,
    N::Int;
    T = 24,
    S_list = [5],
    sigma_list = [1.0],
    mip_gap_list = [1e-2],
    passes_per_mip_list = [1],
)
    combos = MSUC.ieee_combos(
        method_list, time_max_list;
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
    run_simulate_iterations(methods, time_max_list, iteration_list, N; T, S_list, sigma_list, mip_gap_list,
                            passes_per_mip_list)

Like `run_simulate`, but for given iterations: the policy of iteration k is the schedule x^k with the
stage cuts up to iteration k-1. Iterations missing from a run are skipped with a warning.
"""
function run_simulate_iterations(
    method_list,
    time_max_list,
    iteration_list,
    N::Int;
    T = 24,
    S_list = [5],
    sigma_list = [1.0],
    mip_gap_list = [1e-2],
    passes_per_mip_list = [1],
)
    combos = MSUC.ieee_combos(
        method_list, time_max_list;
        T, S_list, sigma_list, passes_per_mip_list, mip_gap_list,
    )
    pmap(combos; on_error = e -> (@warn("job de simulation en echec", exception = e); nothing)) do (instance, method, time_max, settings)
        folder = MSUC.results_folder(instance, method, time_max, settings)
        MSUC.simulate_iterations_job(folder, instance, method, iteration_list, N, settings)
        MSUC.release_memory()
    end
    return
end
