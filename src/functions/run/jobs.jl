# Idle workers never trigger GC, so Gurobi models of a finished job would stay in memory.
# Call after the job has returned, not inside it.
function release_memory()
    GC.gc(true)
    GC.gc(true)   # second pass reclaims what finalizers released
    Sys.islinux() && ccall(:malloc_trim, Cint, (Cint,), 0)
    return
end

function save_run(folder, model, snapshots, schedules, states, passes, config)
    log = model.most_recent_training_results.log
    CSV.write(
        joinpath(folder, "log.csv"),
        DataFrame(
            iteration = [l.iteration for l in log],
            bound = [l.bound for l in log],
            incumbent = [l.incumbent for l in log],
            time = [l.time for l in log],
            total_solves = [l.total_solves for l in log],
        ),
    )

    paths = sort(collect(union(keys.(snapshots)...)))
    phases = DataFrame(iteration = 1:length(snapshots))
    for path in paths
        phases[!, path] = [get(s, path, (0, 0.0))[2] for s in snapshots]
    end
    CSV.write(joinpath(folder, "phases.csv"), phases)

    final = flatten_timer(model.timer_output)
    CSV.write(
        joinpath(folder, "timers.csv"),
        DataFrame(
            phase = paths,
            ncalls = [get(final, p, (0, 0.0))[1] for p in paths],
            seconds = [get(final, p, (0, 0.0))[2] for p in paths],
        ),
    )

    # Sanity check: each iteration adds `passes` cuts per stage node (none at the last node), which is
    # how the cuts available at each iteration are recovered. An iteration stopped by the time limit
    # right after its MIP did no pass (`one_iteration`).
    skipped = get(model.ext, :passes_skipped, false)
    nodes = length(model.nodes)
    for i in 2:nodes
        expected = i == nodes ? 0 : passes * (length(schedules) - skipped)
        length(model[i].cuts) - 1 == expected ||
            error("noeud $i : $(length(model[i].cuts) - 1) coupes, $expected attendues")
    end

    all(length(s) == (skipped && k == length(states) ? 0 : passes) for (k, s) in enumerate(states)) &&
        length(states) == length(log) ||
        error("etats echantillonnes : $passes trajectoires par iteration attendues")

    write_policy(folder, model, schedules)

    config = merge(
        config,
        (
            iterations = length(log),
            total_time = log[end].time,
            final_bound = log[end].bound,
        ),
    )
    CSV.write(joinpath(folder, "config.csv"), DataFrame([config]))
    return
end

# Trains a model and writes bounds, phase timings, schedules and cuts to `results_folder(...)`.
function train_job(instance_name, method, time_max, settings)
    folder = results_folder(instance_name, method, time_max, settings)
    mkpath(folder)

    model = build_model(instance_name, method, settings, run_optimizer())
    record, record_pass, snapshots, schedules, states = make_recorder(model)

    Random.seed!(SEED)
    train(
        model;
        time_limit = time_max,
        mip_gap = settings.mip_gap,
        passes_per_mip = settings.passes_per_mip,
        interval_rule = METHODS[method][3],
        print_level = 0,
        run_numerical_stability_report = false,
        forward_pass_callback = record_pass,
        post_iteration_callback = record,
    )

    config = (
        instance = instance_name,
        method = method,
        time_limit = time_max,
        T = settings.T,
        S = settings.S,
        sigma = settings.sigma,
        mip_gap = something(settings.mip_gap, missing),
        passes_per_mip = settings.passes_per_mip,
    )
    save_run(folder, model, snapshots, schedules, states, settings.passes_per_mip, config)
    return
end

# Simulates the policy of iteration `k` (schedule x^k, stage cuts up to iteration k-1) on `N` paths
# and writes the per-path dispatch cost to `simulation_<label>_<N>.csv`. Must be called with
# increasing `k`: cuts are never removed from `model`.
function simulate_iteration!(model, data, folder, log, k, label, N, settings, loaded)
    load_cuts!(model, data, settings.passes_per_mip, k - 1, loaded)
    schedule = data.schedules[k]
    seconds = @elapsed sim = simulate(model; schedule, samples = N, seed = SEED)
    CSV.write(
        joinpath(folder, "simulation_$(label)_$(N).csv"),
        DataFrame(path = 1:length(sim.costs), dispatch_cost = sim.costs, weight = sim.weights),
    )
    return (
        bound = log.bound[k],
        cuts = sum(loaded),
        policy_cost = sim.cost,
        commitment_cost = sim.commitment_cost,
        dispatch_cost = sim.dispatch_cost,
        stderr = sim.stderr,
        ci95_low = sim.ci95[1],
        ci95_high = sim.ci95[2],
        exhaustive = sim.exhaustive,
        npaths = sim.npaths,
        simulate_seconds = seconds,
    )
end

# Replays the policy of the training run in `folder` at each time in `time_list`: the policy of the
# last iteration finished by then. Times before the first iteration ends are skipped.
function simulate_job(folder, instance_name, method, time_list, N, settings)
    log = CSV.read(joinpath(folder, "log.csv"), DataFrame)

    model = build_model(instance_name, method, settings, run_optimizer())
    data = read_policy(folder, model)

    loaded = zeros(Int, length(model.nodes))
    rows = NamedTuple[]
    for time_limit in sort(time_list)
        k = count(<=(time_limit), log.time)
        if k == 0
            @warn "aucune iteration terminee a $(time_limit) s : instant ignore" folder
            continue
        end
        row = simulate_iteration!(model, data, folder, log, k, time_limit, N, settings, loaded)
        push!(rows, (time_limit = time_limit, iteration = k, row...))
    end
    CSV.write(joinpath(folder, "simulation_$(N).csv"), DataFrame(rows))
    return
end

# Like `simulate_job`, but for given iterations instead of times. Iterations missing from the
# training run are skipped with a warning.
function simulate_iterations_job(folder, instance_name, method, iteration_list, N, settings)
    log = CSV.read(joinpath(folder, "log.csv"), DataFrame)

    model = build_model(instance_name, method, settings, run_optimizer())
    data = read_policy(folder, model)

    loaded = zeros(Int, length(model.nodes))
    rows = NamedTuple[]
    for k in sort(unique(iteration_list))
        if !(1 <= k <= nrow(log))
            @warn "iteration $(k) absente de l'entrainement ($(nrow(log)) iterations) : ignoree" folder
            continue
        end
        row = simulate_iteration!(model, data, folder, log, k, "it$(k)", N, settings, loaded)
        push!(rows, (iteration = k, time = log.time[k], row...))
    end
    CSV.write(joinpath(folder, "simulation_iterations_$(N).csv"), DataFrame(rows))
    return
end
