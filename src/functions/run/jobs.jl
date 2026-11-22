# --- Un run d'entrainement et la simulation de sa politique (voir `runSMS.jl`) -------------------

# Rend au systeme la memoire du job qui vient de se terminer sur ce worker : un worker inactif ne
# declenche jamais le ramasse-miettes, donc les modeles Gurobi d'un long run resteraient en memoire.
# A appeler une fois le job revenu, pas dans le job.
function release_memory()
    GC.gc(true)
    GC.gc(true)   # le second passage recupere ce que les finaliseurs ont libere
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

    # Chaque iteration ajoute `passes` coupes a chaque noeud d'etape (sauf au dernier, sans valeur
    # future) : c'est ce qui permet de savoir quelles coupes existaient a chaque iteration.
    # Une iteration arretee par la limite de temps apres son MIP n'a pas fait de passe (`one_iteration`).
    skipped = get(model.ext, :passes_skipped, false)
    nodes = length(model.nodes)
    for i in 2:nodes
        expected = i == nodes ? 0 : passes * (length(schedules) - skipped)
        length(model[i].cuts) - 1 == expected ||
            error("noeud $i : $(length(model[i].cuts) - 1) coupes, $expected attendues")
    end

    # Une trajectoire par passe : `passes` par iteration.
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

# Entraine un modele (`train` est appele tel quel, avec un releve apres chaque iteration), puis
# ecrit dans `results_folder(...)` la borne, les temps par phase, les plannings et les coupes.
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

# Simule la politique de l'iteration `k` (planning x^k, coupes d'etape jusqu'a l'iteration k-1, chargees dans
# `model` a partir de `loaded`) sur `N` trajectoires, ecrit le cout de dispatch de chacune dans
# `simulation_<label>_<N>.csv` et renvoie les colonnes de resume communes aux deux entrees (`simulate_job` et
# `simulate_iterations_job`). Il faut l'appeler avec des `k` croissants : les coupes ne sont jamais retirees.
function simulate_iteration!(model, data, folder, log, k, label, N, settings, loaded)
    load_cuts!(model, data, settings.passes_per_mip, k - 1, loaded)
    schedule = data.schedules[k]              # le planning x^k de l'iteration k
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

# Rejoue, par instants croissants, la politique de l'entrainement `folder` : a un instant donne, la
# politique de la derniere iteration k terminee (`time` <= instant), c'est-a-dire le planning x^k et les
# coupes d'etape jusqu'a l'iteration k-1, simulee sur `N` trajectoires d'alea. Un instant anterieur a la
# fin de la premiere iteration n'a pas de politique et est ignore. Le modele est construit une seule fois.
function simulate_job(folder, instance_name, method, time_list, N, settings)
    log = CSV.read(joinpath(folder, "log.csv"), DataFrame)

    model = build_model(instance_name, method, settings, run_optimizer())
    data = read_policy(folder, model)

    loaded = zeros(Int, length(model.nodes))
    rows = NamedTuple[]
    for time_limit in sort(time_list)
        k = count(<=(time_limit), log.time)    # iterations terminees a cet instant
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

# Comme `simulate_job`, mais on donne les iterations a simuler au lieu des instants : la politique de
# l'iteration k est le planning x^k et les coupes d'etape jusqu'a l'iteration k-1. Une iteration absente de
# l'entrainement (k < 1 ou k plus grand que le nombre d'iterations) est ignoree avec un avertissement.
# Ecrit `simulation_it<k>_<N>.csv` (cout de dispatch de chaque trajectoire) et `simulation_iterations_<N>.csv`
# (un resume par iteration, avec le temps `time` de fin de l'iteration), sans toucher aux fichiers de
# `simulate_job`.
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
