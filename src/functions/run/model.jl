# Methods: (subproblem builder, formulation, `interval_rule` for `train`). "bin3_gamma" is 3-bin
# with the gamma variables added at the root; "interval_<rule>" is the interval formulation with the
# given linearization rule (`INTERVAL_RULES`).
const METHODS = Dict(
    "interval_dual" => (subproblem_builder_UC_extended, ExtendedFormulation, :dual),
    "interval_projection" => (subproblem_builder_UC_extended, ExtendedFormulation, :projection),
    "interval_minimum" => (subproblem_builder_UC_extended, ExtendedFormulation, :minimum),
    "interval_maximum" => (subproblem_builder_UC_extended, ExtendedFormulation, :maximum),
    "bin3" => (subproblem_builder_UC_def, SDDPFormulation, :dual),
    "bin3_gamma" => (subproblem_builder_UC_def_gamma, SDDPFormulation, :dual),
)

# Seed of all runs, training and simulation.
const SEED = 1

# The IEEE instance is a folder in `Data/IEEE`, not an `.nc4` file in `Data/T-Ramp`.
const IEEE_INSTANCE = "118_syst"

is_ieee(instance_name) =
    isdir(joinpath(@__DIR__, "..", "..", "..", "Data", "IEEE", instance_name))

# `settings`: (T, S, sigma, mip_gap, passes_per_mip). All parameters appear in the folder name so
# that runs never overwrite each other.
function results_folder(instance_name, method, time_max, settings)
    gap = settings.mip_gap === nothing ? "default" : string(settings.mip_gap)
    tag = string(
        method, "_T", settings.T, "_S", settings.S, "_sigma", settings.sigma,
        "_passes", settings.passes_per_mip, "_gap", gap,
    )
    root = is_ieee(instance_name) ? "results_ieee" : "results_sms"
    return "$(root)/$(tag)/$(instance_name)/time_$(time_max)"
end

instance_name(n_units, id) = "$(n_units)_0_$(id)_w"

# All runs of a campaign, as `(instance, method, time limit, settings)`. (size, id) combinations
# without a file in `Data/T-Ramp` are skipped with a warning; it is an error if none exists.
function run_combos(
    size_list,
    id_list,
    method_list,
    time_max_list;
    T,
    S_list,
    sigma_list,
    passes_per_mip_list,
    mip_gap_list,
)
    wanted = [instance_name(n_units, id) for n_units in size_list for id in id_list]
    instances = filter(name -> isfile("Data/T-Ramp/$(name).nc4"), wanted)
    absent = setdiff(wanted, instances)
    isempty(absent) || @warn "instances absentes de Data/T-Ramp, ignorees" absent
    isempty(instances) &&
        error("aucune instance de Data/T-Ramp pour size_list = $(size_list) et id_list = $(id_list)")
    return campaign(
        instances, method_list, time_max_list;
        T, S_list, sigma_list, passes_per_mip_list, mip_gap_list,
    )
end

ieee_combos(method_list, time_max_list; kwargs...) =
    campaign([IEEE_INSTANCE], method_list, time_max_list; kwargs...)

function campaign(
    instances,
    method_list,
    time_max_list;
    T,
    S_list,
    sigma_list,
    passes_per_mip_list,
    mip_gap_list,
)
    return [
        (instance, method, time_max, (; T, S, sigma, mip_gap, passes_per_mip)) for
        instance in instances for method in method_list for time_max in time_max_list for
        S in S_list for sigma in sigma_list for passes_per_mip in passes_per_mip_list for
        mip_gap in mip_gap_list
    ]
end

# Optimizer factory registered in each process (`set_run_optimizer!`). Jobs read it here instead of
# receiving it from `pmap`, which would ship the master's Gurobi environment to the workers.
const RUN_OPTIMIZER = Ref{Any}(nothing)

set_run_optimizer!(factory) = (RUN_OPTIMIZER[] = factory; nothing)

function run_optimizer()
    RUN_OPTIMIZER[] === nothing && error(
        "aucun optimiseur enregistre : appeler `MSUC.set_run_optimizer!(() -> Gurobi.Optimizer(env))` " *
        "sur chaque processus (voir runSMS.jl)",
    )
    return RUN_OPTIMIZER[]
end

function build_model(instance_name, method, settings, optimizer)
    builder, formulation, _ = METHODS[method]
    instance = if is_ieee(instance_name)
        parse_IEEE(instance_name, optimizer, settings.T; sigma = settings.sigma, S = settings.S)
    else
        parse_nc4(
            "Data/T-Ramp/$(instance_name).nc4",
            optimizer,
            settings.T;
            sigma = settings.sigma,
            S = settings.S,
        )
    end
    # force = 1: the noise amplitude is entirely given by `sigma`.
    return PolicyGraph(
        builder,
        instance,
        1.0,
        Graph(settings.T + 1),
        formulation;
        lower_bound = 0.0,
        optimizer = optimizer,
    )
end
