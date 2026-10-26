#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

function _simulate_once(
    model::PolicyGraph,
    variables::Vector{Symbol};
    sampling_scheme::AbstractSamplingScheme,
    custom_recorders::Dict{Symbol,Function},
    duality_handler::Union{Nothing,AbstractDualityHandler},
    skip_undefined_variables::Bool,
    incoming_state::Dict{Symbol,Float64},
)
    scenario_path, _ = sample_scenario(model, sampling_scheme)

    simulation = Dict{Symbol,Any}[]
    cumulative_value = 0.0

    for (depth, (node_index, noise)) in enumerate(scenario_path)
        node = model[node_index]
        subproblem_results = solve_subproblem(
            model,
            node,
            incoming_state,
            noise,
            scenario_path[1:depth];
            duality_handler = duality_handler,
        )
        cumulative_value += subproblem_results.stage_objective
        store = Dict{Symbol,Any}(
            :node_index => node_index,
            :noise_term => noise,
            :stage_objective => subproblem_results.stage_objective,
            :bellman_term =>
                subproblem_results.objective -
                subproblem_results.stage_objective,
        )
        for variable in variables
            if haskey(node.subproblem.obj_dict, variable)
                store[variable] = JuMP.value.(node.subproblem[variable])
            elseif skip_undefined_variables
                store[variable] = NaN
            else
                error(
                    "No variable named $(variable) exists in the subproblem.",
                    " If you want to simulate the value of a variable, make ",
                    "sure it is defined in _all_ subproblems, or pass ",
                    "`skip_undefined_variables=true` to `simulate`.",
                )
            end
        end
        for (sym, recorder) in custom_recorders
            store[sym] = recorder(node.subproblem)
        end
        push!(simulation, store)
        incoming_state = copy(subproblem_results.state)
    end
    return simulation
end

function _simulate(
    model::PolicyGraph,
    number_replications::Int,
    variables::Vector{Symbol};
    kwargs...,
)
    _initialize_solver(model; throw_error = false)
    return map(
        i -> _simulate_once(model, variables; kwargs...),
        1:number_replications,
    )
end

function _initial_state(model::PolicyGraph)
    return Dict(String(k) => v for (k, v) in model.initial_root_state)
end

function simulate(
    model::PolicyGraph,
    number_replications::Int = 1,
    variables::Vector{Symbol} = Symbol[];
    sampling_scheme::AbstractSamplingScheme = InSampleMonteCarlo(),
    custom_recorders = Dict{Symbol,Function}(),
    duality_handler::Union{Nothing,AbstractDualityHandler} = nothing,
    skip_undefined_variables::Bool = false,
    incoming_state::Dict{String,Float64} = _initial_state(model),
)
    return _simulate(
        model,
        number_replications,
        variables;
        sampling_scheme = sampling_scheme,
        custom_recorders = custom_recorders,
        duality_handler = duality_handler,
        skip_undefined_variables = skip_undefined_variables,
        incoming_state = Dict(Symbol(k) => v for (k, v) in incoming_state),
    )
end

function simulate_policy(
    model::PolicyGraph;
    max_paths::Int = 4096,
    samples::Int = 1000,
    seed::Int = 1234,
)
    T = model.instance.TimeHorizon
    S = length(model[2].noise_terms)

    root = model[1]
    JuMP.optimize!(root.subproblem)
    commitment_cost = stage_objective_value(root, root.stage_objective)
    if model.method == ExtendedFormulation
        selected = Tuple{Int,Int,Int}[
            k for (k, v) in root.states_gamma if JuMP.value(v) > 0.5
        ]
        reinstantiate_interval_cuts!(model, selected)
    end
    root_state = get_outgoing_state(root)

    exhaustive = float(S)^T <= max_paths
    paths, weights = if exhaustive
        ps = vec(collect(Iterators.product(ntuple(_ -> 1:S, T)...)))
        (
            [collect(p) for p in ps],
            [prod(model[1+t].noise_terms[p[t]].probability for t in 1:T) for p in ps],
        )
    else
        rng = Random.MersenneTwister(seed)
        probs = [n.probability for n in model[2].noise_terms]
        cum = cumsum(probs)
        draw() = [searchsortedfirst(cum, rand(rng)) for _ in 1:T]
        ([draw() for _ in 1:samples], fill(1 / samples, samples))
    end

    costs = Float64[]
    for path in paths
        state = copy(root_state)
        c = 0.0
        for t in 1:T
            node = model[1+t]
            noise = node.noise_terms[path[t]].term
            res = solve_subproblem(
                model,
                node,
                state,
                noise,
                Tuple{Int,Any}[(1 + t, noise)];
                duality_handler = nothing,
            )
            c += res.stage_objective
            state = res.state
        end
        push!(costs, c)
    end
    dispatch_cost = sum(w * c for (w, c) in zip(weights, costs))
    return (
        cost = commitment_cost + dispatch_cost,
        commitment_cost = commitment_cost,
        dispatch_cost = dispatch_cost,
        exhaustive = exhaustive,
        npaths = length(paths),
        costs = costs,
        weights = weights,
    )
end
