#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

macro _timeit_threadsafe(timer, label, block)
    code = quote
        if Threads.nthreads() == 1
            TimerOutputs.@timeit $timer $label $block
        else
            $block
        end
    end
    return esc(code)
end

function to_nodal_form(model::PolicyGraph, element)
    store = Dict{Int,typeof(element)}()
    for node_index in keys(model.nodes)
        store[node_index] = element
    end
    return store
end

function to_nodal_form(model::PolicyGraph, builder::Function)
    store = Dict{Int,Any}()
    for node_index in keys(model.nodes)
        store[node_index] = builder(node_index)
    end
    V = typeof(first(values(store)))
    for val in values(store)
        V = promote_type(V, typeof(val))
    end
    return Dict{Int,V}(key => val for (key, val) in store)
end

function to_nodal_form(model::PolicyGraph, dict::Dict{Int,V}) where {V}
    for key in keys(model.nodes)
        if !haskey(dict, key)
            error("Missing key: $(key).")
        end
    end
    return dict
end

struct Options
    initial_state::Dict{Symbol,Float64}
    sampling_scheme::AbstractSamplingScheme
    backward_sampling_scheme::AbstractBackwardSamplingScheme
    risk_measures::Dict{Int,AbstractRiskMeasure}
    Φ::Dict{Tuple{Int,Int},Float64}
    stopping_rules::Vector{AbstractStoppingRule}
    print_level::Int
    start_time::Float64
    log::Vector{Log}
    log_file_handle::Any
    log_frequency::Union{Int,Function}
    forward_pass::AbstractForwardPass
    duality_handler::AbstractDualityHandler
    forward_pass_callback::Any
    post_iteration_callback::Any
    last_log_iteration::Ref{Int}
    lock::ReentrantLock
    root_node_risk_measure::AbstractRiskMeasure
    function Options(
        model::PolicyGraph,
        initial_state::Dict{Symbol,Float64};
        sampling_scheme::AbstractSamplingScheme = InSampleMonteCarlo(),
        backward_sampling_scheme::AbstractBackwardSamplingScheme = CompleteSampler(),
        risk_measures = Expectation(),
        stopping_rules::Vector{AbstractStoppingRule} = AbstractStoppingRule[],
        print_level::Int = 0,
        start_time::Float64 = 0.0,
        log::Vector{Log} = Log[],
        log_file_handle = IOBuffer(),
        log_frequency::Union{Int,Function} = 1,
        forward_pass::AbstractForwardPass = DefaultForwardPass(),
        duality_handler::AbstractDualityHandler = ContinuousConicDuality(),
        forward_pass_callback = x -> nothing,
        post_iteration_callback = result -> nothing,
        root_node_risk_measure::AbstractRiskMeasure = Expectation(),
    )
        return new(
            initial_state,
            sampling_scheme,
            backward_sampling_scheme,
            to_nodal_form(model, risk_measures),
            build_Φ(model),
            stopping_rules,
            print_level,
            start_time,
            log,
            log_file_handle,
            log_frequency,
            forward_pass,
            duality_handler,
            forward_pass_callback,
            post_iteration_callback,
            Ref{Int}(0),
            ReentrantLock(),
            root_node_risk_measure,
        )
    end
end

function set_incoming_state(node::Node, state::Dict{Symbol,Float64})
    for (state_name, value) in state
        JuMP.fix(node.states[state_name].in, value)
    end
    return
end

function get_outgoing_state(node::Node)
    values = Dict{Symbol,Float64}()
    for (name, state) in node.states
        outgoing_value = JuMP.value(state.out)
        if JuMP.has_upper_bound(state.out)
            current_bound = JuMP.upper_bound(state.out)
            if current_bound < outgoing_value
                outgoing_value = current_bound
            end
        end
        if JuMP.has_lower_bound(state.out)
            current_bound = JuMP.lower_bound(state.out)
            if current_bound > outgoing_value
                outgoing_value = current_bound
            end
        end
        values[name] = outgoing_value
    end
    return values
end

function set_objective(node::Node)
    if !node.stage_objective_set
        JuMP.set_objective(
            node.subproblem,
            JuMP.objective_sense(node.subproblem),
            @expression(
                node.subproblem,
                node.stage_objective + bellman_term(node.bellman_function)
            )
        )
    end
    node.stage_objective_set = true
    return
end

_value(x::Real) = x
_value(x) = JuMP.value(x)

stage_objective_value(::Node, x::Union{GenericVariableRef,Real}) = _value(x)

function stage_objective_value(node::Node, stage_objective)
    theta = bellman_term(node.bellman_function)
    return JuMP.objective_value(node.subproblem) - JuMP.value(theta)
end

function write_subproblem_to_file(
    node::Node,
    filename::String;
    throw_error::Bool = false,
)
    model = MOI.FileFormats.Model(; filename = filename)
    MOI.copy_to(model, JuMP.backend(node.subproblem))
    MOI.write_to_file(model, filename)
    if throw_error
        error(
            "Unable to retrieve solution from node $(node.index).\n\n",
            "  Termination status : $(JuMP.termination_status(node.subproblem))\n",
            "  Primal status      : $(JuMP.primal_status(node.subproblem))\n",
            "  Dual status        : $(JuMP.dual_status(node.subproblem)).\n\n",
            "The current subproblem was written to `$(filename)`.\n\n",
            "There are two common causes of this error:\n",
            "  1) you have a mistake in your formulation, or you violated\n",
            "     the assumption of relatively complete recourse\n",
            "  2) the solver encountered numerical issues\n\n",
            "See https://odow.github.io/MSUC.jl/stable/tutorial/warnings/ for more information.",
        )
    end
    return
end

function parameterize(node::Node, noise)
    node.parameterize(noise)
    set_objective(node)
    return
end

function _has_primal_solution(node::Node)
    status = JuMP.primal_status(node.subproblem)
    return status in (JuMP.FEASIBLE_POINT, JuMP.NEARLY_FEASIBLE_POINT)
end

function _has_dual_solution(node::Node)
    status = JuMP.dual_status(node.subproblem)
    return status in (JuMP.FEASIBLE_POINT, JuMP.NEARLY_FEASIBLE_POINT)
end

function set_numerical_difficulty_callback(
    model::PolicyGraph,
    callback::Function,
)
    model.ext[:numerical_difficulty_callback] = callback
    return
end

function attempt_numerical_recovery(
    model::PolicyGraph,
    node::Node;
    require_dual::Bool = false,
)
    model.ext[:numerical_issue] = true
    callback = get(
        model.ext,
        :numerical_difficulty_callback,
        default_numerical_difficulty_callback,
    )
    callback(model, node; require_dual)
    missing_dual_solution = require_dual && !_has_dual_solution(node)
    if !_has_primal_solution(node) || missing_dual_solution
        write_subproblem_to_file(
            node,
            "subproblem_$(node.index).mof.json";
            throw_error = true,
        )
    end
    return
end

function default_numerical_difficulty_callback(
    model::PolicyGraph,
    node::Node;
    kwargs...,
)
    if JuMP.mode(node.subproblem) == JuMP.DIRECT
        @warn(
            "Unable to recover in direct mode! Remove `direct = true` when " *
            "creating the policy graph."
        )
        return
    end
    MOI.Utilities.reset_optimizer(node.subproblem)
    optimize!(node.subproblem)
    return
end

function _initialize_solver(node::Node; throw_error::Bool)
    if mode(node.subproblem) == DIRECT
        if throw_error
            error(
                "Cannot use asynchronous solver with optimizers in direct mode.",
            )
        end
    elseif MOI.Utilities.state(backend(node.subproblem)) == MOIU.NO_OPTIMIZER
        if node.optimizer === nothing
            error(
                """
          You must supply an optimizer for the policy graph, either by passing
          one to the `optimizer` keyword argument to `PolicyGraph`, or by
          using `JuMP.set_optimizer(model, optimizer)`.
          """,
            )
        end
        set_optimizer(node.subproblem, node.optimizer)
        set_silent(node.subproblem)
    end
    return
end

function _initialize_solver(model::PolicyGraph; throw_error::Bool)
    for (_, node) in model.nodes
        _initialize_solver(node; throw_error = throw_error)
    end
    return
end

function _uninitialize_solver(model::PolicyGraph; throw_error::Bool)
    for (_, node) in model.nodes
        if mode(node.subproblem) == DIRECT
            if throw_error
                error(
                    "Cannot use asynchronous solver with optimizers in direct mode.",
                )
            end
        elseif MOI.Utilities.state(backend(node.subproblem)) !=
               MOIU.NO_OPTIMIZER
            MOI.Utilities.drop_optimizer(node.subproblem)
        end
    end
    return
end

function solve_subproblem(
    model::PolicyGraph,
    node::Node,
    state::Dict{Symbol,Float64},
    noise,
    scenario_path::Vector{Tuple{Int,S}};
    duality_handler::Union{Nothing,AbstractDualityHandler},
) where {S}
    _initialize_solver(node; throw_error = false)
    set_incoming_state(node, state)
    parameterize(node, noise)
    start = time()
    JuMP.optimize!(node.subproblem)
    push!(node.computation_time, time() - start)
    lock(model.lock) do
        model.ext[:total_solves] = get(model.ext, :total_solves, 0) + 1
        return
    end
    if JuMP.primal_status(node.subproblem) == JuMP.MOI.INTERRUPTED
        throw(InterruptException())
    end
    if !_has_primal_solution(node)
        attempt_numerical_recovery(model, node)
    end
    state = get_outgoing_state(node)
    stage_objective = stage_objective_value(node, node.stage_objective)
    @_timeit_threadsafe model.timer_output "get_dual_solution" begin
        objective, dual_values = get_dual_solution(node, duality_handler)
    end
    return (
        state = state,
        duals = dual_values,
        objective = objective,
        stage_objective = stage_objective,
    )
end

function solve_subproblem_extended(
    model::PolicyGraph,
    node::Node,
    state::Dict{Symbol,Float64},
    noise,
    scenario_path::Vector{Tuple{Int,S}};
    duality_handler::Union{Nothing,AbstractDualityHandler},
) where {S}
    _initialize_solver(node; throw_error = false)
    set_incoming_state(node, state)
    parameterize(node, noise)
    start = time()
    JuMP.optimize!(node.subproblem)
    push!(node.computation_time, time() - start)
    lock(model.lock) do
        model.ext[:total_solves] = get(model.ext, :total_solves, 0) + 1
        return
    end
    if JuMP.primal_status(node.subproblem) == JuMP.MOI.INTERRUPTED
        throw(InterruptException())
    end
    if !_has_primal_solution(node)
        attempt_numerical_recovery(model, node)
    end
    state = get_outgoing_state(node)
    stage_objective = stage_objective_value(node, node.stage_objective)
    @_timeit_threadsafe model.timer_output "get_dual_solution" begin
        objective, σ, ν = get_dual_solution_extended(node)
    end
    return (
        state = state,
        σ = σ,
        ν = ν,
        objective = objective,
        stage_objective = stage_objective,
    )
end

function backward_pass(
    model::PolicyGraph,
    options::Options,
    scenario_path::Vector{Tuple{Int,NoiseType}},
    sampled_states::Vector{Dict{Symbol,Float64}},
    selected_intervals::Vector{Tuple{Int, Int, Int}},
    sampled_states_power::Vector{Vector{Float64}},
    sampled_states_zeta::Vector{Vector{Float64}},
    sampled_states_gamma::Vector{Dict{Tuple{Int, Int,Int},Float64}},
) where {NoiseType}
    cuts = Dict{Int,Vector{Any}}(index => Any[] for index in keys(model.nodes))
    for index in length(scenario_path):-1:1
        outgoing_state = sampled_states[index]
        outgoing_state_power = sampled_states_power[index]
        outgoing_state_zeta = sampled_states_zeta[index]
        outgoing_state_gamma = sampled_states_gamma[index]
        node_index, _ = scenario_path[index]
        node = model[node_index]
        if node.next_node === nothing
            continue
        end
        if model.method == SDDPFormulation
            items = BackwardPassItems(Noise)
            solve_all_children(
                model,
                node,
                items,
                outgoing_state,
                options.backward_sampling_scheme,
                scenario_path[1:index],
                options.duality_handler,
                options,
            )
            new_cuts = refine_bellman_function(
                model,
                node,
                node.bellman_function,
                options.risk_measures[node_index],
                outgoing_state,
                items.duals,
                items.supports,
                items.probability,
                items.objectives,
            )
            push!(cuts[node_index], new_cuts)
        elseif model.method == ExtendedFormulation
            items = BackwardPassItems_extended(Noise)
            solve_all_children_extended(
                model,
                node,
                items,
                outgoing_state,
                options.backward_sampling_scheme,
                scenario_path[1:index],
                options.duality_handler,
                options,
            )

            refine_bellman_function_extended(
                model,
                node,
                selected_intervals,
                outgoing_state_power,
                outgoing_state_gamma,
                outgoing_state_zeta,
                items.σ,
                items.ν,
                items.probability,
                items.objectives,
            )

        else
            error("Unknown method: $(model.method).")
        end
    end
    return cuts
end

struct BackwardPassItems{U}
    cached_solutions::Dict{Tuple{Int,Any},Int}
    duals::Vector{Dict{Symbol,Float64}}
    supports::Vector{U}
    nodes::Vector{Int}
    probability::Vector{Float64}
    objectives::Vector{Float64}
    function BackwardPassItems(U)
        return new{U}(
            Dict{Tuple{Int,Any},Int}(),
            Dict{Symbol,Float64}[],
            U[],
            Int[],
            Float64[],
            Float64[],
        )
    end
end

struct BackwardPassItems_extended{U}
    cached_solutions::Dict{Tuple{Int,Any},Int}
    σ::Vector{Vector{Float64}}
    ν::Vector{Vector{Float64}}
    supports::Vector{U}
    nodes::Vector{Int}
    probability::Vector{Float64}
    objectives::Vector{Float64}
    function BackwardPassItems_extended(U)
        return new{U}(
            Dict{Tuple{Int,Any},Int}(),
            Vector{Float64}[],
            Vector{Float64}[],
            U[],
            Int[],
            Float64[],
            Float64[],
        )
    end
end

function solve_all_children(
    model::PolicyGraph,
    node::Node,
    items::BackwardPassItems,
    outgoing_state::Dict{Symbol,Float64},
    backward_sampling_scheme::AbstractBackwardSamplingScheme,
    scenario_path,
    duality_handler::Union{Nothing,AbstractDualityHandler},
    options,
)
    length_scenario_path = length(scenario_path)
    child_index = node.next_node::Int
    child_node = model[child_index]
    lock(child_node.lock)
    try
        @_timeit_threadsafe model.timer_output "prepare_backward_pass" begin
            restore_duality = prepare_backward_pass(
                child_node,
                options.duality_handler,
                options,
            )
        end
        for noise in sample_backward_noise_terms_with_state(
            backward_sampling_scheme,
            child_node,
            outgoing_state,
        )
            if length(scenario_path) == length_scenario_path
                push!(scenario_path, (child_index, noise.term))
            else
                scenario_path[end] = (child_index, noise.term)
            end
            if haskey(items.cached_solutions, (child_index, noise.term))
                sol_index = items.cached_solutions[(child_index, noise.term)]
                push!(items.duals, items.duals[sol_index])
                push!(items.supports, items.supports[sol_index])
                push!(items.nodes, child_node.index)
                push!(items.probability, items.probability[sol_index])
                push!(items.objectives, items.objectives[sol_index])
            else
                @_timeit_threadsafe model.timer_output "solve_subproblem" begin
                    subproblem_results = solve_subproblem(
                        model,
                        child_node,
                        outgoing_state,
                        noise.term,
                        scenario_path;
                        duality_handler = duality_handler,
                    )
                end
                push!(items.duals, subproblem_results.duals)
                push!(items.supports, noise)
                push!(items.nodes, child_node.index)
                push!(
                    items.probability,
                    noise.probability,
                )
                push!(items.objectives, subproblem_results.objective)
                items.cached_solutions[(child_index, noise.term)] =
                    length(items.duals)
            end
        end
        @_timeit_threadsafe model.timer_output "prepare_backward_pass" begin
            restore_duality()
        end
    finally
        unlock(child_node.lock)
    end
    if length(scenario_path) == length_scenario_path
    else
        pop!(scenario_path)
    end
    return
end

function solve_all_children_extended(
    model::PolicyGraph,
    node::Node,
    items::BackwardPassItems_extended,
    outgoing_state::Dict{Symbol,Float64},
    backward_sampling_scheme::AbstractBackwardSamplingScheme,
    scenario_path,
    duality_handler::Union{Nothing,AbstractDualityHandler},
    options,
)
    length_scenario_path = length(scenario_path)
    child_index = node.next_node::Int
    child_node = model[child_index]
    lock(child_node.lock)
    try
        @_timeit_threadsafe model.timer_output "prepare_backward_pass" begin
            restore_duality = prepare_backward_pass(
                child_node,
                options.duality_handler,
                options,
            )
        end
        for noise in sample_backward_noise_terms_with_state(
            backward_sampling_scheme,
            child_node,
            outgoing_state,
        )
            if length(scenario_path) == length_scenario_path
                push!(scenario_path, (child_index, noise.term))
            else
                scenario_path[end] = (child_index, noise.term)
            end
            if haskey(items.cached_solutions, (child_index, noise.term))
                sol_index = items.cached_solutions[(child_index, noise.term)]
                push!(items.σ, items.σ[sol_index])
                push!(items.ν, items.ν[sol_index])
                push!(items.supports, items.supports[sol_index])
                push!(items.nodes, child_node.index)
                push!(items.probability, items.probability[sol_index])
                push!(items.objectives, items.objectives[sol_index])
            else
                @_timeit_threadsafe model.timer_output "solve_subproblem" begin
                    subproblem_results = solve_subproblem_extended(
                        model,
                        child_node,
                        outgoing_state,
                        noise.term,
                        scenario_path;
                        duality_handler = duality_handler,
                    )
                end

                push!(items.supports, noise)
                push!(items.nodes, child_node.index)
                push!(items.σ, subproblem_results.σ)
                push!(items.ν, subproblem_results.ν)
                push!(
                    items.probability,
                    noise.probability,
                )
                push!(items.objectives, subproblem_results.objective)
                items.cached_solutions[(child_index, noise.term)] =
                    length(items.σ)
            end
        end
        @_timeit_threadsafe model.timer_output "prepare_backward_pass" begin
            restore_duality()
        end
    finally
        unlock(child_node.lock)
    end
    if length(scenario_path) == length_scenario_path
    else
        pop!(scenario_path)
    end
    return
end

function calculate_bound(
    model::PolicyGraph,
    root_state::Dict{Symbol,Float64} = model.initial_root_state;
    risk_measure::AbstractRiskMeasure = Expectation(),
)
    noise_supports = Any[]
    probabilities = Float64[]
    objectives = Float64[]
    node = model[model.first_node]
    lock(node.lock)
    try
        for noise in node.noise_terms
            subproblem_results = solve_subproblem(
                model,
                node,
                root_state,
                noise.term,
                Tuple{Int,Any}[(node.index, noise.term)];
                duality_handler = nothing,
            )
            push!(objectives, subproblem_results.objective)
            push!(probabilities, noise.probability)
            push!(noise_supports, noise.term)
        end
    finally
        unlock(node.lock)
    end
    risk_adjusted_probability = similar(probabilities)
    offset = adjust_probability(
        risk_measure,
        risk_adjusted_probability,
        probabilities,
        noise_supports,
        objectives,
        model.objective_sense == MOI.MIN_SENSE,
    )
    return sum(
        obj * prob for (obj, prob) in zip(objectives, risk_adjusted_probability)
    ) + offset
end

struct IterationResult
    pid::Int64
    bound::Float64
    cumulative_value::Float64
    has_converged::Bool
    status::Symbol
    numerical_issue::Bool
end

function iteration(model::PolicyGraph, options::Options)
    model.ext[:numerical_issue] = false
    @_timeit_threadsafe model.timer_output "forward_pass" begin
        forward_trajectory = forward_pass(model, options, options.forward_pass)
        options.forward_pass_callback(forward_trajectory)
    end
    @_timeit_threadsafe model.timer_output "backward_pass" begin
        backward_pass(
            model,
            options,
            forward_trajectory.scenario_path,
            forward_trajectory.sampled_states,
            forward_trajectory.selected_intervals,
            forward_trajectory.sampled_states_power,
            forward_trajectory.sampled_states_zeta,
            forward_trajectory.sampled_states_gamma,
        )
    end
    is_min = model.objective_sense == MOI.MIN_SENSE
    best = get(model.ext, :best_bound, is_min ? -Inf : Inf)
    bound = is_min ? max(best, forward_trajectory.lower_bound) :
            min(best, forward_trajectory.lower_bound)
    model.ext[:best_bound] = bound
    lock(options.lock)
    try
        push!(
            options.log,
            Log(
                length(options.log) + 1,
                bound,
                forward_trajectory.cumulative_value,
                time() - options.start_time,
                max(Threads.threadid(), Distributed.myid()),
                lock(() -> model.ext[:total_solves], model.lock),
                duality_log_key(options.duality_handler),
                lock(() -> model.ext[:numerical_issue], model.lock),
            ),
        )
        has_converged, status =
            convergence_test(model, options.log, options.stopping_rules)
        return IterationResult(
            max(Threads.threadid(), Distributed.myid()),
            bound,
            forward_trajectory.cumulative_value,
            has_converged,
            status,
            lock(() -> model.ext[:numerical_issue], model.lock),
        )
    finally
        unlock(options.lock)
    end
end

function termination_status(model::PolicyGraph)
    if model.most_recent_training_results === nothing
        return :model_not_solved
    end
    return model.most_recent_training_results.status
end

function train(
    model::PolicyGraph;
    iteration_limit::Union{Int,Nothing} = nothing,
    time_limit::Union{Real,Nothing} = nothing,
    print_level::Int = 1,
    log_file::String = "MSUC.log",
    log_frequency::Union{Nothing,Int} = nothing,
    log_every_seconds::Float64 = log_frequency === nothing ? -1.0 : 0.0,
    log_every_iteration::Bool = false,
    run_numerical_stability_report::Bool = true,
    stopping_rules = AbstractStoppingRule[],
    risk_measure = Expectation(),
    root_node_risk_measure::AbstractRiskMeasure = Expectation(),
    sampling_scheme = InSampleMonteCarlo(),
    cut_type = SINGLE_CUT,
    backward_sampling_scheme::AbstractBackwardSamplingScheme = CompleteSampler(),
    parallel_scheme::AbstractParallelScheme = Serial(),
    forward_pass::AbstractForwardPass = DefaultForwardPass(),
    forward_pass_resampling_probability::Union{Nothing,Float64} = nothing,
    add_to_existing_cuts::Bool = false,
    duality_handler::AbstractDualityHandler = ContinuousConicDuality(),
    forward_pass_callback::Function = (x) -> nothing,
    post_iteration_callback = result -> nothing,
)
    log_frequency = something(log_frequency, 1)
    if log_frequency <= 0
        msg = "`log_frequency` must be at least `1`. Got $log_frequency."
        throw(ArgumentError(msg))
    end
    if log_every_iteration
        log_frequency = 1
        log_every_seconds = 0.0
    end
    function log_frequency_f(log::Vector{Log})
        if mod(length(log), log_frequency) != 0
            return false
        end
        last = options.last_log_iteration[]
        if last == 0
            return true
        elseif last == length(log)
            return false
        end
        seconds = log_every_seconds
        if log_every_seconds < 0.0
            if log[end].time <= 10
                seconds = 1.0
            elseif log[end].time <= 120
                seconds = 5.0
            else
                seconds = 30.0
            end
        end
        return log[end].time - log[last].time >= seconds
    end

    if !add_to_existing_cuts && model.most_recent_training_results !== nothing
        @warn("""
        Re-training a model with existing cuts!

        Are you sure you want to do this? The output from this training may be
        misleading because the policy is already partially trained.

        If you meant to train a new policy with different settings, you must
        build a new model.

        If you meant to refine a previously trained policy, turn off this
        warning by passing `add_to_existing_cuts = true` as a keyword argument
        to `MSUC.train`.

        In a future release, this warning may turn into an error.
        """)
    end
    if forward_pass_resampling_probability !== nothing
        forward_pass = RiskAdjustedForwardPass(;
            forward_pass = forward_pass,
            risk_measure = risk_measure,
            resampling_probability = forward_pass_resampling_probability,
        )
    end
    TimerOutputs.reset_timer!(model.timer_output)
    log_file_handle = open(log_file, "a")
    log = Log[]

    if print_level > 0
        print_helper(print_banner, log_file_handle)
        print_helper(
            print_problem_statistics,
            log_file_handle,
            model,
            model.most_recent_training_results !== nothing,
            parallel_scheme,
            risk_measure,
            sampling_scheme,
        )
    end
    if run_numerical_stability_report
        report = sprint(
            io -> numerical_stability_report(
                io,
                model;
                print = print_level > 0,
            ),
        )
        print_helper(print, log_file_handle, report)
    end
    if print_level > 0
        print_helper(print_iteration_header, log_file_handle)
    end
    stopping_rules = convert(Vector{AbstractStoppingRule}, stopping_rules)
    if iteration_limit !== nothing
        push!(stopping_rules, IterationLimit(iteration_limit))
    end
    if time_limit !== nothing
        push!(stopping_rules, TimeLimit(time_limit))
    end
    if isempty(stopping_rules)
        push!(stopping_rules, SimulationStoppingRule())
    end
    for (_, node) in model.nodes
        node.bellman_function.cut_type = cut_type
    end
    options = Options(
        model,
        model.initial_root_state;
        sampling_scheme,
        backward_sampling_scheme,
        risk_measures = risk_measure,
        stopping_rules,
        print_level,
        start_time = time(),
        log,
        log_file_handle,
        log_frequency = log_frequency_f,
        forward_pass,
        duality_handler,
        forward_pass_callback,
        post_iteration_callback,
        root_node_risk_measure,
    )
    status = :not_solved
    try
        status = master_loop(parallel_scheme, model, options)
    catch ex
        if ex isa CompositeException
            ex = last(ex.exceptions)
        end
        if ex isa TaskFailedException
            ex = ex.task.exception
        end
        if ex isa InterruptException
            status = :interrupted
            interrupt(parallel_scheme)
        else
            close(log_file_handle)
            throw(ex)
        end
    finally
    end
    training_results = TrainingResults(status, log)
    model.most_recent_training_results = training_results
    if print_level > 0
        log_iteration(options; force_if_needed = true)
        print_helper(print_footer, log_file_handle, training_results)
        if print_level > 1
            print_helper(
                TimerOutputs.print_timer,
                log_file_handle,
                model.timer_output,
            )
            print_helper(println, log_file_handle)
        end
    end
    close(log_file_handle)
    return
end

function _simulate(
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
        lock(node.lock)
        try
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
        finally
            unlock(node.lock)
        end
    end
    return simulation
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
    parallel_scheme::AbstractParallelScheme = Serial(),
    incoming_state::Dict{String,Float64} = _initial_state(model),
)
    return _simulate(
        model,
        parallel_scheme,
        number_replications,
        variables;
        sampling_scheme = sampling_scheme,
        custom_recorders = custom_recorders,
        duality_handler = duality_handler,
        skip_undefined_variables = skip_undefined_variables,
        incoming_state = Dict(Symbol(k) => v for (k, v) in incoming_state),
    )
end
