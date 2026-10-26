#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

struct Options
    initial_state::Dict{Symbol,Float64}
    sampling_scheme::AbstractSamplingScheme
    backward_sampling_scheme::AbstractBackwardSamplingScheme
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
    function Options(
        model::PolicyGraph,
        initial_state::Dict{Symbol,Float64};
        sampling_scheme::AbstractSamplingScheme = InSampleMonteCarlo(),
        backward_sampling_scheme::AbstractBackwardSamplingScheme = CompleteSampler(),
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
    )
        return new(
            initial_state,
            sampling_scheme,
            backward_sampling_scheme,
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
    model.ext[:total_solves] = get(model.ext, :total_solves, 0) + 1
    if JuMP.primal_status(node.subproblem) == JuMP.MOI.INTERRUPTED
        throw(InterruptException())
    end
    if !_has_primal_solution(node)
        attempt_numerical_recovery(model, node)
    end
    state = get_outgoing_state(node)
    stage_objective = stage_objective_value(node, node.stage_objective)
    TimerOutputs.@timeit model.timer_output "get_dual_solution" begin
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
    model.ext[:total_solves] = get(model.ext, :total_solves, 0) + 1
    if JuMP.primal_status(node.subproblem) == JuMP.MOI.INTERRUPTED
        throw(InterruptException())
    end
    if !_has_primal_solution(node)
        attempt_numerical_recovery(model, node)
    end
    state = get_outgoing_state(node)
    stage_objective = stage_objective_value(node, node.stage_objective)
    TimerOutputs.@timeit model.timer_output "get_dual_solution" begin
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

function calculate_bound(
    model::PolicyGraph,
    root_state::Dict{Symbol,Float64} = model.initial_root_state;
)
    probabilities = Float64[]
    objectives = Float64[]
    node = model[model.first_node]
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
    end
    return sum(obj * prob for (obj, prob) in zip(objectives, probabilities))
end

struct IterationResult
    bound::Float64
    cumulative_value::Float64
    has_converged::Bool
    status::Symbol
    numerical_issue::Bool
end

function iteration(model::PolicyGraph, options::Options)
    model.ext[:numerical_issue] = false
    TimerOutputs.@timeit model.timer_output "forward_pass" begin
        forward_trajectory = forward_pass(model, options, options.forward_pass)
        options.forward_pass_callback(forward_trajectory)
    end
    TimerOutputs.@timeit model.timer_output "backward_pass" begin
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
    push!(
        options.log,
        Log(
            length(options.log) + 1,
            bound,
            forward_trajectory.cumulative_value,
            time() - options.start_time,
            model.ext[:total_solves],
            duality_log_key(options.duality_handler),
            model.ext[:numerical_issue],
        ),
    )
    has_converged, status =
        convergence_test(model, options.log, options.stopping_rules)
    return IterationResult(
        bound,
        forward_trajectory.cumulative_value,
        has_converged,
        status,
        model.ext[:numerical_issue],
    )
end

function _should_log(options, log_frequency::Int)
    return options.print_level > 0 &&
           mod(length(options.log), log_frequency) == 0
end

function _should_log(options, log_frequency::Function)
    return options.print_level > 0 && log_frequency(options.log)
end

function log_iteration(options; force_if_needed::Bool = false)
    force_if_needed &= options.last_log_iteration[] != length(options.log)
    if force_if_needed || _should_log(options, options.log_frequency)
        print_helper(print_iteration, options.log_file_handle, options.log[end])
        flush(options.log_file_handle)
        options.last_log_iteration[] = length(options.log)
    end
    return
end

function master_loop(model::PolicyGraph, options::Options)
    _initialize_solver(model; throw_error = false)
    while true
        result = iteration(model, options)
        options.post_iteration_callback(result)
        log_iteration(options)
        if result.has_converged
            return result.status
        end
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
    sampling_scheme = InSampleMonteCarlo(),
    cut_type = SINGLE_CUT,
    backward_sampling_scheme::AbstractBackwardSamplingScheme = CompleteSampler(),
    forward_pass::AbstractForwardPass = DefaultForwardPass(),
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
    )
    status = :not_solved
    try
        status = master_loop(model, options)
    catch ex
        if ex isa InterruptException
            status = :interrupted
        else
            close(log_file_handle)
            throw(ex)
        end
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
