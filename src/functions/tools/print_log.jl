#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

const _RULE = "-------------------------------------------------------------------"

function print_banner(io)
    println(io, _RULE)
    println(io, "         MSUC.jl (c) Oscar Dowson and contributors, 2017-25")
    println(io, _RULE)
    return
end

function _unique_paths(model::PolicyGraph)
    return prod(length(node.noise_terms) for node in values(model.nodes); init = 1)
end

function _merge_tuple(x, y)
    if x == (-1, -1)
        return (y, y)
    elseif y < x[1]
        return (y, x[2])
    elseif y > x[2]
        return (x[1], y)
    else
        return x
    end
end

_constraint_key(F, S) = replace("$(F) in $(S)", "MathOptInterface" => "MOI")

function print_problem_statistics(
    io::IO,
    model::PolicyGraph,
    existing_cuts::Bool,
)
    constraint_types = Dict{String,Tuple{Int,Int}}()
    variables = (-1, -1)
    for (_, node) in model.nodes
        variables = _merge_tuple(variables, JuMP.num_variables(node.subproblem))
        for (F, S) in JuMP.list_of_constraint_types(node.subproblem)
            key = _constraint_key(F, S)
            num_con = get(constraint_types, key, (-1, -1))
            constraint_types[key] = _merge_tuple(
                num_con,
                JuMP.num_constraints(node.subproblem, F, S),
            )
        end
    end
    pad = maximum(length(k) for k in keys(constraint_types))
    println(io, "problem")
    println(io, "  nodes           : ", length(model.nodes))
    println(io, "  state variables : ", length(model.initial_root_state))
    paths = Printf.@sprintf("%1.5e", _unique_paths(model))
    println(io, "  scenarios       : ", paths)
    println(io, "  existing cuts   : ", existing_cuts)
    println(io, "subproblem structure")
    a, b = variables
    println(io, "  ", rpad("VariableRef", pad), " : [", a, ", ", b, "]")
    for k in sort!(collect(keys(constraint_types)))
        F, S = constraint_types[k]
        println(io, "  ", rpad(k, pad), " : [", F, ", ", S, "]")
    end
    return
end

function print_iteration_header(io)
    println(io, _RULE)
    println(
        io,
        " iteration     bound        time (s)     solves",
    )
    println(io, _RULE)
    return
end

print_value(x::Real) = lpad(Printf.@sprintf("%1.6e", x), 13)

print_value(x::Int) = Printf.@sprintf("%9d", x)

function print_iteration(io, log::Log)
    print(io, log.serious_numerical_issue ? "†" : " ")
    print(io, print_value(log.iteration))
    print(io, " ", print_value(log.bound))
    print(io, " ", print_value(log.time))
    print(io, " ", print_value(log.total_solves))
    log.first_stage_solved || print(io, "   MIP evite")
    println(io)
    return
end

function print_footer(io, training_results::TrainingResults)
    println(io, _RULE)
    println(io, "status         : ", training_results.status)
    println(io, "total time (s) :", print_value(training_results.log[end].time))
    println(io, "total solves   : ", training_results.log[end].total_solves)
    println(
        io,
        "best bound     : ",
        print_value(training_results.log[end].bound),
    )
    num_issues = sum(l -> l.serious_numerical_issue, training_results.log)
    println(io, "numeric issues : ", num_issues)
    skipped = count(l -> !l.first_stage_solved, training_results.log)
    skipped > 0 && println(io, "MIP evites     : ", skipped)
    println(io, _RULE)
    println(io)
    return
end

function _merge(x::Vector{Float64}, y::Vector{Float64})
    x[1] = min(x[1], y[1])
    x[2] = max(x[2], y[2])
    return
end

function _merge(x::CoefficientRanges, y::CoefficientRanges)
    _merge(x.matrix, y.matrix)
    _merge(x.objective, y.objective)
    _merge(x.bounds, y.bounds)
    _merge(x.rhs, y.rhs)
    return
end

function _stringify_bounds(bounds::Vector{Float64})
    lower = bounds[1] < Inf ? _print_value(bounds[1]) : "0e+00"
    upper = bounds[2] > -Inf ? _print_value(bounds[2]) : "0e+00"
    return string("[", lower, ", ", upper, "]")
end

function _print_numerical_stability_report(
    io::IO,
    ranges::CoefficientRanges,
    print::Bool,
    warn::Bool,
)
    warnings = Tuple{String,String}[]
    _print_coefficients(io, "matrix", ranges.matrix, print, warnings)
    _print_coefficients(io, "objective", ranges.objective, print, warnings)
    _print_coefficients(io, "bounds", ranges.bounds, print, warnings)
    _print_coefficients(io, "rhs", ranges.rhs, print, warnings)
    if warn && !isempty(warnings)
        println(io, "WARNING: numerical stability issues detected")
        for (name, sense) in warnings
            println(io, "  - $(name) range contains $(sense) coefficients")
        end
        println(
            io,
            "Very large or small absolute values of coefficients\n",
            "can cause numerical stability issues. Consider\n",
            "reformulating the model.",
        )
    end
    return
end

function _print_coefficients(
    io::IO,
    name::String,
    range,
    print::Bool,
    warnings::Vector{Tuple{String,String}},
)
    if print
        println(
            io,
            "  ",
            rpad(string(name, " range"), 17),
            _stringify_bounds(range),
        )
    end
    if range[1] < 1e-4
        push!(warnings, (name, "small"))
    end
    if range[2] > 1e7
        push!(warnings, (name, "large"))
    end
    return
end

_print_value(x::Real) = Printf.@sprintf("%1.0e", x)

function _update_range(range::Vector{Float64}, value::Real)
    if !(value ≈ 0.0)
        range[1] = min(range[1], abs(value))
        range[2] = max(range[2], abs(value))
    end
    return
end

function _update_range(range::Vector{Float64}, func::JuMP.GenericAffExpr)
    for coefficient in values(func.terms)
        _update_range(range, coefficient)
    end
    return
end

function _update_range(range::Vector{Float64}, func::MOI.LessThan)
    _update_range(range, func.upper)
    return
end

function _update_range(range::Vector{Float64}, func::MOI.GreaterThan)
    _update_range(range, func.lower)
    return
end

function _update_range(range::Vector{Float64}, func::MOI.EqualTo)
    _update_range(range, func.value)
    return
end

function _update_range(range::Vector{Float64}, func::MOI.Interval)
    _update_range(range, func.upper)
    _update_range(range, func.lower)
    return
end

_update_range(range::Vector{Float64}, x) = nothing

function _coefficient_ranges(model::JuMP.Model)
    ranges = CoefficientRanges()
    _update_range(ranges.objective, JuMP.objective_function(model))
    for var in JuMP.all_variables(model)
        if JuMP.has_lower_bound(var)
            _update_range(ranges.bounds, JuMP.lower_bound(var))
        end
        if JuMP.has_upper_bound(var)
            _update_range(ranges.bounds, JuMP.upper_bound(var))
        end
    end
    for (F, S) in JuMP.list_of_constraint_types(model)
        F == JuMP.VariableRef && continue
        for con in JuMP.all_constraints(model, F, S)
            con_obj = JuMP.constraint_object(con)
            _update_range(ranges.matrix, con_obj.func)
            _update_range(ranges.rhs, con_obj.set)
        end
    end
    return ranges
end

function numerical_stability_report(
    io::IO,
    model::PolicyGraph;
    by_node::Bool = false,
    print::Bool = true,
    warn::Bool = true,
)
    graph_ranges = CoefficientRanges()
    node_keys = sort_nodes(collect(keys(model.nodes)))
    for key in node_keys
        node = model[key]
        node_ranges = CoefficientRanges()
        for noise in node.noise_terms
            parameterize(node, noise.term)
            node_ranges_2 = _coefficient_ranges(node.subproblem)
            _merge(node_ranges, node_ranges_2)
        end
        if by_node
            print && println(io, "numerical stability report for node: ", key)
            _print_numerical_stability_report(io, node_ranges, print, warn)
        end
        _merge(graph_ranges, node_ranges)
    end
    if !by_node
        print && println(io, "numerical stability report")
        _print_numerical_stability_report(io, graph_ranges, print, warn)
    end
    return
end

function numerical_stability_report(model::PolicyGraph; kwargs...)
    return numerical_stability_report(stdout, model; kwargs...)
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
        print_iteration(stdout, options.log[end])
        options.last_log_iteration[] = length(options.log)
    end
    return
end
