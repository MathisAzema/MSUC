#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

struct Options
    initial_state::NamedTuple
    Φ::Dict{Tuple{Int,Int},Float64}
    iteration_limit::Union{Int,Nothing}
    time_limit::Union{Float64,Nothing}
    passes_per_mip::Int
    print_level::Int
    start_time::Float64
    log::Vector{Log}
    log_frequency::Union{Int,Function}
    forward_pass_callback::Any
    post_iteration_callback::Any
    last_log_iteration::Ref{Int}
    function Options(
        model::PolicyGraph,
        initial_state::NamedTuple;
        iteration_limit::Union{Int,Nothing} = nothing,
        time_limit::Union{Float64,Nothing} = nothing,
        passes_per_mip::Int = 1,
        print_level::Int = 0,
        start_time::Float64 = 0.0,
        log::Vector{Log} = Log[],
        log_frequency::Union{Int,Function} = 1,
        forward_pass_callback = x -> nothing,
        post_iteration_callback = result -> nothing,
    )
        return new(
            initial_state,
            build_Φ(model),
            iteration_limit,
            time_limit,
            passes_per_mip,
            print_level,
            start_time,
            log,
            log_frequency,
            forward_pass_callback,
            post_iteration_callback,
            Ref{Int}(0),
        )
    end
end

struct IterationResult
    bound::Float64
    status::Symbol
    numerical_issue::Bool
end
