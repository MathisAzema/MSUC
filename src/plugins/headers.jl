#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v. 2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

abstract type AbstractRiskMeasure end

function adjust_probability end

abstract type AbstractSamplingScheme end

function sample_scenario end

abstract type AbstractStoppingRule end

function stopping_rule_status end

function convergence_test(
    graph::PolicyGraph,
    log::Vector{Log},
    stopping_rules::Vector{AbstractStoppingRule},
)
    for stopping_rule in stopping_rules
        if convergence_test(graph, log, stopping_rule)
            return true, stopping_rule_status(stopping_rule)
        end
    end
    return false, :not_solved
end

abstract type AbstractBackwardSamplingScheme end

function sample_backward_noise_terms end

function sample_backward_noise_terms_with_state(
    sampler::AbstractBackwardSamplingScheme,
    node::Node,
    ::Dict{Symbol,Float64},
)
    return sample_backward_noise_terms(sampler, node)
end

abstract type AbstractDualityHandler end

function get_dual_solution end

function prepare_backward_pass(::Node, ::AbstractDualityHandler, ::Any)
    return () -> nothing
end

abstract type AbstractParallelScheme end

function master_loop end

function _simulate end

abstract type AbstractForwardPass end

function forward_pass end
