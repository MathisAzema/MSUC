#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v. 2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

# --- Valeur future de chaque noeud : variable theta et coupe initiale -----------------------

bellman_term(bellman_function::BellmanFunction) = bellman_function.theta

# Coupe nulle : theta >= 0, point de depart des deux formulations.
function _trivial_cut(model::PolicyGraph, row)
    instance = model.instance
    nz = zeta_dimension(instance.arma)
    if model.method == ExtendedFormulation
        τ = Dict{Tuple{Int64,Int64,Int64},Float64}()
        β = Dict{Tuple{Int64,Int64,Int64},Float64}()
        for unit in instance.Thermalunits, (a, b) in unit.intervals
            τ[(unit.name, a, b)] = 0.0
            β[(unit.name, a, b)] = 0.0
        end
        return IntervalCut(0.0, τ, β, zeros(nz), row)
    end
    return BinCut(
        0.0,
        zeros(commitment_dimension(instance)),
        zeros(instance.N),
        zeros(nz),
        row,
    )
end

function initialize_bellman_function(
    factory::InstanceFactory{BellmanFunction},
    model::PolicyGraph,
    node::Node,
)
    lower_bound, upper_bound = -Inf, Inf
    if length(factory.args) > 0
        error(
            "Positional arguments $(factory.args) ignored in BellmanFunction.",
        )
    end
    for (kw, value) in factory.kwargs
        if kw == :lower_bound
            lower_bound = value
        elseif kw == :upper_bound
            upper_bound = value
        else
            error(
                "Keyword $(kw) not recognised as argument to BellmanFunction.",
            )
        end
    end
    if lower_bound == -Inf && upper_bound == Inf
        error("You must specify a finite bound on the cost-to-go term.")
    end
    if node.next_node === nothing
        lower_bound = upper_bound = 0.0
    end
    Θᴳ = @variable(node.subproblem)
    cstr = @constraint(node.subproblem, Θᴳ >= 0.0)
    push!(node.cuts, _trivial_cut(model, cstr))
    return BellmanFunction(Θᴳ)
end
