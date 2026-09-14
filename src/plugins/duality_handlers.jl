#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors, Lea Kapelevich.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

function get_dual_solution(node::Node, ::Nothing)
    return JuMP.objective_value(node.subproblem), Dict{Symbol,Float64}()
end

# ========================= Continuous relaxation ============================ #

"""
    ContinuousConicDuality(optimizer = nothing)

Compute dual variables in the backward pass using conic duality, relaxing any
binary or integer restrictions as necessary.

## Arguments

 * `optimizer`: if  specified, MSUC.jl will call
   `JuMP.set_optimizer(subproblem, optimizer)` before solving problems on the
   backward pass. Use this option only if your default optimizer does not
   support returning a dual solution after the integrality has been relaxed.

## Example

Train a model using `ContinuousConicDuality` by passing it to the
`duality_handler` keyword argument of [`MSUC.train`](@ref):

```jldoctest
julia> import MSUC, HiGHS, Ipopt

julia> duality_handler = MSUC.ContinuousConicDuality(Ipopt.Optimizer)
MSUC.ContinuousConicDuality{DataType}(Ipopt.Optimizer)

julia> model = MSUC.LinearPolicyGraph(;
           stages = 2,
           lower_bound = 0.0,
           optimizer = HiGHS.Optimizer,
       ) do sp, t
           @variable(sp, x, MSUC.State, Int, initial_value = 0)
           @constraint(sp, x.out >= x.in + 0.5)
           @stageobjective(sp, x.out)
       end;

julia> MSUC.train(
           model;
           duality_handler = MSUC.ContinuousConicDuality(),
           print_level = 0,
       )
```
## Theory

Given the problem
```
min Cᵢ(x̄, u, w) + θᵢ
 st (x̄, x′, u) in Xᵢ(w) ∩ S
    x̄ - x == 0          [λ]
```
where `S ⊆ ℝ×ℤ`, we relax integrality and using conic duality to solve for `λ`
in the problem:
```
min Cᵢ(x̄, u, w) + θᵢ
 st (x̄, x′, u) in Xᵢ(w)
    x̄ - x == 0          [λ]
```
"""
struct ContinuousConicDuality{O} <: AbstractDualityHandler
    optimizer::O

    function ContinuousConicDuality(optimizer = nothing)
        return new{typeof(optimizer)}(optimizer)
    end
end

function get_dual_solution(node::Node, ::ContinuousConicDuality)
    if !_has_dual_solution(node)
        model = node.subproblem.ext[:MSUC_policy_graph]
        attempt_numerical_recovery(model, node; require_dual = true)
    end
    # Note: due to JuMP's dual convention, we need to flip the sign for
    # maximization problems.
    dual_sign = JuMP.objective_sense(node.subproblem) == MOI.MIN_SENSE ? 1 : -1
    λ = Dict{Symbol,Float64}(
        name => dual_sign * JuMP.dual(JuMP.FixRef(state.in)) for
        (name, state) in node.states
    )
    return objective_value(node.subproblem), λ
end

"""
    get_dual_solution_extended(node) -> (valeur, sigma, nu)

Multiplicateurs exterieurs du dual (42) au noeud courant :

  * `sigma` : multiplicateurs des lignes d'epigraphe (41g), un par coupe intervalle.
    C'est le sigma in Delta_k du papier, PAS le mu des lignes techniques (42b) --
    ce dernier n'est jamais materialise, la forme close (49) fournit directement le
    support local.
  * `nu` : multiplicateurs des equations de bilan par noeud (41f).
"""
function get_dual_solution_extended(node::Node)
    if !_has_dual_solution(node)
        model = node.subproblem.ext[:MSUC_policy_graph]
        attempt_numerical_recovery(model, node; require_dual = true)
    end
    σ = [JuMP.dual(cut.constraint_ref) for cut in node.interval_cuts]
    ν = JuMP.dual.(node.demand_constraint)
    return objective_value(node.subproblem), σ, ν
end

function _relax_integrality(node::Node, optimizer)
    if !node.has_integrality
        return () -> nothing
    elseif optimizer === nothing
        return JuMP.relax_integrality(node.subproblem)
    end
    undo_relax = JuMP.relax_integrality(node.subproblem)
    JuMP.set_optimizer(node.subproblem, optimizer)
    return () -> begin
        JuMP.set_optimizer(node.subproblem, node.optimizer)
        undo_relax()
        return
    end
end

function prepare_backward_pass(
    node::Node,
    handler::ContinuousConicDuality,
    ::Options,
)
    return _relax_integrality(node, handler.optimizer)
end

duality_log_key(::ContinuousConicDuality) = " "
