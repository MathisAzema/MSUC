#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
    simulate(model; schedule = nothing, samples = 1000, max_paths = 4096, seed = 1234)

Simulates the policy after `train`: solves the root problem with the final cuts, fixes the
resulting schedule x^k, then solves the dispatch stages along noise paths. With `schedule`, the root
problem is not solved and the policy of that schedule is simulated instead. If the number of paths
`S^T` is at most `max_paths`, all are enumerated and the expected cost is exact; otherwise `samples`
paths are drawn with `seed`.

Returns `(cost, commitment_cost, dispatch_cost, exhaustive, npaths, costs, weights, stderr, ci95)`.
`cost` is the expected policy cost (an upper bound on the optimum); `stderr` and
`ci95 = cost ± 1.96 stderr` are zero when `exhaustive`.
"""
function simulate(
    model::PolicyGraph;
    schedule::Union{Nothing,Schedule} = nothing,
    samples::Int = 1000,
    max_paths::Int = 4096,
    seed::Int = 1234,
)
    T = model.instance.TimeHorizon
    S = length(model[2].noise_terms)

    # Root problem: the policy's schedule and its commitment cost.
    state = initial_state(model.instance)
    if schedule === nothing
        root = model[model.first_node]
        solve_first_stage_problem(model, state)
        commitment_cost = stage_objective_value(root, root.stage_objective)
        schedule = extract_schedule(model)
    else
        commitment_cost = schedule_commitment_cost(model.instance, schedule)
    end
    set_schedule!(model, schedule)

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
        current = state
        c = 0.0
        for t in 1:T
            node = model[1+t]
            noise = node.noise_terms[path[t]].term
            res = solve_subproblem(model, node, current, noise)
            c += res.stage_objective
            current = res.state
        end
        push!(costs, c)
    end
    dispatch_cost = sum(w * c for (w, c) in zip(weights, costs))
    cost = commitment_cost + dispatch_cost
    stderr = exhaustive ? 0.0 : Statistics.std(costs) / sqrt(length(costs))
    return (
        cost = cost,
        commitment_cost = commitment_cost,
        dispatch_cost = dispatch_cost,
        exhaustive = exhaustive,
        npaths = length(paths),
        costs = costs,
        weights = weights,
        stderr = stderr,
        ci95 = (cost - 1.96 * stderr, cost + 1.96 * stderr),
    )
end
