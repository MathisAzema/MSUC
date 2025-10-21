#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v. 2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

module TestStoppingRules

using Random
using MSUC
using Test
import HiGHS

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

function test_TimeLimit()
    graph = MSUC.PolicyGraph(
        MSUC.LinearGraph(2);
        lower_bound = 0.0,
        direct_mode = false,
    ) do node, stage
        @variable(node, x, MSUC.State, initial_value = 0)
    end
    rule = MSUC.TimeLimit(0.5)
    @test MSUC.stopping_rule_status(rule) == :time_limit
    @test MSUC.convergence_test(
        graph,
        [MSUC.Log(1, 0.0, 0.0, 1.0, 1, 1, " ", false)],
        rule,
    )
    @test !MSUC.convergence_test(
        graph,
        [MSUC.Log(1, 0.0, 0.0, 0.1, 1, 1, " ", false)],
        rule,
    )
    return
end

function test_IterationLimit()
    graph = MSUC.PolicyGraph(
        MSUC.LinearGraph(2);
        lower_bound = 0.0,
        direct_mode = false,
    ) do node, stage
        @variable(node, x, MSUC.State, initial_value = 0)
    end
    rule = MSUC.IterationLimit(2)
    @test MSUC.stopping_rule_status(rule) == :iteration_limit
    @test MSUC.convergence_test(
        graph,
        [
            MSUC.Log(1, 0.0, 0.0, 1.0, 1, 1, " ", false),
            MSUC.Log(2, 0.0, 0.0, 1.0, 1, 1, " ", false),
        ],
        rule,
    )
    @test !MSUC.convergence_test(
        graph,
        [MSUC.Log(1, 0.0, 0.0, 0.1, 1, 1, " ", false)],
        rule,
    )
    return
end

function test_Statistical()
    model = MSUC.PolicyGraph(
        MSUC.LinearGraph(2);
        bellman_function = MSUC.BellmanFunction(; lower_bound = 0.0),
        optimizer = HiGHS.Optimizer,
        sense = :Min,
    ) do node, stage
        @variable(node, x >= 0, MSUC.State, initial_value = 0.0)
        MSUC.parameterize(node, stage * [1, 3], [0.5, 0.5]) do ω
            return JuMP.set_lower_bound(x.out, ω)
        end
        @stageobjective(node, x.out)
    end
    MSUC.train(model; iteration_limit = 1, print_level = 0)
    rule = MSUC.Statistical(; num_replications = 20)
    @test MSUC.stopping_rule_status(rule) == :statistical
    Random.seed!(123)
    @test MSUC.convergence_test(
        model,
        [MSUC.Log(1, 6.0, 9.0, 1.0, 1, 1, " ", false)],
        rule,
    )
    @test !MSUC.convergence_test(
        model,
        [MSUC.Log(1, 0.0, 9.0, 1.0, 1, 1, " ", false)],
        rule,
    )
    @test MSUC.convergence_test(
        model,
        [MSUC.Log(1, 12.0, 9.0, 1.0, 1, 1, " ", false)],
        rule,
    )

    model = MSUC.PolicyGraph(
        MSUC.LinearGraph(2);
        bellman_function = MSUC.BellmanFunction(; upper_bound = 6.0),
        optimizer = HiGHS.Optimizer,
        sense = :Max,
    ) do node, stage
        @variable(node, x >= 0, MSUC.State, initial_value = 0.0)
        MSUC.parameterize(node, stage * [1, 3], [0.5, 0.5]) do ω
            return JuMP.set_upper_bound(x.out, ω)
        end
        @stageobjective(node, x.out)
    end
    MSUC.train(model; iteration_limit = 1, print_level = 0)
    rule = MSUC.Statistical(; num_replications = 20)
    @test MSUC.stopping_rule_status(rule) == :statistical
    Random.seed!(123)
    @test MSUC.convergence_test(
        model,
        [MSUC.Log(1, 6.0, 9.0, 1.0, 1, 1, " ", false)],
        rule,
    )
    @test MSUC.convergence_test(
        model,
        [MSUC.Log(1, 0.0, 9.0, 1.0, 1, 1, " ", false)],
        rule,
    )
    @test !MSUC.convergence_test(
        model,
        [MSUC.Log(1, 12.0, 9.0, 1.0, 1, 1, " ", false)],
        rule,
    )
    rule = MSUC.Statistical(; num_replications = 20, iteration_period = 2)
    @test !MSUC.convergence_test(
        model,
        [MSUC.Log(1, 6.0, 9.0, 1.0, 1, 1, " ", false)],
        rule,
    )
    return
end

function test_BoundStalling()
    graph = MSUC.PolicyGraph(
        MSUC.LinearGraph(2);
        lower_bound = 0.0,
        direct_mode = false,
    ) do node, stage
        @variable(node, x, MSUC.State, initial_value = 0)
    end
    rule = MSUC.BoundStalling(3, 1.0)
    @test MSUC.stopping_rule_status(rule) == :bound_stalling
    # Not enough iterations to terminate.
    @test !MSUC.convergence_test(
        graph,
        [
            MSUC.Log(1, 0.0, 0.0, 1.0, 1, 1, " ", false),
            MSUC.Log(2, 1.9, 0.0, 1.0, 1, 1, " ", false),
            MSUC.Log(3, 2.0, 0.0, 1.0, 1, 1, " ", false),
            MSUC.Log(4, 2.0, 0.0, 1.0, 1, 1, " ", false),
        ],
        rule,
    )
    # Now there is. But only just...
    @test MSUC.convergence_test(
        graph,
        [
            MSUC.Log(1, 0.0, 0.0, 1.0, 1, 1, " ", false),
            MSUC.Log(2, 1.9, 0.0, 1.0, 1, 1, " ", false),
            MSUC.Log(3, 2.0, 0.0, 1.0, 1, 1, " ", false),
            MSUC.Log(4, 2.0, 0.0, 1.0, 1, 1, " ", false),
            MSUC.Log(5, 2.9, 0.0, 1.0, 1, 1, " ", false),
        ],
        rule,
    )
    # This also meets the test, but we don't terminate because it hasn't
    # differed from the initial bound.
    @test !MSUC.convergence_test(
        graph,
        [
            MSUC.Log(1, 0.0, 0.1, 1.0, 1, 1, " ", false),
            MSUC.Log(2, 0.0, 0.2, 1.1, 1, 2, " ", false),
            MSUC.Log(3, 0.0, 0.1, 1.2, 1, 3, " ", false),
            MSUC.Log(4, 0.0, 0.0, 1.3, 1, 4, " ", false),
        ],
        rule,
    )
    # This also meets the test, because it looks like a deterministic
    # policy
    @test MSUC.convergence_test(
        graph,
        [
            MSUC.Log(1, 0.0, 0.0, 1.0, 1, 1, " ", false),
            MSUC.Log(2, 0.0, 0.0, 1.0, 1, 1, " ", false),
            MSUC.Log(3, 0.0, 0.0, 1.0, 1, 1, " ", false),
            MSUC.Log(4, 0.0, 0.0, 1.0, 1, 1, " ", false),
        ],
        rule,
    )
    rule = MSUC.BoundStalling(5, 1.0)
    @test !MSUC.convergence_test(
        graph,
        [
            MSUC.Log(1, 0.0, 0.0, 1.0, 1, 1, " ", false),
            MSUC.Log(2, 0.0, 0.0, 1.0, 1, 1, " ", false),
            MSUC.Log(3, 0.0, 0.0, 1.0, 1, 1, " ", false),
            MSUC.Log(4, 0.0, 0.0, 1.0, 1, 1, " ", false),
        ],
        rule,
    )
    return
end

function test_StoppingChain()
    graph = MSUC.PolicyGraph(
        MSUC.LinearGraph(2);
        lower_bound = 0.0,
        direct_mode = false,
    ) do node, stage
        @variable(node, x, MSUC.State, initial_value = 0)
    end
    rule = MSUC.StoppingChain(MSUC.IterationLimit(2), MSUC.TimeLimit(60.0))
    @test MSUC.stopping_rule_status(rule) ==
          Symbol("iteration_limit ∧ time_limit")
    # Not enough iterations to terminate.
    @test !MSUC.convergence_test(
        graph,
        [MSUC.Log(1, 0.0, 0.0, 1.0, 1, 1, " ", false)],
        rule,
    )
    # How there is. But not enough time.
    @test !MSUC.convergence_test(
        graph,
        [
            MSUC.Log(1, 0.0, 0.0, 1.0, 1, 1, " ", false),
            MSUC.Log(2, 0.0, 0.0, 59.0, 1, 1, " ", false),
        ],
        rule,
    )
    # Both satisfied.
    @test MSUC.convergence_test(
        graph,
        [
            MSUC.Log(1, 0.0, 0.0, 1.0, 1, 1, " ", false),
            MSUC.Log(2, 0.0, 0.0, 59.0, 1, 1, " ", false),
            MSUC.Log(3, 0.0, 0.0, 60.1, 1, 1, " ", false),
        ],
        rule,
    )
    return
end

function test_SimulationStoppingRule()
    graph = MSUC.LinearPolicyGraph(;
        stages = 2,
        lower_bound = 0.0,
        optimizer = HiGHS.Optimizer,
    ) do node, stage
        @variable(node, x >= 0, MSUC.State, initial_value = 0)
        @stageobjective(node, x.out)
    end
    rule = MSUC.SimulationStoppingRule()
    @test rule.replications == -1
    @test MSUC.stopping_rule_status(rule) == :simulation_stopping
    log = [
        MSUC.Log(1, 0.000000e+00, 8.316000e+03, 1.559195, 1, 14, "", false),
        MSUC.Log(2, 3.171195e+03, 8.767171e+03, 1.801409, 1, 136, "", false),
        MSUC.Log(3, 4.057980e+03, 4.500000e+03, 1.807249, 1, 150, "", false),
        MSUC.Log(4, 4.074139e+03, 2.314272e+03, 1.813528, 1, 164, "", false),
        MSUC.Log(5, 4.074139e+03, 4.716000e+03, 1.819679, 1, 178, "", false),
        MSUC.Log(6, 4.074139e+03, 2.308500e+03, 1.824431, 1, 192, "", false),
        MSUC.Log(7, 4.074139e+03, 2.308500e+03, 1.830817, 1, 206, "", false),
        MSUC.Log(8, 4.074139e+03, 2.308500e+03, 1.837420, 1, 220, "", false),
        MSUC.Log(9, 4.074139e+03, 5.132230e+03, 1.843861, 1, 234, "", false),
        MSUC.Log(10, 4.074139e+03, 5.197500e+03, 1.850351, 1, 248, "", false),
        MSUC.Log(11, 4.074139e+03, 4.716000e+03, 1.856620, 1, 262, "", false),
        MSUC.Log(12, 4.074139e+03, 2.308500e+03, 1.862838, 1, 276, "", false),
        MSUC.Log(13, 4.074139e+03, 2.308500e+03, 1.869224, 1, 290, "", false),
        MSUC.Log(14, 4.074139e+03, 2.308500e+03, 1.875853, 1, 304, "", false),
        MSUC.Log(15, 4.074139e+03, 2.308500e+03, 1.882504, 1, 318, "", false),
        MSUC.Log(16, 4.074139e+03, 5.197500e+03, 1.889759, 1, 332, "", false),
        MSUC.Log(17, 4.074139e+03, 5.132230e+03, 1.896462, 1, 346, "", false),
        MSUC.Log(18, 4.074139e+03, 8.086500e+03, 1.903102, 1, 360, "", false),
        MSUC.Log(19, 4.074139e+03, 2.308500e+03, 1.910075, 1, 374, "", false),
        MSUC.Log(20, 4.074139e+03, 5.132230e+03, 1.917460, 1, 388, "", false),
    ]
    @test !MSUC.convergence_test(graph, log[1:1], rule)
    @test rule.replications == 1
    @test !MSUC.convergence_test(graph, log[1:4], rule)
    @test !MSUC.convergence_test(graph, log[1:10], rule)
    @test !MSUC.convergence_test(graph, log[1:19], rule)
    @test MSUC.convergence_test(graph, log[1:20], rule)
    rule = MSUC.SimulationStoppingRule(; period = 1)
    rule.last_iteration = 20
    @test rule.period == 1
    @test !MSUC.convergence_test(graph, log, rule)
    return
end

function test_FirstStageStoppingRule()
    model = MSUC.LinearPolicyGraph(;
        stages = 2,
        lower_bound = 0.0,
        optimizer = HiGHS.Optimizer,
    ) do node, stage
        @variable(node, x >= 0, MSUC.State, initial_value = 0)
        @stageobjective(node, x.out)
    end
    rule = MSUC.FirstStageStoppingRule()
    MSUC.train(model; stopping_rules = [rule])
    n_iterations = length(rule.data)
    @test 50 <= n_iterations <= 55  # Depends on nthreads
    log = model.most_recent_training_results.log
    set_lower_bound(model[1].subproblem[:x].out, 1.0)
    @test !MSUC.convergence_test(model, log, rule)
    @test length(rule.data) == n_iterations + 1
    model = MSUC.LinearPolicyGraph(;
        stages = 2,
        lower_bound = 0.0,
        optimizer = HiGHS.Optimizer,
    ) do node, stage
        @variable(node, x >= 0, MSUC.State, initial_value = 0)
        MSUC.parameterize(node, 1:2) do w
            return @stageobjective(node, w * x.out)
        end
        return
    end
    @test_throws(
        ErrorException(
            "FirstStageStoppingRule cannot be applied because first-stage is " *
            "not deterministic",
        ),
        MSUC.train(
            model;
            print_level = 0,
            stopping_rules = [MSUC.FirstStageStoppingRule()],
        ),
    )
    graph = MSUC.Graph(0, [1, 2], [(0 => 1, 0.5), (0 => 2, 0.5)])
    model = MSUC.PolicyGraph(
        graph;
        lower_bound = 0.0,
        optimizer = HiGHS.Optimizer,
    ) do node, stage
        @variable(node, x >= 0, MSUC.State, initial_value = 0)
        @stageobjective(node, x.out)
        return
    end
    @test_throws(
        ErrorException(
            "FirstStageStoppingRule cannot be applied because first-stage is " *
            "not deterministic",
        ),
        MSUC.train(
            model;
            print_level = 0,
            stopping_rules = [MSUC.FirstStageStoppingRule()],
        ),
    )
    return
end

end  # module

TestStoppingRules.runtests()
