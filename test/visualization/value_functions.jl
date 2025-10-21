#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

module TestValueFunctions

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

function test_ValueFunction_Min()
    model = MSUC.LinearPolicyGraph(;
        stages = 2,
        lower_bound = 0.0,
        optimizer = HiGHS.Optimizer,
    ) do sp, t
        @variable(sp, x >= 0, MSUC.State, initial_value = 1.5)
        @constraint(sp, x.out == x.in)
        @stageobjective(sp, 2 * x.out)
    end
    V1 = MSUC.ValueFunction(model[1])
    @test MSUC.evaluate(V1, Dict(:x => 1.0)) == (0.0, Dict(:x => 0.0))
    MSUC.train(model; iteration_limit = 2, print_level = 0)
    V1 = MSUC.ValueFunction(model[1])
    @test sprint(show, V1) == "A value function for node 1"
    for (xhat, yhat, pihat) in
        [(0.0, 0.0, 0.0), (1.0, 2.0, 2.0), (2.0, 4.0, 2.0)]
        @test MSUC.evaluate(V1, Dict(:x => xhat)) == (yhat, Dict(:x => pihat))
    end
    return
end

function test_ValueFunction_Max()
    model = MSUC.LinearPolicyGraph(;
        stages = 2,
        sense = :Max,
        upper_bound = 0.0,
        optimizer = HiGHS.Optimizer,
    ) do sp, t
        @variable(sp, x >= 0, MSUC.State, initial_value = 1.5)
        @constraint(sp, x.out == x.in)
        @stageobjective(sp, -2 * x.out)
    end
    MSUC.train(model; iteration_limit = 2, print_level = 0)
    V1 = MSUC.ValueFunction(model[1])
    for (xhat, yhat, pihat) in
        [(0.0, 0.0, 0.0), (1.0, 2.0, 2.0), (2.0, 4.0, 2.0)]
        (y, duals) = MSUC.evaluate(V1, Dict(:x => xhat))
        @test y == -yhat
        @test duals == Dict(:x => -pihat)
    end
    return
end

function test_ValueFunction_optimizer()
    model = MSUC.LinearPolicyGraph(;
        stages = 2,
        lower_bound = 0.0,
        optimizer = HiGHS.Optimizer,
        direct_mode = true,
    ) do sp, t
        @variable(sp, x >= 0, MSUC.State, initial_value = 1.5)
        @constraint(sp, x.out == x.in)
        @stageobjective(sp, 2 * x.out)
    end
    MSUC.train(model; iteration_limit = 2, print_level = 0)
    V1 = MSUC.ValueFunction(model[1])
    @test_throws JuMP.NoOptimizer() MSUC.evaluate(V1, Dict(:x => 1.0))
    JuMP.set_optimizer(V1, HiGHS.Optimizer)
    (y, _) = MSUC.evaluate(V1, Dict(:x => 1.0))
    @test y == 2.0
    return
end

function test_ValueFunction_objective_state()
    model = MSUC.LinearPolicyGraph(;
        stages = 2,
        lower_bound = 0.0,
        optimizer = HiGHS.Optimizer,
    ) do sp, t
        @variable(sp, x >= 0, MSUC.State, initial_value = 1.5)
        MSUC.add_objective_state(
            sp;
            initial_value = 0.0,
            lipschitz = 10.0,
        ) do p, ω
            return p + ω
        end
        @constraint(sp, x.out == x.in)
        MSUC.parameterize(sp, [1, 2]) do ω
            price = MSUC.objective_state(sp)
            @stageobjective(sp, price * x.out)
        end
    end
    MSUC.train(model; iteration_limit = 10, print_level = 0)
    V1 = MSUC.ValueFunction(model[1])
    @test_throws AssertionError MSUC.evaluate(V1, Dict(:x => 1.0))
    @test MSUC.evaluate(V1, Dict(:x => 1.0); objective_state = 1) ==
          (2.5, Dict(:x => 2.5))
    @test MSUC.evaluate(V1, Dict(:x => 0.0); objective_state = 2) ==
          (0.0, Dict(:x => 3.5))
    return
end

function test_ValueFunction_belief_state()
    graph = MSUC.MarkovianGraph(Matrix{Float64}[[0.5 0.5], [1.0 0.0; 0.0 1.0]])
    MSUC.add_ambiguity_set(graph, [(1, 1), (1, 2)])
    MSUC.add_ambiguity_set(graph, [(2, 1), (2, 2)])
    model = MSUC.PolicyGraph(
        graph;
        lower_bound = 0.0,
        optimizer = HiGHS.Optimizer,
    ) do sp, node
        (t, i) = node
        @variable(sp, x >= 0, MSUC.State, initial_value = 1.5)
        @constraint(sp, x.out == x.in)
        P = [[0.2, 0.8], [0.8, 0.2]]
        MSUC.parameterize(sp, [1, 2], P[i]) do ω
            @stageobjective(sp, ω * x.out)
        end
    end
    MSUC.train(model; iteration_limit = 10, print_level = 0)
    V11 = MSUC.ValueFunction(model[(1, 1)])
    @test_throws AssertionError MSUC.evaluate(V11, Dict(:x => 1.0))
    b = Dict((1, 1) => 0.8, (1, 2) => 0.2)
    (y, duals) = MSUC.evaluate(V11, Dict(:x => 1.0); belief_state = b)
    @test duals[:x] ≈ y ≈ 1.68
    V12 = MSUC.ValueFunction(model[(1, 2)])
    (y, duals) = MSUC.evaluate(V12, Dict(:x => 1.0); belief_state = b)
    @test duals[:x] ≈ y ≈ 1.68
    return
end

function test_ValuaeFunction_plot()
    model = MSUC.LinearPolicyGraph(;
        stages = 2,
        sense = :Min,
        lower_bound = 0.0,
        optimizer = HiGHS.Optimizer,
    ) do sp, t
        @variable(sp, x >= 0, MSUC.State, initial_value = 1.5)
        @variable(sp, y >= 0, MSUC.State, initial_value = 0)
        @constraint(sp, x.out >= x.in)
        @constraint(sp, x.out >= 2 * x.in - 1)
        @constraint(sp, y.out == y.in)
        @stageobjective(sp, x.out + y.out)
    end
    MSUC.train(model; iteration_limit = 3, print_level = 0)
    V1 = MSUC.ValueFunction(model[1])
    MSUC.plot(V1; x = 0:0.1:2, y = 0, open = false)
    MSUC.plot(V1; x = 0:0.1:2, y = 0:0.1:2, open = false)
    return
end

function test_ValueFunction_risk_measure_min()
    model = MSUC.LinearPolicyGraph(;
        stages = 2,
        lower_bound = 0.0,
        optimizer = HiGHS.Optimizer,
    ) do sp, t
        @variable(sp, x >= 0, MSUC.State, initial_value = 1.5)
        @constraint(sp, x.out == x.in)
        MSUC.parameterize(sp, [1, 2]) do w
            @stageobjective(sp, w * x.out)
            return
        end
        return
    end
    MSUC.train(
        model;
        iteration_limit = 2,
        print_level = 0,
        risk_measure = MSUC.CVaR(0.25),
        cut_type = MSUC.MULTI_CUT,
    )
    V1 = MSUC.ValueFunction(model[1])
    for (xhat, yhat, pihat) in
        [(0.0, 0.0, 0.0), (1.0, 2.0, 2.0), (2.0, 4.0, 2.0)]
        @test MSUC.evaluate(V1, Dict(:x => xhat)) == (yhat, Dict(:x => pihat))
    end
    return
end

function test_ValueFunction_risk_measure_max()
    model = MSUC.LinearPolicyGraph(;
        stages = 2,
        upper_bound = 10.0,
        sense = :Max,
        optimizer = HiGHS.Optimizer,
    ) do sp, t
        @variable(sp, x >= 0, MSUC.State, initial_value = 1.5)
        @constraint(sp, x.out == x.in)
        MSUC.parameterize(sp, [1, 2]) do w
            @stageobjective(sp, w * x.out)
            return
        end
        return
    end
    MSUC.train(
        model;
        iteration_limit = 2,
        print_level = 0,
        risk_measure = MSUC.CVaR(0.25),
        cut_type = MSUC.MULTI_CUT,
    )
    V1 = MSUC.ValueFunction(model[1])
    for (xhat, yhat, pihat) in
        [(0.0, 0.0, 1.0), (1.0, 1.0, 1.0), (2.0, 2.0, 1.0)]
        @test MSUC.evaluate(V1, Dict(:x => xhat)) == (yhat, Dict(:x => pihat))
        @test MSUC.evaluate(V1, Dict("x" => xhat)) == (yhat, Dict(:x => pihat))
    end
    return
end

end  # module

TestValueFunctions.runtests()
