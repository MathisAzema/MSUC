# Toy example: short horizon and small innovation support, so that everything can be checked exactly.
#
# At T = 24 the 3-bin formulation does not converge and the scenario tree is out of reach. With T = 4 and
# S = 2 the tree has 2^4 = 16 leaves, so the exact multistage optimum is computable from its deterministic
# equivalent and, for both formulations,
#
#     lower bound (training)  <=  optimum  <=  policy cost (simulation)
#
# This covers cut validity, the equivalence of Prop. 3, the ARMA process, the state zeta and the delta
# coefficients.
#
#   julia --project=. examples/toy_convergence.jl

using MSUC, Gurobi, Printf, Random

const GRB_ENV = Gurobi.Env()
optimizer = () -> Gurobi.Optimizer(GRB_ENV)

T     = 4
S     = 2
FORCE = 1.0
ITER  = 30
TOL   = 1e-6
SEED  = 1234    # the forward pass samples from the global rand(): fixed so both formulations see the same scenarios

instance = MSUC.parse_nc4("Data/T-Ramp/10_0_1_w.nc4", optimizer, T; S = S)
@printf("toy: T = %d, S = %d, %d units, %d intervals, %d scenarios\n\n",
        T, S, instance.N, sum(length(u.intervals) for u in instance.Thermalunits), S^T)

opt = MSUC.multistage_extensive(instance; force = FORCE, gap = 1e-9, timelimit = 300)[1]
@printf("EXACT multistage optimum (tree of %d nodes) = %.4f\n\n",
        sum(S^t for t in 0:T), opt)

function run(builder, method, label)
    model = MSUC.PolicyGraph(
        builder, instance, FORCE, MSUC.Graph(T + 1), method;
        lower_bound = 0.0, optimizer = optimizer,
    )
    Random.seed!(SEED)
    MSUC.train(model; iteration_limit = ITER, print_level = 0)
    bounds = [l.bound for l in model.most_recent_training_results.log]
    k = findfirst(x -> (opt - x) / opt < TOL, bounds)
    sim = MSUC.simulate(model)

    @printf("%-16s bound = %.4f   policy = %.4f   converges at iteration %s\n",
            label, bounds[end], sim.cost, k === nothing ? "> $ITER" : string(k))
    @printf("%-16s gap = %.2e   (%d paths, exhaustive = %s)\n",
            "", (sim.cost - bounds[end]) / opt, sim.npaths, sim.exhaustive)

    @assert bounds[end] <= opt * (1 + TOL)  "bound above the optimum"
    @assert sim.cost >= opt * (1 - TOL)     "policy cost below the optimum"
    return bounds, sim
end

b_ext, s_ext = run(MSUC.subproblem_builder_UC_extended, MSUC.ExtendedFormulation, "interval (38)")
b_3bin, s_3bin = run(MSUC.subproblem_builder_UC_def, MSUC.SDDPFormulation, "3-bin (13)")

println("\n it   interval (38)     3-bin (13)     % optimum (int. / 3-bin)")
for k in 1:min(ITER, length(b_ext), length(b_3bin))
    @printf("%3d  %15.2f  %13.2f     %6.2f / %6.2f\n",
            k, b_ext[k], b_3bin[k], 100 * b_ext[k] / opt, 100 * b_3bin[k] / opt)
end

# Prop. 3: both formulations have the same optimum.
@assert abs(b_ext[end] - b_3bin[end]) / opt < TOL
println()
@printf("OK  bound <= optimum <= policy cost, verified for both formulations.\n")
@printf("OK  both bounds converge to the same optimum (Prop. 3).\n")
@printf("    exact optimum    %.4f\n", opt)
@printf("    interval         bound %.4f / policy %.4f\n", b_ext[end], s_ext.cost)
@printf("    3-bin            bound %.4f / policy %.4f\n", b_3bin[end], s_3bin.cost)
