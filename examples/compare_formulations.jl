# Compares the three solvers on the same deterministic problem (force = 0), where the exact optimum is
# computable and serves as the reference for the two SDDP lower bounds.
#
#   julia --project=. examples/compare_formulations.jl

using MSUC, Gurobi, Printf

const GRB_ENV = Gurobi.Env()
optimizer = () -> Gurobi.Optimizer(GRB_ENV)

T          = 24
INSTANCE   = "Data/T-Ramp/10_0_1_w.nc4"
ITERATIONS = 10
FORCE      = 0.0

instance = MSUC.parse_nc4(INSTANCE, optimizer, T; S = 1)
@printf("instance %s : %d unites, %d intervalles, T = %d\n\n",
        basename(INSTANCE), instance.N,
        sum(length(u.intervals) for u in instance.Thermalunits), T)

res = MSUC.multistage_extensive(instance; force = FORCE, gap = 1e-6, timelimit = 300)
opt = res[1]
@printf("Formulation extensive   optimum = %.1f   (%.1f s)\n", opt, res[3])

function run_sddp(builder, method, label)
    model = MSUC.PolicyGraph(
        builder, instance, FORCE, MSUC.Graph(T + 1), method;
        lower_bound = 0.0, optimizer = optimizer,
    )
    t0 = time()
    MSUC.train(model; iteration_limit = ITERATIONS, print_level = 0)
    log = model.most_recent_training_results.log
    @printf("SDDP %-17s borne   = %.1f   (%5.1f %% de l'optimum, %.1f s)\n",
            label, log[end].bound, 100 * log[end].bound / opt, time() - t0)
    return [l.bound for l in log]
end

b_ext = run_sddp(MSUC.subproblem_builder_UC_extended, MSUC.ExtendedFormulation, "intervalle")
b_3bin = run_sddp(MSUC.subproblem_builder_UC_def, MSUC.SDDPFormulation, "3-bin")

println("\n it   intervalle (39)        3-bin (13)        % optimum (int. / 3-bin)")
for k in 1:ITERATIONS
    @printf("%3d  %16.1f  %16.1f     %5.1f / %5.1f\n",
            k, b_ext[k], b_3bin[k], 100 * b_ext[k] / opt, 100 * b_3bin[k] / opt)
end

# A valid lower bound cannot exceed the exact optimum.
@assert all(b .<= opt * (1 + 1e-6) for b in (b_ext[end], b_3bin[end]))
println("\nOK : les deux bornes restent sous l'optimum exact.")
