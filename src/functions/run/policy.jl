# Components of x used by at least one visited schedule (chosen intervals, or nonzero (u, v, w)
# entries for 3-bin): the only ones needed to evaluate a cut at those schedules.
function used_components(model, schedules)
    if model.method == ExtendedFormulation
        return sort(unique(reduce(vcat, [s.selected for s in schedules])))
    end
    return findall(l -> any(s.x[l] != 0 for s in schedules), eachindex(schedules[1].x))
end

# Rebuilds x^k = (u, v, w) from its nonzero components (all equal to 1).
function rebuild_schedule(instance, selected, x_nonzero)
    N, T = instance.N, instance.TimeHorizon
    x = zeros(commitment_dimension(instance))
    x[x_nonzero] .= 1.0
    nu, nv = N * (T + 1), N * T
    u = reshape(x[1:nu], N, T + 1)
    v = reshape(x[(nu+1):(nu+nv)], N, T)
    w = reshape(x[(nu+nv+1):end], N, T)
    return Schedule(u, v, w, selected, instance.intervals.unit[selected])
end

# Writes the schedules and the stage cuts restricted to the used components. 3-bin `beta` does not
# depend on the schedule and is kept whole. Root cuts are not saved: simulation never solves the root MIP.
function write_policy(folder, model, schedules)
    interval = model.method == ExtendedFormulation
    used = used_components(model, schedules)
    CSV.write(
        joinpath(folder, "schedules.csv"),
        DataFrame(
            iteration = 1:length(schedules),
            selected = [join(s.selected, ";") for s in schedules],
            x_nonzero = [join(findall(!iszero, s.x), ";") for s in schedules],
        ),
    )
    CSV.write(joinpath(folder, "used_components.csv"), DataFrame(component = used))

    nodes = length(model.nodes)
    rows = [
        vcat(i, c.alpha, c.delta, c.tau[used], interval ? c.beta[used] : c.beta) for
        i in 2:(nodes-1) for c in model[i].cuts[2:end]
    ]
    write_cuts(joinpath(folder, CUTS_FILE), permutedims(reduce(hcat, rows)))
    return
end

# Cut matrix (one row per cut): Float64 binary, xz-compressed. Format: two Int64 (rows, columns),
# then the values in column-major order. About 4.6x smaller than CSV. `xz -9` needs ~700 MB of memory.
const CUTS_FILE = "cuts.bin.xz"

function write_cuts(path, cuts::Matrix{Float64})
    open(pipeline(`$(XZ_jll.xz()) -9 -c`; stdout = path), "w") do io
        write(io, Int64(size(cuts, 1)), Int64(size(cuts, 2)))
        write(io, cuts)
    end
    return
end

function read_cuts(path)
    open(`$(XZ_jll.xz()) -dc $path`) do io
        nrows, ncols = read(io, Int64), read(io, Int64)
        cuts = Matrix{Float64}(undef, nrows, ncols)
        read!(io, cuts)
        return cuts
    end
end

# Reads `cuts.bin.xz`, or `cuts.csv` for older results.
function read_cuts_folder(folder)
    binary = joinpath(folder, CUTS_FILE)
    isfile(binary) && return read_cuts(binary)
    return Matrix{Float64}(
        CSV.read(joinpath(folder, "cuts.csv"), DataFrame; header = false, types = Float64),
    )
end

# Reads back what `write_policy` wrote, plus for each node the rows of its cuts in creation order.
function read_policy(folder, model)
    table = CSV.read(
        joinpath(folder, "schedules.csv"),
        DataFrame;
        types = Dict(:selected => String, :x_nonzero => String),
        missingstring = nothing,
    )
    ints(text) = isempty(text) ? Int[] : parse.(Int, split(text, ";"))
    schedules = [
        rebuild_schedule(model.instance, ints(r.selected), ints(r.x_nonzero)) for
        r in eachrow(table)
    ]
    used = CSV.read(joinpath(folder, "used_components.csv"), DataFrame).component
    cuts = read_cuts_folder(folder)
    node_rows = [findall(==(i), view(cuts, :, 1)) for i in 1:length(model.nodes)]
    return (; schedules, used, cuts, node_rows)
end

function expand(v, used, n)
    full = zeros(n)
    full[used] .= v
    return full
end

# Full-size cut from a saved row, zero outside the used components: it agrees with the original
# cut at every training schedule.
function new_cut(model, row, used)
    interval = model.method == ExtendedFormulation
    trivial = model[2].cuts[1]    # gives the sizes
    nz, nu = length(trivial.delta), length(used)
    tau = expand(row[(3+nz):(2+nz+nu)], used, length(trivial.tau))
    beta_part = row[(3+nz+nu):end]
    beta = interval ? expand(beta_part, used, length(trivial.beta)) : beta_part
    Cut = interval ? IntervalCut : BinCut
    return Cut(row[2], tau, beta, row[3:(2+nz)], nothing)
end

# Adds to the stage nodes the cuts created up to iteration `j` (`passes` per iteration). `loaded[i]`
# counts the cuts already loaded at node i, so successive calls only add the new ones. The trivial
# cut (position 1) already exists in a fresh model.
function load_cuts!(model, data, passes::Int, j::Int, loaded::Vector{Int})
    for i in 2:length(loaded)
        target = min(passes * j, length(data.node_rows[i]))
        for r in data.node_rows[i][(loaded[i]+1):target]
            add_cut!(model, model[i], new_cut(model, data.cuts[r, :], data.used))
        end
        loaded[i] = max(loaded[i], target)
    end
    return
end
