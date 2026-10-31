# --- Sauvegarde et rechargement de la politique ------------------------------------------------

# Composantes de x dont au moins un planning visite a besoin : intervalles choisis en formulation
# intervalle, composantes non nulles de (u, v, w) en 3-bin. C'est tout ce dont l'evaluation d'une
# coupe en l'un de ces plannings depend.
function used_components(model, schedules)
    if model.method == ExtendedFormulation
        return sort(unique(reduce(vcat, [s.selected for s in schedules])))
    end
    return findall(l -> any(s.x[l] != 0 for s in schedules), eachindex(schedules[1].x))
end

# Planning x^k reconstruit a partir de ses composantes non nulles : il ne contient que des 0 et des
# 1, et x = (u, v, w) se decoupe en u (N x (T+1)), v et w (N x T).
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

# Ecrit les plannings et les coupes d'etape restreintes aux composantes utiles. En 3-bin, `beta` (une
# pente par unite en y) ne depend pas du planning et est gardee entiere. Les coupes de la racine ne
# sont pas enregistrees : le MIP racine n'est jamais resolu a la simulation.
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
    CSV.write(
        joinpath(folder, "cuts.csv"),
        DataFrame(permutedims(reduce(hcat, rows)), :auto);
        writeheader = false,
    )
    return
end

# Relit ce que `write_policy` a ecrit : les plannings, les composantes utiles, la matrice des coupes
# et, pour chaque noeud, les lignes de ses coupes dans l'ordre de creation.
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
    cuts = Matrix{Float64}(
        CSV.read(joinpath(folder, "cuts.csv"), DataFrame; header = false, types = Float64),
    )
    node_rows = [findall(==(i), view(cuts, :, 1)) for i in 1:length(model.nodes)]
    return (; schedules, used, cuts, node_rows)
end

function expand(v, used, n)
    full = zeros(n)
    full[used] .= v
    return full
end

# Coupe de taille complete a partir d'une ligne de `cuts.csv`, nulle hors des composantes utiles :
# ses valeurs en tout planning de l'entrainement sont celles de la coupe d'origine.
function new_cut(model, row, used)
    interval = model.method == ExtendedFormulation
    trivial = model[2].cuts[1]    # la coupe triviale donne les tailles
    nz, nu = length(trivial.delta), length(used)
    tau = expand(row[(3+nz):(2+nz+nu)], used, length(trivial.tau))
    beta_part = row[(3+nz+nu):end]
    beta = interval ? expand(beta_part, used, length(trivial.beta)) : beta_part
    Cut = interval ? IntervalCut : BinCut
    return Cut(row[2], tau, beta, row[3:(2+nz)], nothing)
end

# Ajoute aux etapes de dispatch les coupes creees jusqu'a l'iteration `j` (0 : aucune) : chaque passe
# ajoute une coupe a chaque noeud d'etape, donc `passes` par iteration. `loaded[i]` est le nombre de
# coupes deja chargees au noeud i, ce qui permet de charger par instants croissants sans rien
# recharger. La coupe triviale (position 1 de chaque noeud) existe deja dans un modele neuf.
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
