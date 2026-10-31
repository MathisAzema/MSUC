# --- Modele et dossier de resultats d'un run (voir `runSMS.jl`) --------------------------------

# Les deux formulations de la coupe : (constructeur des sous-problemes, formulation).
const METHODS = Dict(
    "interval" => (subproblem_builder_UC_extended, ExtendedFormulation),
    "bin3" => (subproblem_builder_UC_def, SDDPFormulation),
)

# Graine de tous les runs : entrainement (tirages de la passe avant) et simulation.
const SEED = 1

# `settings` : (T, S, sigma, mip_gap, passes_per_mip). Tous les parametres figurent dans le nom du
# dossier, pour que des runs de parametres differents ne s'ecrasent jamais.
function results_folder(instance_name, method, time_max, settings)
    gap = settings.mip_gap === nothing ? "default" : string(settings.mip_gap)
    tag = string(
        method, "_T", settings.T, "_S", settings.S, "_sigma", settings.sigma,
        "_passes", settings.passes_per_mip, "_gap", gap,
    )
    return "results_sms/$(tag)/$(instance_name)/time_$(time_max)"
end

# Nom, sans extension, du fichier de `Data/T-Ramp` d'une instance : nombre de groupes, 0, identifiant, w.
instance_name(n_units, id) = "$(n_units)_0_$(id)_w"

# Tous les runs d'une campagne : le produit des listes. Chaque element est
# `(instance, methode, temps limite, settings)`. Les combinaisons (taille, identifiant) sans fichier
# dans `Data/T-Ramp` sont ignorees avec un avertissement (les instances a 200 groupes vont jusqu'a
# l'identifiant 11, les autres jusqu'a 5) ; il y a une erreur si aucune n'existe.
function run_combos(
    size_list,
    id_list,
    method_list,
    time_max_list;
    T,
    S_list,
    sigma_list,
    passes_per_mip_list,
    mip_gap_list,
)
    wanted = [instance_name(n_units, id) for n_units in size_list for id in id_list]
    instances = filter(name -> isfile("Data/T-Ramp/$(name).nc4"), wanted)
    absent = setdiff(wanted, instances)
    isempty(absent) || @warn "instances absentes de Data/T-Ramp, ignorees" absent
    isempty(instances) &&
        error("aucune instance de Data/T-Ramp pour size_list = $(size_list) et id_list = $(id_list)")
    return [
        (instance, method, time_max, (; T, S, sigma, mip_gap, passes_per_mip)) for
        instance in instances for method in method_list for time_max in time_max_list for
        S in S_list for sigma in sigma_list for passes_per_mip in passes_per_mip_list for
        mip_gap in mip_gap_list
    ]
end

# Fabrique d'optimiseur d'un run, enregistree dans chaque processus (`set_run_optimizer!`, appelee sur
# chaque worker par `runSMS.jl`). Les jobs la relisent ici au lieu de la recevoir de `pmap`, qui
# enverrait sinon l'environnement Gurobi du processus principal aux workers.
const RUN_OPTIMIZER = Ref{Any}(nothing)

set_run_optimizer!(factory) = (RUN_OPTIMIZER[] = factory; nothing)

function run_optimizer()
    RUN_OPTIMIZER[] === nothing && error(
        "aucun optimiseur enregistre : appeler `MSUC.set_run_optimizer!(() -> Gurobi.Optimizer(env))` " *
        "sur chaque processus (voir runSMS.jl)",
    )
    return RUN_OPTIMIZER[]
end

# Un modele neuf (sans coupe) : meme construction a l'entrainement et a la simulation.
function build_model(instance_name, method, settings, optimizer)
    builder, formulation = METHODS[method]
    instance = parse_nc4(
        "Data/T-Ramp/$(instance_name).nc4",
        optimizer,
        settings.T;
        sigma = settings.sigma,
        S = settings.S,
    )
    # force = 1 : l'amplitude de l'alea est entierement donnee par `sigma`.
    return PolicyGraph(
        builder,
        instance,
        1.0,
        Graph(settings.T + 1),
        formulation;
        lower_bound = 0.0,
        optimizer = optimizer,
    )
end
