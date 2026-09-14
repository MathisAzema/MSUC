# MSUC.jl — SDDP à coupes intervalles pour l'unit commitment stochastique multi-étapes

Code de l'article **« Stronger cuts for stochastic dual dynamic programming for multistage
stochastic unit commitment problems based on interval variables »**
(M. Azéma, V. Leclère, W. van Ackooij).

Le problème : l'engagement (*commitment*) des groupes est décidé la veille et reste **fixe**,
tandis que le dispatch s'adapte séquentiellement à la demande nette observée. Seul le problème
racine est non convexe ; conditionnellement au planning, chaque étape est un programme linéaire.
On peut donc appliquer un SDDP paramétrique.

La contribution est une **formulation étendue par intervalles d'exploitation** qui produit des
coupes de programmation dynamique plus fortes que la représentation 3-bin standard.

Ce dépôt est un fork de [SDDP.jl](https://github.com/odow/SDDP.jl) (module renommé `MSUC`)
augmenté du modèle UC et du générateur de coupes intervalles.

---

## Installation

| | |
|---|---|
| Julia | **1.12.6** (version de référence du `Manifest.toml`) |
| Solveur | **Gurobi** avec licence valide (`GRB_LICENSE_FILE`) |

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using MSUC; println("ok")'
```

> **Julia 1.11** — le `Manifest.toml` versionné est résolu pour 1.12. Pour travailler en 1.11,
> supprimer `Manifest.toml` et relancer `Pkg.instantiate()`.

---

## Démarrage rapide

```julia
using MSUC, Gurobi
const GRB_ENV = Gurobi.Env()
optimizer = () -> Gurobi.Optimizer(GRB_ENV)

T = 24
instance = MSUC.parse_nc4("Data/T-Ramp/10_0_1_w.nc4", optimizer, T)

model = MSUC.PolicyGraph(
    MSUC.subproblem_builder_UC_extended,   # ou subproblem_builder_UC_def pour la 3-bin
    instance,
    1.0,                                   # `force` : amplitude de l'aléa (0 => déterministe)
    MSUC.LinearGraph(T + 1),               # T+1 nœuds : racine + T étapes de dispatch
    MSUC.ExtendedFormulation,              # ou MSUC.SDDPFormulation
    sense = :Min,
    lower_bound = 0.0,
    optimizer = optimizer,
)

MSUC.train(model; iteration_limit = 20)
```

La colonne `bound` du journal est la borne inférieure, issue du problème racine (26) résolu
avec l'approximation courante (Algorithme 2, ligne 5).

> Le problème racine est un **MIP**. La borne publiée est sa borne **duale**
> (`objective_bound`), pas la valeur de l'incumbent : avec un `MIPGap` non nul, l'incumbent
> *majore* l'optimum du MIP et n'est donc pas une borne inférieure valide. La trajectoire de
> la passe avant utilise en revanche la solution incumbent — n'importe quel planning
> admissible convient pour engendrer des coupes. Voir `root_outer_bound`.
>
> La borne duale est valide à chaque itération mais **pas monotone** (le branch-and-bound
> peut s'arrêter à un point moins avancé qu'à l'itération précédente). Comme chaque borne
> minore l'optimum vrai, c'est leur **maximum courant** qui est publié : valide et monotone.

### Comparer les trois solveurs

[`examples/compare_formulations.jl`](examples/compare_formulations.jl) lance les trois sur le
même problème **déterministe** (`force = 0`), où l'optimum exact est calculable et sert de
référence aux deux bornes SDDP :

```bash
julia --project=. examples/compare_formulations.jl
```

```julia
using MSUC, Gurobi, Printf
const GRB_ENV = Gurobi.Env()
optimizer = () -> Gurobi.Optimizer(GRB_ENV)

T, FORCE, ITER = 24, 0.0, 10
instance = MSUC.parse_nc4("Data/T-Ramp/10_0_1_w.nc4", optimizer, T)

# 1. Formulation extensive : optimum exact du problème déterministe
res = MSUC.bin_extensive_neutral(instance; silent = true, force = FORCE, S = 1,
                                 batch = 1, gap = 1e-6, timelimit = 300)
opt = res[3]

# 2. et 3. Les deux SDDP sur le même problème
function run_sddp(builder, method, label)
    model = MSUC.PolicyGraph(builder, instance, FORCE, MSUC.LinearGraph(T + 1), method;
                             sense = :Min, lower_bound = 0.0, optimizer = optimizer)
    MSUC.train(model; iteration_limit = ITER, print_level = 0)
    log = model.most_recent_training_results.log
    @printf("SDDP %-11s borne = %.1f  (%.1f %% de l'optimum)\n",
            label, log[end].bound, 100 * log[end].bound / opt)
    return [l.bound for l in log]
end

b_ext  = run_sddp(MSUC.subproblem_builder_UC_extended, MSUC.ExtendedFormulation, "intervalle")
b_3bin = run_sddp(MSUC.subproblem_builder_UC_def,      MSUC.SDDPFormulation,     "3-bin")

# Une borne inférieure valide ne peut pas dépasser l'optimum exact.
@assert b_ext[end] <= opt && b_3bin[end] <= opt
```

Sortie obtenue sur `10_0_1_w.nc4` (10 unités, 1968 intervalles, optimum exact **1 850 053,2**) :

| itération | intervalle (39) | % opt. | 3-bin (13) | % opt. |
|---|---|---|---|---|
| 1 | 58 914 | 3,2 | 58 914 | 3,2 |
| 4 | 383 739 | 20,7 | 82 955 | 4,5 |
| 5 | 1 795 493 | 97,1 | 85 452 | 4,6 |
| 10 | **1 820 387** | **98,4** | 101 084 | 5,5 |

C'est la thèse de l'article : les coupes intervalles atteignent 98 % de l'optimum en 5 itérations
là où la 3-bin plafonne à 5 %.

> ⚠️ **L'aléa n'est pas paramétré de la même façon** dans `bin_extensive_neutral`
> (demande nette `= Demandbus - force * WGscenario`, soustraction éolienne absolue) et dans les
> builders SDDP (`= Demandbus * (1 + force * ξ)`, erreur relative). Les deux ne coïncident qu'en
> `force = 0` — seul régime où la comparaison à l'optimum exact est licite.


### Exemple jouet : encadrement complet de l'optimum

À `T = 24` la 3-bin ne converge pas et l'arbre de scénarios est hors de portée. À `T = 4`
avec `S = 2`, l'arbre n'a que $2^4 = 16$ feuilles : on calcule l'optimum multi-étapes
**exact** par son équivalent déterministe et on encadre sans rien supposer.
[`examples/toy_convergence.jl`](examples/toy_convergence.jl) :

```bash
julia --project=. examples/toy_convergence.jl
```

$$\underbrace{\text{borne inférieure}}_{\text{entraînement}} \;\le\; \underbrace{\text{optimum}}_{\text{arbre exact}} \;\le\; \underbrace{\text{coût de la politique}}_{\text{simulation}}$$

Optimum multi-étapes exact = **235 205,1131** (arbre de 31 nœuds) :

| itération | intervalle (39) | % opt. | 3-bin (13) | % opt. |
|---|---|---|---|---|
| 1 | 43 262,46 | 18,4 | 43 262,46 | 18,4 |
| 3 | 220 124,06 | 93,6 | 47 397,96 | 20,2 |
| **6** | **235 205,11** | **100,0** | 224 640,38 | 95,5 |
| **15** | 235 205,11 | 100,0 | **235 205,11** | **100,0** |

| | borne | politique | encadrement |
|---|---|---|---|
| intervalle (39) | 235 205,1131 | 235 205,1131 | $-5\cdot10^{-16}$ |
| 3-bin (13) | 235 205,1131 | 235 205,1131 | $0$ |

Les trois quantités coïncident à la précision machine, pour les deux formulations — la
formulation intervalle en **6 itérations contre 15**. C'est la vérification la plus
complète du dépôt : elle couvre la validité des coupes, l'équivalence de la
**Proposition 3**, le processus ARMA, l'état $\zeta$ et les coefficients $\tilde\delta$.

Trois briques la rendent possible :

- **`multistage_extensive(instance; force)`** — équivalent déterministe de l'arbre de
  scénarios, donc l'optimum **exact** du problème multi-étapes (1) avec la
  non-anticipativité (1j). À ne pas confondre avec `bin_extensive_neutral`, qui est
  **deux-étages** : son dispatch voit toute la trajectoire d'un coup, elle *minore* le
  multi-étapes. L'arbre a $\sum_{t=0}^{T} S^t$ nœuds, donc n'est tractable que pour
  $S$ et $T$ petits. Les lignes (44) n'y sont pas imposées — sur l'horizon complet elles
  sont impliquées par les rampes — ce qui rend la référence indépendante de (44) et (46).
- **`simulate_policy(model)`** — coût espéré de la politique entraînée, donc un
  **majorant** de l'optimum. L'engagement est figé (état sortant de la racine), les coupes
  intervalles réinstanciées à ce planning, puis chaque chemin est parcouru en résolvant les
  sous-problèmes d'étage. Le coût d'étage exclut la variable d'épigraphe : on mesure le coût
  réellement encouru, pas l'approximation de la valeur future. Énumère les $S^T$ chemins
  quand c'est tractable (coût espéré **exact**, aucune erreur d'échantillonnage), sinon tire
  `samples` chemins avec une graine.
- **`scenario_tree(instance, T)`** — l'arbre lui-même. La récursion (1g) y est évaluée
  numériquement, donc $\zeta$ n'est pas une variable du programme étendu.

> ⚠️ La comparaison à `bin_extensive_neutral` (deux-étages) ne vaut qu'en `force = 0`, où
> les deux modèles d'aléa coïncident. En stochastique, la référence est
> `multistage_extensive`.

---

## Incertitude : processus ARMA

L'erreur relative de prévision de demande nette suit le processus ARMA de la §2.1, avec les
paramètres calibrés de l'article PSCC, éq. (40) :

$$\eta_t = 0{,}67\,\eta_{t-1} + \xi_t + 0{,}117\,\xi_{t-1}, \qquad \xi_t \sim \mathcal N(0,\ 2{,}9\cdot10^{-4}\sigma)$$

appliqué **indépendamment à chaque nœud**. Le processus est rendu markovien par
l'augmentation d'état de la §2.1 : $\zeta_t = (\eta_t, \xi_t)$ pour un ARMA(1,1), avec

$$\zeta_t = \Psi\,\zeta_{t-1} + E\,\xi_t, \qquad \eta_t = S_\eta\,\zeta_t$$

$\zeta$ est une **variable d'état** du graphe, déclarée à tous les nœuds, et les coupes
portent le terme $\tilde\delta_t^\top\zeta$ des éq. (10), (12) et (40b).

> ⚠️ **Collision de notation entre les deux articles.** Ce que PSCC (40) note $\eta_t$ est
> l'**innovation**, notre $\xi_t$ ; ce que PSCC note $\varepsilon_t$ est l'**erreur**, notre
> $\eta_t$. Le code suit la notation MSUC.

### Régler le support

PSCC spécifie une innovation **gaussienne continue**, mais l'Hypothèse 1 exige un support
**fini**. La discrétisation n'est donnée dans aucun des deux articles ; elle est ici un
paramètre du parsing :

```julia
instance = MSUC.parse_nc4("Data/T-Ramp/10_0_1_w.nc4", optimizer, 24;
                          sigma          = 1.0,              # σ de l'éq. (40)
                          S              = 5,                # taille du support
                          discretization = :gauss_hermite)   # ou :equiprobable
```

Les défauts sont les constantes `ARMA_SUPPORT_SIZE` et `ARMA_DISCRETIZATION` de
[`src/UC/Struct/ARMA.jl`](src/UC/Struct/ARMA.jl). Le nombre d'aléas par étape vaut
exactement `S` : une seule innovation scalaire est tirée et appliquée à tous les nœuds
(`common_innovation = true`), sans quoi il faudrait $S^{|\mathcal N|}$ combinaisons.

**Les deux schémas**, tous deux de moyenne et variance exactes :

| | $S=5$, nœud extrême | $p_{\min}$ | moments exacts |
|---|---|---|---|
| `:gauss_hermite` (défaut) | $\pm 2{,}86\,\sigma_\xi$ | 1,1 % | jusqu'à l'ordre $2S-1$ |
| `:equiprobable` | $\pm 1{,}48\,\sigma_\xi$ | 20 % | moyenne et variance |

- **Gauss-Hermite** (Golub-Welsch : valeurs propres de la matrice de Jacobi d'Hermite,
  poids = carrés des premières composantes des vecteurs propres). Optimal pour une
  **espérance** — c'est le cas risque-neutre du papier. Revers : les nœuds extrêmes portent
  des probabilités infimes ($8{,}6\cdot10^{-10}$ à $S=15$), donc les queues comptent peu.
- **Équiprobable** : chaque scénario pèse $1/S$, nœud = moyenne conditionnelle de sa strate.
  À préférer pour une **mesure de risque** type AVaR, où les queues doivent être peuplées.

> Le schéma équiprobable brut **sous-estime la variance** (0,897 à $S=5$) : la dispersion
> intra-strate est perdue. Les nœuds sont donc renormalisés par défaut
> (`match_variance = true`) pour que la variance discrète vaille exactement
> $2{,}9\cdot10^{-4}\sigma$. Sans effet sur Gauss-Hermite, déjà exacte.

`force` (argument de `PolicyGraph`) multiplie $\eta_t$ dans le bilan (1h) : `force = 0`
neutralise l'aléa et, comme $\tilde\delta \equiv 0$ par récurrence, redonne **exactement**
le problème déterministe — c'est ce qui rend le protocole de validation reproductible.

---

## Correspondance article ↔ code

C'est la table de référence pour lire le code.

### Données et ensembles

| Article | Code | Où |
|---|---|---|
| $\mathcal J$ — groupes | `instance.Thermalunits` | `UC/Struct/Instance.jl` |
| $\mathcal N$ — nœuds du réseau | `instance.Next`, `instance.Demandbus` | idem |
| $\mathcal L$ — lignes | `instance.Lines` | idem |
| $T$ — horizon | `instance.TimeHorizon` | idem |
| $\mathcal T_j$ — intervalles admissibles | `unit.intervals`, vecteurs `[a, b]` | `set_intervals` |
| $\underline P_j,\ \overline P_j$ | `unit.MinPower`, `unit.MaxPower` | |
| $R^U_j,\ R^D_j$ | `unit.DeltaRampUp`, `unit.DeltaRampDown` | |
| $S^U_j,\ S^D_j$ | `unit.MinPower` (ce modèle pose $S^U = S^D = \underline P$) | |
| $y_{j,0}$ | `unit.InitialPower` | |
| $\xi_t$ — aléa | `forecast_error_support(instance)` | `UC/builder.jl` |
| $p_{t,\xi}$ | `probability` | |

### Variables

| Article | Code |
|---|---|
| $x_j = (u_j, v_j, w_j)$ | `is_on`, `start_up`, `start_down` (variables d'état) |
| $y_{j,t}$ | `power[j]` (variable d'état), `node.states_power` |
| $\gamma_{j,a,b}$ | `gamma[j,a,b]`, `node.states_gamma` |
| $\tilde y_{j,a,b,t}$ | jamais matérialisé : identité $\tilde y = u_{a,b,t}\, y_{j,t}$, cf. (41g) |
| $f_{n,m,t},\ \theta_{n,t}$ | `flow`, `θ` |

### Section 4 — modèle capacité + rampes (`UC/Unit/Thermal_unit.jl`)

| Article | Code |
|---|---|
| $u_{a,b,t}$ — éq. (15a) | `is_on_interval(a, b, t)` |
| $\underline p_{j,a,b,t}$ | `p_lower(unit, a, b, t)` |
| $\bar p_{j,a,b,t}$ — inclut (46) | `p_upper(unit, a, b, t, T)` |
| $R^U_{j,a,b,t}$ — membre droit de (45b) | `ramp_up(unit, a, b, t)` |
| $R^D_{j,a,b,t}$ — membre droit de (45c) | `ramp_down(unit, a, b, t)` |
| $\tilde y_{j,a,b,0}$ — éq. (17) | `initial_interval_output(unit, a)` |
| $q_{j,a,b,t}$ — éq. (49) | `reference_output(unit, a, b, t, T)` |
| $\hat Q_{j,a,b,t}$ et son support (48)–(49) | `local_support(unit, a, b, t, T, y_prev, z)` → `(h, β)` |

### Coupes (`plugins/bellman_functions.jl`)

| Article | Code |
|---|---|
| coupe (39) | `IntervalCut` |
| $\tilde\alpha_t$ — (40a) | `IntervalCut.alpha` |
| $\tilde\tau_{j,a,b,t}$ — (40c) | `IntervalCut.tau[(j,a,b)]` |
| $\tilde\beta_{j,a,b,t}$ — (40d) | `IntervalCut.beta[(j,a,b)]` |
| $V^{\text{ext}}_{t}$ — (27) | `node.interval_cuts` |
| $\hat z_{j,a,b,\xi}$ — Prop. 4 | `z` |
| $(\hat h, \hat\beta)$ — (34), (49) | valeur de retour de `local_support` |

### Multiplicateurs

**Attention à ce point** : le code nommait `μ` ce qui est le $\sigma$ de l'article.

| Article | Code | Nature |
|---|---|---|
| $\sigma \in \Delta_k$ — (41g) | `σ` | duaux des lignes d'épigraphe, un par coupe |
| $\nu_{n,t}$ — (41f) | `ν` | duaux du bilan par nœud |
| $\mu_{j,a,b}$ — (42b) | *jamais matérialisé* | la forme close (49) donne directement le support |

---

## Comment l'Algorithme 2 se lit dans le code

| Alg. 2 | Code |
|---|---|
| l. 4–5 — résoudre (26), stocker $(\gamma^k, x^k)$ et $L^k_{\text{ext}}$ | `forward_pass`, nœud 1 ; `lower_bound` |
| l. 9 — résoudre (41) à chaque étape | `solve_subproblem_extended` |
| l. 10 — état d'intervalle | identité $\tilde y = u_{a,b,t} y_{j,t}$, pas de variable dédiée |
| l. 15–16 — duaux $(\hat\nu, \hat\sigma)$ | `get_dual_solution_extended` |
| l. 17 — $\hat z_{j,a,b,\xi}$ | `refine_bellman_function_extended` |
| l. 19–28 — support local | `local_support` |
| l. 29, (36) — $\hat\tau = \hat h + \tilde\tau^\sigma$ | `τ_ab` |
| l. 33–34, (39)–(40) — moyenne et ajout | `IntervalCut(α, τ, β, cstr)` |

**Le point le plus subtil.** Les coupes sont stockées avec leurs coefficients *par intervalle*,
puis la ligne d'épigraphe est **réécrite** au planning $\gamma$ courant à chaque passe avant
(`set_normalized_rhs` / `set_normalized_coefficient` dans `plugins/forward_passes.jl`). C'est
ainsi qu'on obtient (41g) sans porter les produits $\gamma \cdot \tilde y$ comme variables.

**Le piège à ne pas réintroduire.** La Prop. 4 annule $\hat h$ et $\hat\beta$ sur un intervalle
inactif, mais l'éq. (36) ajoute ensuite $\tilde\tau^\sigma$ : la **propagation des coupes futures
a lieu pour tous les intervalles**, y compris inactifs. Un intervalle qui démarre plus tard est
inactif à l'étape courante mais porte bien une valeur future. L'omettre rend la borne invalide
(mesuré : 1 905 374 pour un optimum exact de 1 850 053).

---

## Protocole de validation

Avec `force = 0.0` le problème devient **déterministe**, donc l'optimum exact est calculable par
la formulation extensive. La borne SDDP doit alors rester **sous** cet optimum, croître, et
converger dessus. C'est le seul test qui tranche.

```julia
r = MSUC.bin_extensive_neutral(instance; silent=true, force=0.0, S=1, batch=1,
                               gap=1e-6, timelimit=120)
opt = r[3]          # 1 850 053.2 sur Data/T-Ramp/10_0_1_w.nc4

model = MSUC.PolicyGraph(MSUC.subproblem_builder_UC_extended, instance, 0.0,
                         MSUC.LinearGraph(25), MSUC.ExtendedFormulation;
                         sense=:Min, lower_bound=0.0, optimizer=optimizer)
MSUC.train(model; iteration_limit = 10)
```

Référence (instance `10_0_1_w.nc4`, optimum exact **1 850 053,2**) :

| itération | 1 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|
| borne | 58 914 | 383 739 | 1 803 230 | 1 807 036 | **1 820 387** (98,4 %) |

> ⚠️ **Ne comparez pas** la borne étendue à celle de la 3-bin pour juger de la validité : tant
> que la 3-bin n'a pas convergé, l'étendue est légitimement bien plus haute — c'est exactement
> ce que l'article prédit. Seule la comparaison à un optimum exact est concluante.

> ⚠️ En stochastique (`force > 0`) les tirages ne sont **pas** reproductibles (pas de graine),
> donc deux exécutions donnent des bornes différentes. Utilisez `force = 0` pour toute
> comparaison A/B de code.

---

## Organisation du dépôt

```
src/
  MSUC.jl                      module principal
  UC/
    Struct/Instance.jl         ThermalUnit, Line, Instance
    Struct/parsing.jl          parse_nc4 (SMS++), parse_IEEE (118-bus)
    Unit/Thermal_unit.jl       Section 4 : T_j, p, R^U, R^D, q, local_support
    Unit/Line.jl               réseau
    builder.jl                 sous-problèmes d'étape des deux formulations
    extensive_formulation.jl   formulation extensive 3-bin (vérité terrain)
  plugins/
    bellman_functions.jl       génération des coupes intervalles (Sect. 3.3)
    forward_passes.jl          passe avant + réinstanciation des coupes
    duality_handlers.jl        récupération de (σ, ν) et ContinuousConicDuality
    risk_measures.jl           mesures de risque (hérité SDDP.jl)
    sampling_schemes.jl        échantillonnage des scénarios (hérité)
    stopping_rules.jl          critères d'arrêt (hérité)
    backward_sampling_schemes.jl, parallel_schemes.jl, headers.jl  (hérités)
  algorithm.jl                 boucle SDDP, passe arrière
  user_interface.jl            PolicyGraph, Node, IntervalCut
  print.jl, JuMP.jl            journal et surcharges JuMP
Data/
  T-Ramp/                      41 instances SMS++ (.nc4)
  IEEE/118_syst/               instance IEEE 118-bus
  Uncertainty/                 scénarios de demande nette
```

Le **Benders deux-étages** de l'article [12] (`benders_callback`, `add_cut_SP`, l'heuristique
duale, les options `UCOptions`) a été retiré : c'est un autre algorithme, hors du périmètre de
ce papier. Seule `bin_extensive_neutral` a été conservée, car elle fournit la vérité terrain.

Le code mort hérité de SDDP.jl a été supprimé : `Inner.jl`, `Experimental.jl`, `MSPFormat.jl`,
`biobjective.jl`, `binary_expansion.jl`, `cyclic.jl`, `alternative_forward.jl`,
`deterministic_equivalent.jl`, `modeling_aids.jl` et tout `visualization/`. Les dépendances
correspondantes (`HTTP`, `SHA`, `XLSX`, `RecipesBase`, `Plots`) sont sorties de `Project.toml`.

Trois fonctionnalités de SDDP.jl ont également été retirées, **inatteignables** avec un
`LinearGraph` :

| | à quoi ça servait | pourquoi inatteignable |
|---|---|---|
| `belief_state` | graphes partiellement observables (POMDP) : loi de probabilité sur la partition de nœuds, mise à jour bayésienne, terme de coupe concave $\langle b,\mu\rangle$ | `belief_partition` n'est rempli que par `add_ambiguity_set` ; `LinearGraph` le laisse vide |
| `objective_state` | objectif dépendant d'un prix exogène markovien, avec interpolation et terme $\mu^\top y$ | positionné seulement par `add_objective_state`, que les builders n'appellent jamais |
| cycles | horizon infini : `terminated_due_to_cycle`, `starting_states`, `cycle_discretization_delta` | un chemin ne boucle pas |
| persistance de politique | `write_cuts_to_file` / `read_cuts_from_file` : sauver une politique entraînée pour la recharger sans réentraîner | ne sérialisent que les coupes génériques, **pas** les `IntervalCut` ; le dump était vide en formulation étendue, et `read_` jamais appelée |
| handlers SDDiP | `LagrangianDuality`, `StrengthenedConicDuality`, `BanditDuality`, `FixedDiscreteDuality` et `local_improvement_search.jl` : duaux de sous-problèmes **entiers** | §1 du papier — seule la racine est non convexe, chaque étage est un LP, donc `ContinuousConicDuality` est exacte |

> **`Plots` n'est plus une dépendance du projet.** Si vos notebooks font `using Plots`,
> rajoutez-le : `julia --project=. -e 'using Pkg; Pkg.add("Plots")'`.

---

## Ce qui n'est pas implémenté

| Article | État |
|---|---|
| Processus ARMA $\zeta_t = \Psi\zeta_{t-1} + E\xi_t$ — (1g) | **absent**. L'aléa est i.i.d. entre étapes ; tout le volet $\delta/\lambda$ des coupes (10), (12), (37), (40b) n'existe pas. |
| §3.4 — dominance par étape et par itération | vide dans l'article |
| Étude numérique IEEE 118-bus | `parse_IEEE` réparé mais non validé de bout en bout |
| Horizon $T \neq 24$ | le builder et le parsing prennent `T`, mais les instances fournies sont à 24 pas |

Autres limites connues :

- **Pas de graine aléatoire** : les runs stochastiques ne sont pas reproductibles.
- **Coût en $O(k)$ par intervalle et par aléa** dans le calcul de $\hat z$ (`plugins/bellman_functions.jl`),
  donc entraînement en $O(k^2)$. Le terme $\tilde\tau^\sigma$ est déjà factorisé via `σ_mean` ;
  reste $\hat z$, qui se ramènerait à un produit matrice-vecteur en indexant les intervalles de
  façon dense.
- **Pas de tests automatisés.**

Le suivi détaillé des corrections et des points ouverts est dans [`CORRECTIONS.md`](CORRECTIONS.md).
