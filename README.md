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

La colonne `bound` du journal est la borne inférieure : la valeur optimale du problème racine (26)
avec l'approximation courante, c'est-à-dire la ligne 5 de l'Algorithme 2.

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
    Struct/tools.jl            utilitaires du Benders deux-étages
    Unit/Thermal_unit.jl       Section 4 : T_j, p, R^U, R^D, q, local_support
    Unit/Line.jl               réseau
    builder.jl                 sous-problèmes d'étape des deux formulations
    Optimizer/                 Benders deux-étages de l'article [12] — autre algorithme
  plugins/
    bellman_functions.jl       génération des coupes intervalles (Sect. 3.3)
    forward_passes.jl          passe avant + réinstanciation des coupes
    duality_handlers.jl        récupération de (σ, ν)
    ...                        hérités de SDDP.jl
  algorithm.jl                 boucle SDDP, passe arrière
  user_interface.jl            PolicyGraph, Node, IntervalCut
Data/
  T-Ramp/                      41 instances SMS++ (.nc4)
  IEEE/118_syst/               instance IEEE 118-bus
  Uncertainty/                 scénarios de demande nette
```

`src/UC/Optimizer/` implémente le **Benders deux-étages** de l'article [12] (`benders_callback`,
`bin_extensive_neutral`). C'est un autre algorithme, conservé pour les comparaisons et pour la
vérité terrain du protocole ci-dessus.

Sont hérités de SDDP.jl et **non utilisés** par ce travail : `Inner.jl`, `Experimental.jl`,
`MSPFormat.jl`, `biobjective.jl`, `binary_expansion.jl`, `cyclic.jl`, `alternative_forward.jl`,
`deterministic_equivalent.jl`, `visualization/`.

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
