# =============================================================================
#  Processus ARMA(p,q) de l'erreur de prevision de demande nette (Sect. 2.1).
#
#  Notation du papier MSUC :
#     eta_t = sum_i Phi_i eta_{t-i} + xi_t + sum_k Theta_k xi_{t-k}
#  avec eta_t l'erreur RELATIVE (commune a tous les noeuds) et xi_t l'innovation, a support fini
#  (Hypothese 1). Le processus devient markovien apres augmentation de l'etat :
#     zeta_t = (eta_t, .., eta_{t-p+1}, xi_t, .., xi_{t-q+1})  in R^{p+q}
#     zeta_t = Psi zeta_{t-1} + E xi_t                          (1g)
#     eta_t  = S_eta zeta_t
#
#  ATTENTION aux notations croisees avec l'article PSCC, eq. (40), d'ou viennent
#  les parametres calibres : ce que PSCC note eta (erreur) est notre eta, mais ce
#  que PSCC note eta_t ~ N(0, .) est notre INNOVATION xi_t. La correspondance est
#     PSCC epsilon <-> MSUC eta   (erreur relative)
#     PSCC eta     <-> MSUC xi    (innovation)
# =============================================================================

"Coefficients AR de l'ARMA(1,1) calibre sur la demande francaise, PSCC eq. (40)."
const ARMA_PHI = [0.67]

"Coefficients MA de l'ARMA(1,1) calibre, PSCC eq. (40)."
const ARMA_THETA = [0.117]

"""
Variance de base de l'innovation, PSCC eq. (40) : `xi_t ~ N(0, 2.9e-4)`.

Le second argument de `N(.,.)` est lu comme une VARIANCE : l'ecart-type de l'innovation vaut
sqrt(2.9e-4) ~ 1.7 %, ordre de grandeur d'une erreur de prevision de demande a un jour. Le parametre
`sigma` multiplie l'ECART-TYPE : la variance vaut `2.9e-4 * sigma^2`. (Dans PSCC, le parametre
`sigma` multiplie la variance : c'est le carre de celui d'ici.)
"""
const ARMA_INNOVATION_VARIANCE = 2.9e-4

"""Taille par defaut du support de l'innovation xi_t (Hypothese 1)."""
const ARMA_SUPPORT_SIZE = 5

"""Schema de discretisation par defaut : `:gauss_hermite` ou `:equiprobable`."""
const ARMA_DISCRETIZATION = :gauss_hermite

"""
    gauss_hermite(S) -> (noeuds, poids)

Quadrature de Gauss-Hermite a `S` points pour la loi N(0,1), par Golub-Welsch : les
noeuds sont les valeurs propres de la matrice de Jacobi de la recurrence des polynomes
d'Hermite probabilistes, les poids les carres des premieres composantes des vecteurs
propres. Exacte pour les polynomes de degre <= 2S-1, donc les deux premiers moments
de la loi discretisee sont exacts.
"""
function gauss_hermite(S::Int)
    S <= 1 && return ([0.0], [1.0])
    J = LinearAlgebra.SymTridiagonal(zeros(S), [sqrt(k) for k in 1:(S-1)])
    F = LinearAlgebra.eigen(J)
    return collect(F.values), [abs2(F.vectors[1, i]) for i in 1:S]
end

"""
    standard_normal_quantile(p)

Quantile de N(0,1) par l'approximation rationnelle d'Acklam (erreur relative < 1.2e-9).
Evite une dependance a SpecialFunctions pour le seul schema equiprobable.
"""
function standard_normal_quantile(p::Float64)
    @assert 0 < p < 1
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow
        q = sqrt(-2 * log(p))
        return (((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
               ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)
    elseif p <= phigh
        q = p - 0.5; r = q * q
        return (((((a[1]*r+a[2])*r+a[3])*r+a[4])*r+a[5])*r+a[6]) * q /
               (((((b[1]*r+b[2])*r+b[3])*r+b[4])*r+b[5])*r+1)
    else
        q = sqrt(-2 * log(1 - p))
        return -(((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
                ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)
    end
end

"""
    equiprobable_normal(S) -> (noeuds, poids)

Discretisation de N(0,1) en `S` strates EQUIPROBABLES : chaque scenario a la probabilite
1/S et pour noeud la moyenne conditionnelle de sa strate,
    m_i = S (phi(z_{i-1}) - phi(z_i)),   z_i = Phi^{-1}(i/S),
ou phi est la densite standard. Preserve exactement la moyenne, mais pas les moments
d'ordre superieur : la variance est sous-estimee (effet de regroupement intra-strate).

A preferer a Gauss-Hermite si l'on veut des scenarios de poids egal ou une meilleure
representation des queues, par exemple pour une mesure de risque type AVaR.
"""
function equiprobable_normal(S::Int)
    S <= 1 && return ([0.0], [1.0])
    φ(z) = exp(-z^2 / 2) / sqrt(2π)
    z = [standard_normal_quantile(i / S) for i in 1:(S-1)]
    dens = vcat(0.0, φ.(z), 0.0)
    return [S * (dens[i] - dens[i+1]) for i in 1:S], fill(1 / S, S)
end

"""
    innovation_support(S, sd; discretization, match_variance) -> (noeuds, poids)

Support fini de l'innovation d'ecart-type `sd`.

* `:gauss_hermite` : les 2S-1 premiers moments sont exacts. Optimal pour une esperance,
  mais les noeuds extremes portent des probabilites tres faibles (8.6e-10 a S=15), donc
  les queues sont peu representees en pratique.
* `:equiprobable` : tous les scenarios ont le poids 1/S et les noeuds restent proches du
  centre. A preferer pour une mesure de risque type AVaR.

`match_variance` renormalise les noeuds pour que la variance discrete vaille exactement
`sd^2`. Sans cela le schema equiprobable la SOUS-ESTIME nettement (0.897 sd^2 a S=5,
0.976 a S=15), la dispersion intra-strate etant perdue. Sans effet sur Gauss-Hermite,
dont la variance est deja exacte.
"""
function innovation_support(
    S::Int,
    sd::Float64;
    discretization::Symbol = ARMA_DISCRETIZATION,
    match_variance::Bool = true,
)
    nodes, weights = if discretization === :gauss_hermite
        gauss_hermite(S)
    elseif discretization === :equiprobable
        equiprobable_normal(S)
    else
        error("discretization inconnue : $discretization (:gauss_hermite ou :equiprobable)")
    end
    if match_variance && S > 1
        m2 = sum(weights .* nodes .^ 2)
        m2 > 0 && (nodes ./= sqrt(m2))
    end
    return sd .* nodes, weights
end

"""
    ARMAProcess(nbus; sigma, Phi, Theta, S, discretization)

Construit le processus et ses matrices compagnons pour `nbus` noeuds.

Un SEUL facteur ARMA(p,q) scalaire pilote la demande de tous les noeuds : l'erreur relative `eta_t`
est la meme partout (`S_eta` ne fait que la recopier sur les `nbus` noeuds), comme une charge nette
de reseau repartie au prorata des charges (Zou, Ahmed & Sun, 2018). L'etat `zeta` est donc de
dimension `p + q`, quel que soit le reseau, et le nombre d'aleas par etape vaut exactement `S`
(sinon il faudrait `S^|N|` combinaisons).

`sigma` multiplie l'ecart-type de l'innovation : sa variance vaut `2.9e-4 * sigma^2` (PSCC eq. (40)
multiplie la variance par son propre parametre, qui est donc ici `sigma^2`). `sigma = 1` correspond a
l'erreur de prevision de demande seule, des valeurs plus grandes a l'incertitude additionnelle de la
production eolienne, et `sigma = 0` au probleme deterministe. Multiplier `sigma` par `c` a le meme effet
que `force = c` dans `PolicyGraph`.

`S` est la taille du support de l'innovation et `discretization` son schema
(`:gauss_hermite` ou `:equiprobable`).
"""
function ARMAProcess(
    nbus::Int;
    sigma::Float64 = 1.0,
    Phi::Vector{Float64} = ARMA_PHI,
    Theta::Vector{Float64} = ARMA_THETA,
    variance::Float64 = ARMA_INNOVATION_VARIANCE * sigma^2,
    S::Int = ARMA_SUPPORT_SIZE,
    discretization::Symbol = ARMA_DISCRETIZATION,
    match_variance::Bool = true,
)
    p, q = length(Phi), length(Theta)
    n = p + q

    Psi = zeros(n, n)
    # Ligne 1 : eta_t = sum Phi_i eta_{t-i} + sum Theta_k xi_{t-k}  (+ xi_t via E)
    Psi[1, 1:p] .= Phi
    Psi[1, (p+1):(p+q)] .= Theta
    # Lignes 2..p et p+2..p+q : decalages des retards
    for i in 2:p
        Psi[i, i-1] = 1.0
    end
    for k in 2:q
        Psi[p+k, p+k-1] = 1.0
    end
    # E : l'innovation entre dans eta_t et est stockee dans la composante p+1
    E = zeros(n, 1)
    E[1, 1] = 1.0
    q >= 1 && (E[p+1, 1] = 1.0)

    # eta_t, recopie sur tous les noeuds
    S_eta = zeros(nbus, n)
    S_eta[:, 1] .= 1.0

    nodes, weights = innovation_support(
        S,
        sqrt(variance);
        discretization = discretization,
        match_variance = match_variance,
    )
    support = [[z] for z in nodes]
    return ARMAProcess(Psi, E, S_eta, support, weights, zeros(n))
end

"Dimension de l'etat d'incertitude zeta_t."
zeta_dimension(a::ARMAProcess) = size(a.Psi, 1)
