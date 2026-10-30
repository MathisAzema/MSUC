#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v. 2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

function sample_noise(noise_terms::Vector{<:Noise})
    if length(noise_terms) == 0
        return nothing
    end
    cumulative_probability = sum(noise.probability for noise in noise_terms)
    if cumulative_probability > 1.0 + 1e-6
        error("Cumulative probability cannot be greater than 1.0.")
    end
    rnd = rand() * cumulative_probability
    for noise in noise_terms
        rnd -= noise.probability
        if rnd <= 0.0
            return noise.term
        end
    end
    return error(
        "Internal MSUC error: unable to sample noise from $(noise_terms)",
    )
end

"""
    sample_scenario(model)

Tire une realisation de l'alea par noeud (le noeud k est a la profondeur k, la racine a un alea
nul) : `sample_scenario(model)[k]` est le terme tire pour le noeud k.
"""
function sample_scenario(model::PolicyGraph)
    return [sample_noise(model[k].noise_terms) for k in 1:length(model.nodes)]
end
