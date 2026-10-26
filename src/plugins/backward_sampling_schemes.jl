#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors and contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

struct CompleteSampler <: AbstractBackwardSamplingScheme end

sample_backward_noise_terms(::CompleteSampler, node) = node.noise_terms

struct MonteCarloSampler <: AbstractBackwardSamplingScheme
    number_of_samples::Int
end

function sample_backward_noise_terms(sampler::MonteCarloSampler, node::Node)
    prob = 1 / sampler.number_of_samples
    return [
        Noise(sample_noise(node.noise_terms), prob) for
        _ in 1:sampler.number_of_samples
    ]
end
