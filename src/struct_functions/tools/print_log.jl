#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

struct CoefficientRanges
    matrix::Vector{Float64}
    objective::Vector{Float64}
    bounds::Vector{Float64}
    rhs::Vector{Float64}
    function CoefficientRanges()
        return new([Inf, -Inf], [Inf, -Inf], [Inf, -Inf], [Inf, -Inf])
    end
end
