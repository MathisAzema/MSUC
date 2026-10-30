#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

struct BackwardPassItems
    duals::Vector{StageDuals}
    probability::Vector{Float64}
    objectives::Vector{Float64}
    function BackwardPassItems()
        return new(StageDuals[], Float64[], Float64[])
    end
end
