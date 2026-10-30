#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v. 2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

struct InstanceFactory{T}
    args::Any
    kwargs::Any
    InstanceFactory{T}(args...; kwargs...) where {T} = new{T}(args, kwargs)
end

mutable struct BellmanFunction
    theta::JuMP.VariableRef
end

function BellmanFunction(; lower_bound = -Inf, upper_bound = Inf)
    return InstanceFactory{BellmanFunction}(;
        lower_bound = lower_bound,
        upper_bound = upper_bound,
    )
end
