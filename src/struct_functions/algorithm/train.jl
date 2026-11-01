#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

struct Log
    iteration::Int
    bound::Float64
    incumbent::Float64
    time::Float64
    total_solves::Int
    serious_numerical_issue::Bool
end

struct TrainingResults
    status::Symbol
    log::Vector{Log}
end
