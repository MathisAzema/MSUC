#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

module MSUC

import Reexport
Reexport.@reexport using JuMP

import LinearAlgebra
import MutableArithmetics
import Printf
import Random
import Statistics
import TimerOutputs

using Gurobi
using NCDatasets
using CSV
using DataFrames

export @stageobjective

#UC 
const SHEDDING_COST=700.0
const CURTAILEMENT_COST=700.0

include("UC/Struct/ARMA.jl")
include("UC/Struct/Instance.jl")
include("UC/Unit/Thermal_unit.jl")
include("UC/Unit/Line.jl")
include("UC/Struct/parsing.jl")
include("UC/extensive_formulation.jl")
include("UC/multistage_extensive.jl")

include("interface/user_interface.jl")
include("interface/headers.jl")
include("interface/JuMP.jl")
include("interface/print.jl")

include("algorithm/train.jl")
include("algorithm/backward_pass.jl")

include("UC/builder.jl")

include("algorithm/sampling_schemes.jl")
include("algorithm/bellman_functions.jl")
include("algorithm/stopping_rules.jl")
include("algorithm/duality_handlers.jl")
include("algorithm/backward_sampling_schemes.jl")
include("algorithm/forward_passes.jl")

include("interface/simulate.jl")

end
