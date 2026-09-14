#  Copyright (c) 2017-25, Oscar Dowson and MSUC.jl contributors
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

module MSUC

import Reexport
Reexport.@reexport using JuMP

import Distributed
import LinearAlgebra
import JSON
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

# Modelling interface.
include("user_interface.jl")

# Default definitions for MSUC related modular utilities.
include("plugins/headers.jl")

# Tools for overloading JuMP functions
include("JuMP.jl")

# Printing utilities.
include("print.jl")

# The core MSUC code.
include("algorithm.jl")


include("UC/builder.jl")

# Specific plugins.
include("plugins/risk_measures.jl")
include("plugins/sampling_schemes.jl")
include("plugins/bellman_functions.jl")
include("plugins/stopping_rules.jl")
include("plugins/local_improvement_search.jl")
include("plugins/duality_handlers.jl")
include("plugins/parallel_schemes.jl")
include("plugins/backward_sampling_schemes.jl")
include("plugins/forward_passes.jl")

# Visualization related code.

# Other solvers.

# Inner approximation


end
