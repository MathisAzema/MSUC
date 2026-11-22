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
using XZ_jll
using DataFrames

export @stageobjective

const SHEDDING_COST=700.0
const CURTAILEMENT_COST=700.0

# Struct definitions, in dependency order.
include("struct_functions/instance/arma_process.jl")
include("struct_functions/instance/instance.jl")
include("struct_functions/model/schedule.jl")
include("struct_functions/model/state_variable.jl")
include("struct_functions/model/bellman_function.jl")
include("struct_functions/tools/sampling.jl")
include("struct_functions/cuts/cuts.jl")
include("struct_functions/cuts/duals.jl")
include("struct_functions/algorithm/train.jl")
include("struct_functions/model/build_policy_graph.jl")
include("struct_functions/algorithm/one_iteration.jl")
include("struct_functions/algorithm/backward_pass.jl")
include("struct_functions/tools/print_log.jl")

# Functions. Macros (`@stageobjective`) must be defined before first use.
include("functions/instance/arma_process.jl")
include("functions/instance/parse_instance.jl")
include("functions/instance/thermal_unit.jl")
include("functions/instance/unit_constraints.jl")
include("functions/reference/multistage_extensive.jl")
include("functions/model/state_variable.jl")
include("functions/model/build_policy_graph.jl")
include("functions/model/schedule.jl")
include("functions/model/build_subproblems.jl")
include("functions/model/build_root_problem.jl")
include("functions/model/build_stage_problem.jl")
include("functions/model/bellman_function.jl")
include("functions/tools/sampling.jl")
include("functions/cuts/duals.jl")
include("functions/cuts/cuts.jl")
include("functions/cuts/refine_cuts.jl")
include("functions/algorithm/solve_subproblem.jl")
include("functions/algorithm/solve_first_stage_problem.jl")
include("functions/algorithm/interval_states.jl")
include("functions/algorithm/forward_pass.jl")
include("functions/algorithm/backward_pass.jl")
include("functions/algorithm/one_iteration.jl")
include("functions/algorithm/train.jl")
include("functions/tools/simulate.jl")
include("functions/tools/print_log.jl")
include("functions/run/model.jl")
include("functions/run/record.jl")
include("functions/run/policy.jl")
include("functions/run/jobs.jl")

end
