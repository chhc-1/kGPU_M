module kGPU_M

import CUDA
import Adapt
import FFTW
import AbstractFFTs



include("typedefs.jl")
include("Fourier_array.jl")
include("solver.jl")
include("flow_measures.jl")
include("convection.jl")
include("run_simulation.jl")
include("Adapt_defs.jl")


end # module kGPU_M
