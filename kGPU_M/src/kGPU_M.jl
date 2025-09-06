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

#=
function init_F_array(solver1::solver, F_arr::Fourier_array)    
    F_arr.pad = repeat([convert(Float64, 0.0)], solver1.N1_padded, solver1.N2_padded);
    F_arr.hat = repeat([0.0 + 0.0*im], solver1.N1r, solver1.N2);
    F_arr.hat_padded = repeat([0.0 + 0.0*im], solver1.N1r_padded, solver1.N2_padded);
end;
=#

end # module kGPU_M
