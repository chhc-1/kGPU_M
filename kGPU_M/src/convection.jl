import CUDA

include("typedefs.jl")
include("solver.jl")


#function calc_conv(solver1::solver, ω_hat::CUDA.CuArray{ComplexF64, 2}, return_arr::CUDA.CuArray{ComplexF64, 2})
function calc_conv(solver1::solver, ω_hat::arr_type, return_arr::arr_type)

    solver1.dωdx.hat_padded .= im .* solver1.kx .* ω_hat .* solver1.mask;
    solver1.dωdy.hat_padded .= im .* solver1.ky .* ω_hat .* solver1.mask;

    solver1.u.hat_padded .= solver1.dωdy.hat_padded ./ solver1.kxy2;
    solver1.v.hat_padded .= -solver1.dωdx.hat_padded ./ solver1.kxy2;
    
    CUDA.CUBLAS.mul!(solver1.u.pad, solver1.irfft_plan_padded, solver1.u.hat_padded);
    CUDA.CUBLAS.mul!(solver1.v.pad, solver1.irfft_plan_padded, solver1.v.hat_padded);
    CUDA.CUBLAS.mul!(solver1.dωdx.pad, solver1.irfft_plan_padded, solver1.dωdx.hat_padded);
    CUDA.CUBLAS.mul!(solver1.dωdy.pad, solver1.irfft_plan_padded, solver1.dωdy.hat_padded);

    solver1.udωdx.pad .= solver1.u.pad .* solver1.dωdx.pad;
    solver1.vdωdy.pad .= solver1.v.pad .* solver1.dωdy.pad;

    CUDA.CUBLAS.mul!(solver1.udωdx.hat_padded, solver1.rfft_plan_padded, solver1.udωdx.pad);
    CUDA.CUBLAS.mul!(solver1.vdωdy.hat_padded, solver1.rfft_plan_padded, solver1.vdωdy.pad);
    
    return_arr .= (solver1.udωdx.hat_padded .+ solver1.vdωdy.hat_padded);# .* solver1.pad_scaling_factor;
    return nothing;
end;
