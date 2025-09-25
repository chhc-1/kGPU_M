import CUDA


include("typedefs.jl")
include("Fourier_array.jl")
include("solver.jl")
include("convection.jl")
include("flow_measures.jl")

function update(solver1::solver)
    calc_conv(solver1, solver1.ω_hat_prev, solver1.conv_prev);
    solver1.ω_hat_intermediate .= solver1.Ginv_1 .* (solver1.ω_hat_prev .+ solver1.dt .* (-0.5 .* solver1.G .* solver1.ω_hat_prev .- solver1.conv_prev .+ solver1.source_hat));
    
    calc_conv(solver1, solver1.ω_hat_intermediate, solver1.conv_intermediate);

    solver1.ω_hat_new .= solver1.Ginv_1 .* (solver1.ω_hat_prev .+ solver1.dt .* (-0.5 .* solver1.G .* solver1.ω_hat_prev .- 0.5 .* (solver1.conv_prev .+ solver1.conv_intermediate) .+ solver1.source_hat));

    return nothing;
end

function iter(solver1::solver, iteration::Int64)
    update(solver1);

    view(solver1.ω_arr, 1:solver1.N1r_padded, 1:solver1.N2_padded, iteration) .= view(solver1.ω_hat_new, 1:solver1.N1r_padded, 1:solver1.N2_padded);

    CUDA.copyto!(solver1.ω_hat_prev, solver1.ω_hat_new);

    #CUDA.CUBLAS.mul!(solver1.ω_temp, solver1.irfft_plan_padded, solver1.ω_hat_new);
    
    return nothing;
end;

function iter(solver1::solver, measures::flow_measures, iteration::Int64)
    update(solver1);

    CUDA.copyto!(solver1.ω_hat_prev, solver1.ω_hat_new);

    CUDA.CUBLAS.mul!(solver1.ω_temp, solver1.irfft_plan_padded, solver1.ω_hat_new);
    view(solver1.ω_arr, 1:solver1.x_len, 1:solver1.y_len, iteration) .= solver1.ω_temp;

    calc_flow_measures_iter(solver1, measures, iteration);

    return nothing;
end;


function run(solver1::solver, num_iters::Int64)
    for i in 1:num_iters
        iter(solver1, solver1.current_iter);
        solver1.current_iter += 1;
    end;
end;


function run(solver1::solver, measures::flow_measures, num_iters::Int64)
    for i in 1:num_iters
        iter(solver1, measures, solver1.current_iter);
        solver1.current_iter += 1;
    end;
end;
