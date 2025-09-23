using Test
using CUDA
using Adapt
using FFTW
using AbstractFFTs
using kGPU_M


@testset "test1" begin
    dt = 0.01
    Re = 40.0
    max_iter = 1000;

    x_len = 150;
    y_len = 150;
    xs_full = LinRange(0, 2pi, x_len+1);
    ys_full = LinRange(0, 2pi, y_len+1);
    xs = Array{Float64}(xs_full[1:x_len]);
    ys = Array{Float64}(ys_full[1:y_len]);

    ω0 = CUDA.rand(x_len, y_len);
   
    x_cutoff = 100;
    y_cutoff = 100;

    function source_fn(x, y)
        return -4 * cos(4 * y);
    end;

    source = [source_fn(x, y) for x in xs, y in ys];
    source_hat = FFTW.rfft(source);

    forcing_freq = 4;

    solver_GPU = kGPU_M.solver{CUDA.CuArray, kGPU_M.GPU_rfft_type64, kGPU_M.GPU_irfft_type64}();
    kGPU_M.init_solver(solver_GPU, dt, Re, max_iter, xs, ys, x_cutoff, y_cutoff, ω0, source_hat, forcing_freq);
    
    kGPU_M.run(solver_GPU, max_iter);
    solver_CPU = Adapt.adapt_structure(Array, solver_GPU);
    solver_GPU = Adapt.adapt_structure(CuArray, solver_CPU);

    measures = kGPU_M.flow_measures{Array}(solver_CPU);
    temp_arr = Array(repeat([0.0], x_len, y_len));
    kGPU_M.calc_flow_measures(solver_CPU, measures, 1, temp_arr);
    
end