import CUDA

include("typedefs.jl")
include("solver.jl")


if !isdefined(@__MODULE__, :flow_measures)
    mutable struct flow_measures{Arraytype<:arr_type}
        dissipation_rate::Arraytype # float array type containing dissipation rate values  # CUDA.CuArray{Float64, 1};
        EIR::Arraytype # float array type containing EIR values #CUDA.CuArray{Float64, 1}; # energy input rate

        weighting::Float64
        #weighting_arr::Union{Float64, Arraytype} # weighting array for trapezoidal integration

        # temporary variables
        u::Arraytype # float array type containing x-velocity values in real space #CUDA.CuArray{Float64, 2};
        u_hat::Arraytype # complex array type containing x-velocity values in Fourier space #CUDA.CuArray{ComplexF64, 2};
        v::Arraytype  # float array type containing y-velocity values in real space #CUDA.CuArray{Float64, 2};
        v_hat::Arraytype  # complex array type containing y-velocity values in real space #CUDA.CuArray{ComplexF64, 2};
        dudx::Arraytype  # float array type containing x derivative of x-velocity values in real space #CUDA.CuArray{Float64, 2};
        dudx_hat::Arraytype # complex array type containing x derivative of x-velocity values in Fourier space #CUDA.CuArray{ComplexF64, 2};
        dudy::Arraytype # float array type containing y derivative of x-velocity values in real space #CUDA.CuArray{Float64, 2};
        dudy_hat::Arraytype # complex array type containing y derivative of x-velocity values in Fourier space #CUDA.CuArray{ComplexF64, 2};
        dvdx::Arraytype # float array type containing x derivative of y-velocity values in real space #CUDA.CuArray{Float64, 2};
        dvdx_hat::Arraytype # complex array type containing x derivative of y-velocity values in Fourier space #CUDA.CuArray{ComplexF64, 2};
        dvdy::Arraytype # float array type containing y derivative of y-velocity values in real space #CUDA.CuArray{Float64, 2};
        dvdy_hat::Arraytype # complex array type containing y derivative of y-velocity values in Fourier space #CUDA.CuArray{ComplexF64, 2};
        
        D_arr::Arraytype # float array type containing values of dissipiation rate for each node #CUDA.CuArray{Float64, 2};
        I_arr::Arraytype # float array type containing values of EIR for each node #CUDA.CuArray{Float64, 2};

        #weighted_D_arr::Union{Float64, Arraytype}
        #weighted_I_arr::Union{Float64, Arraytype}

        function flow_measures{Arraytype}(solver1::solver) where {Arraytype}
            CUDA.@allowscalar dx = solver1.x_arr[2, 1] - solver1.x_arr[1, 1];
            CUDA.@allowscalar dy = solver1.y_arr[1, 2] - solver1.y_arr[1, 1];
            node_weighting = dx*dy;

            new{Arraytype}(Arraytype{Float64}(repeat([0.0], solver1.n_iter)), 
                Arraytype{Float64}(repeat([0.0], solver1.n_iter)), 
                node_weighting, #Arraytype{Float64}(weight1),
                Arraytype{Float64}(repeat([0.0], solver1.x_len, solver1.y_len)),
                Arraytype{ComplexF64}(repeat([0.0 + 0.0*im], solver1.N1r_padded, solver1.N2_padded)),
                Arraytype{Float64}(repeat([0.0], solver1.x_len, solver1.y_len)),
                Arraytype{ComplexF64}(repeat([0.0 + 0.0*im], solver1.N1r_padded, solver1.N2_padded)),
                Arraytype{Float64}(repeat([0.0], solver1.x_len, solver1.y_len)),
                Arraytype{ComplexF64}(repeat([0.0 + 0.0*im], solver1.N1r_padded, solver1.N2_padded)),
                Arraytype{Float64}(repeat([0.0], solver1.x_len, solver1.y_len)),
                Arraytype{ComplexF64}(repeat([0.0 + 0.0*im], solver1.N1r_padded, solver1.N2_padded)),
                Arraytype{Float64}(repeat([0.0], solver1.x_len, solver1.y_len)),
                Arraytype{ComplexF64}(repeat([0.0 + 0.0*im], solver1.N1r_padded, solver1.N2_padded)),
                Arraytype{Float64}(repeat([0.0], solver1.x_len, solver1.y_len)),
                Arraytype{ComplexF64}(repeat([0.0 + 0.0*im], solver1.N1r_padded, solver1.N2_padded)),
                Arraytype{Float64}(repeat([0.0], solver1.x_len, solver1.y_len)),
                Arraytype{Float64}(repeat([0.0], solver1.x_len, solver1.y_len)));
        end;
    end;
end;

if !isdefined(@__MODULE__, :calc_flow_measures)
    function calc_flow_measures(solver1::solver{Arraytype, rfft_t, irfft_t}, measures::flow_measures, iter::Int64, temp_arr::arr_type) where {Arraytype, rfft_t, irfft_t}
        #view(temp_arr, 1:solver1.N1_padded, 1:solver1.N2_padded) .= view(solver1.ω_arr, 1:solver1.N1_padded, 1:solver1.N2_padded, iter);
        #CUDA.CUBLAS.mul!(solver1.ω_hat_new, solver1.rfft_plan_padded, temp_arr);
        view(solver1.ω_hat_new, 1:solver1.N1r_padded, 1:solver1.N2_padded) .= view(solver1.ω_arr, 1:solver1.N1r_padded, 1:solver1.N2_padded, iter)

        measures.u_hat .= im .* solver1.ω_hat_new .* solver1.ky ./ solver1.kxy2;
        measures.v_hat .= -im .* solver1.ω_hat_new .* solver1.kx ./ solver1.kxy2;

        measures.dudx_hat .= im .* solver1.kx .* measures.u_hat;
        measures.dudy_hat .= im .* solver1.ky .* measures.u_hat;
        measures.dvdx_hat .= im .* solver1.kx .* measures.v_hat;
        measures.dvdy_hat .= im .* solver1.ky .* measures.v_hat;

        CUDA.CUBLAS.mul!(measures.u, solver1.irfft_plan_padded, measures.u_hat);
        CUDA.CUBLAS.mul!(measures.v, solver1.irfft_plan_padded, measures.v_hat);
        CUDA.CUBLAS.mul!(measures.dudx, solver1.irfft_plan_padded, measures.dudx_hat);
        CUDA.CUBLAS.mul!(measures.dudy, solver1.irfft_plan_padded, measures.dudy_hat);
        CUDA.CUBLAS.mul!(measures.dvdx, solver1.irfft_plan_padded, measures.dvdx_hat);
        CUDA.CUBLAS.mul!(measures.dvdy, solver1.irfft_plan_padded, measures.dvdy_hat);

        measures.D_arr = (measures.dudx).^2 + (measures.dudy).^2 + (measures.dvdx).^2 + (measures.dvdy).^2;
        measures.I_arr = measures.u .* sin.(solver1.n .* solver1.y_arr);
        
        view(measures.dissipation_rate, iter) .= 1 / (4 * pi^2 * solver1.Re) * measures.weighting * reduce(+, measures.D_arr);
        view(measures.EIR, iter) .= 1 / (2pi)^2 * measures.weighting * reduce(+, measures.I_arr);
        
    end;
end;

if !isdefined(@__MODULE__, :calc_flow_measures_iter)
    # function to calculate flow measures whilst solving
    function calc_flow_measures_iter(solver1::solver, measures::flow_measures, iter::Int64)

        measures.u_hat .= im .* solver1.ω_hat_new .* solver1.ky ./ solver1.kxy2;
        measures.v_hat .= -im .* solver1.ω_hat_new .* solver1.kx ./ solver1.kxy2;

        measures.dudx_hat .= im .* solver1.kx .* measures.u_hat;
        measures.dudy_hat .= im .* solver1.ky .* measures.u_hat;
        measures.dvdx_hat .= im .* solver1.kx .* measures.v_hat;
        measures.dvdy_hat .= im .* solver1.ky .* measures.v_hat;
     
        CUDA.CUBLAS.mul!(measures.u, solver1.irfft_plan_padded, measures.u_hat);
        CUDA.CUBLAS.mul!(measures.v, solver1.irfft_plan_padded, measures.v_hat);
        CUDA.CUBLAS.mul!(measures.dudx, solver1.irfft_plan_padded, measures.dudx_hat);
        CUDA.CUBLAS.mul!(measures.dudy, solver1.irfft_plan_padded, measures.dudy_hat);
        CUDA.CUBLAS.mul!(measures.dvdx, solver1.irfft_plan_padded, measures.dvdx_hat);
        CUDA.CUBLAS.mul!(measures.dvdy, solver1.irfft_plan_padded, measures.dvdy_hat);

        measures.D_arr = (measures.dudx).^2 + (measures.dudy).^2 + (measures.dvdx).^2 + (measures.dvdy).^2;
        measures.I_arr = measures.u .* sin.(solver1.n .* solver1.y_arr);

        #measures.weighted_D_arr = measures.D_arr .* measures.weighting; #measures.weighting_arr;
        #measures.weighted_I_arr = measures.D_arr .* measures.weighting; #measures.weighting_arr;

        view(measures.dissipation_rate, iter) .= 1 / (4 * pi^2 * solver1.Re) * measures.weighting * reduce(+, measures.D_arr);
        view(measures.EIR, iter) .= 1 / (2pi)^2 * measures.weighting * reduce(+, measures.I_arr);

    end;
end;

if !isdefined(@__MODULE__, :calc_physical_data)
    # function to compute flow data in physical space
    function calc_physical_data(solver1::solver{Arraytype}) where Arraytype
        real_data = Arraytype{Float64}(repeat([0.0], solver1.x_len, solver1.y_len, solver1.n_iter))
        #temp_arr = CuArray{Float64}(repeat([0.0], solver1.x_len, solver1.y_len))
        #temp_arr_hat = Array{ComplexF64}(repeat([0.0], solver1.N1r_padded, solver1.N2_padded))
        for i in 1:solver1.n_iter
            view(solver1.ω_hat_new, 1:solver1.N1r_padded, 1:solver1.N2_padded) .= view(solver1.ω_arr, 1:solver1.N1r_padded, 1:solver1.N2_padded, i)
            CUDA.CUBLAS.mul!(solver1.ω_temp, solver1.irfft_plan_padded, solver1.ω_hat_new);
            view(real_data, 1:solver1.x_len, 1:solver1.y_len, i) .= view(solver1.ω_temp, 1:solver1.x_len, 1:solver1.y_len);
        end;
        
        return real_data;
    end;
end;