module kGPU_M

import CUDA
import Adapt
import FFTW
import AbstractFFTs

complex_type = Union{ComplexF32, ComplexF64}
float_type = Union{Float16, Float32, Float64}

arr_type = Union{Array, CUDA.CuArray};
rfft_type = Union{FFTW.rFFTWPlan{Float64, -1, false, 2, Tuple{Int64, Int64}}, 
                    CUDA.CUFFT.CuFFTPlan{ComplexF32, Float32, -1, false, 2, 2, CUDA.CuArray{ComplexF32, 2, CUDA.DeviceMemory}},
                    CUDA.CUFFT.CuFFTPlan{ComplexF64, Float64, -1, false, 2, 2, CUDA.CuArray{ComplexF64, 2, CUDA.DeviceMemory}},
                    CUDA.CUFFT.CuFFTPlan{complex_type, float_type, -1, false, 2, 2, CUDA.CuArray{complex_type, 2, CUDA.DeviceMemory}}};
irfft_type = Union{AbstractFFTs.ScaledPlan{ComplexF64, FFTW.rFFTWPlan{ComplexF64, 1, false, 2, UnitRange{Int64}}, Float64},
    #AbstractFFTs.ScaledPlan{ComplexF32, CUDA.CUFFT.CuFFTPlan{Float32, ComplexF32, 1, false, 2, 2, CUDA.CuArray{ComplexF32, 2, CUDA.DeviceMemory}}, Float32},
    AbstractFFTs.ScaledPlan{ComplexF64, CUDA.CUFFT.CuFFTPlan{Float64, ComplexF64, 1, false, 2, 2, CUDA.CuArray{ComplexF64, 2, CUDA.DeviceMemory}}, Float64}};


mutable struct Fourier_array{Arraytype}
    pad::Union{Float64, Arraytype}   #CUDA.CUDA.CuArray{Float64, 2}; # padded real array
    hat_padded::Union{ComplexF64, Arraytype}   #CUDA.CUDA.CuArray{ComplexF64, 2}; #  padded Fourier space array

    Fourier_array{Arraytype}() where {Arraytype} = new{Arraytype}();

    function Fourier_array{Arraytype}(pad_arr::Arraytype, hat_padded_arr::Arraytype) where Arraytype
        new{Arraytype}(pad_arr, hat_padded_arr);
    end;     

    Fourier_array(
        x_size_pad::Int64, y_size_pad::Int64, 
        N1r_pad::Int64, N2_pad::Int64) = 
    new{Array}(
        Array{Float64}(repeat([0.0], x_size_pad, y_size_pad)),
        Array{ComplexF64}(repeat([0.0 + 0.0*im], N1r_pad, N2_pad)));
end;


# solver is in padded space - provide space with 1.5x domain size, 

mutable struct solver{Arraytype}
    dt::Float64;
    Re::Float64;
    n_iter::Int64;
    current_iter::Int64;
    x_len::Int64;
    y_len::Int64;
    x_arr::Arraytype #CUDA.CuArray{Float64, 2};
    y_arr::Arraytype #CUDA.CuArray{Float64, 2};
    n::Int64; # Forcing frequency

    ω_arr::Union{Float64, Arraytype}  #CUDA.CuArray{Float64, 3}; # contains real space ω values
    ω_temp::Union{Float64, Arraytype} #CUDA.CuArray{Float64, 2}; # temporary storage for ω values -> not necesary if can fix mul! not working on selected array size?
    ω0::Union{Float64, Arraytype} #CUDA.CuArray{Float64, 2} # initial condition in real space
    ω_hat_prev::Union{ComplexF64, Arraytype} #CUDA.CuArray{ComplexF64, 2}; # previous ω in Fourier space
    ω_hat_intermediate::Union{ComplexF64, Arraytype} #CUDA.CuArray{ComplexF64, 2}; # omega i+1 tilde in Fourier space
    ω_hat_new::Union{ComplexF64, Arraytype} #CUDA.CuArray{ComplexF64, 2}; # next ω in Fourier space
    
    source_hat::Union{ComplexF64, Arraytype}

    #intermediate arrays
    u::Fourier_array{Arraytype};
    v::Fourier_array{Arraytype};
    dωdx::Fourier_array{Arraytype};
    dωdy::Fourier_array{Arraytype};
    udωdx::Fourier_array{Arraytype};
    vdωdy::Fourier_array{Arraytype};

    G::Union{ComplexF64, Arraytype} #CUDA.CuArray{Float64, 2}; # Implicit operator in Fourier space
    Ginv_1::Union{ComplexF64, Arraytype} #CUDA.CuArray{Float64, 2};

    conv_prev::Union{ComplexF64, Arraytype} # CUDA.CuArray{ComplexF64, 2}; # convection of previous time step
    conv_intermediate::Union{ComplexF64, Arraytype} # CUDA.CuArray{ComplexF64, 2} # convection of intermediate term

    mask::Union{Bool, Arraytype}

    N1::Int64; # direction 1 size of fft array # necessary?
    N1r::Int64; # direction 1 size of rfft array # necessary?
    N2::Int64; # direction 2 size of rfft array # necessary?
    N2_half::Int64; # necessary?
    
    N1r_padded::Int64; # total size of now padded array
    N1_padded::Int64;
    N2_padded::Int64;

    #pad_scaling_factor::Float64; # scaling factor to transfer from padded to non-padded array  # necessary?
    # wavenumbers
    kx::Union{Int64, Arraytype} #CUDA.CuArray{Int64, 2};
    ky::Union{Int64, Arraytype} #CUDA.CuArray{Int64, 2};
    kxy2::Union{Int64, Arraytype} #CUDA.CuArray{Int64, 2}; # array containing kx^2 + ky^2 

    # FFT plans
    rfft_plan_padded::rfft_type;
    irfft_plan_padded::irfft_type;
    
    solver{Arraytype}() where Arraytype = new{Arraytype}();
end;


mutable struct flow_measures{Arraytype}
    dissipation_rate::Union{Float64, Arraytype} # CUDA.CuArray{Float64, 1};
    EIR::Union{Float64, Arraytype} #CUDA.CuArray{Float64, 1}; # energy input rate

    # temporary variables
    u::Union{Float64, Arraytype} #CUDA.CuArray{Float64, 2};
    u_hat::Union{Float64, Arraytype} #CUDA.CuArray{ComplexF64, 2};
    v::Union{Float64, Arraytype} #CUDA.CuArray{Float64, 2};
    v_hat::Union{Float64, Arraytype} #CUDA.CuArray{ComplexF64, 2};
    dudx::Union{Float64, Arraytype} #CUDA.CuArray{Float64, 2};
    dudx_hat::Union{Float64, Arraytype} #CUDA.CuArray{ComplexF64, 2};
    dudy::Union{Float64, Arraytype} #CUDA.CuArray{Float64, 2};
    dudy_hat::Union{Float64, Arraytype} #CUDA.CuArray{ComplexF64, 2};
    dvdx::Union{Float64, Arraytype} #CUDA.CuArray{Float64, 2};
    dvdx_hat::Union{Float64, Arraytype} #CUDA.CuArray{ComplexF64, 2};
    dvdy::Union{Float64, Arraytype} #CUDA.CuArray{Float64, 2};
    dvdy_hat::Union{Float64, Arraytype} #CUDA.CuArray{ComplexF64, 2};
    
    D_arr::Union{Float64, Arraytype} #CUDA.CuArray{Float64, 2};
    I_arr::Union{Float64, Arraytype} #CUDA.CuArray{Float64, 2};
    
    function flow_measures{Arraytype}(solver1::solver) where Arraytype
        new(repeat([0.0], solver1.n_iter), 
            repeat([0.0], solver1.n_iter), 
            repeat([0.0], solver1.x_len, solver1.y_len),
            repeat([0.0 + 0.0*im], solver1.N1r, solver1.N2),
            repeat([0.0], solver1.x_len, solver1.y_len),
            repeat([0.0 + 0.0*im], solver1.N1r, solver1.N2),
            repeat([0.0], solver1.x_len, solver1.y_len),
            repeat([0.0 + 0.0*im], solver1.N1r, solver1.N2),
            repeat([0.0], solver1.x_len, solver1.y_len),
            repeat([0.0 + 0.0*im], solver1.N1r, solver1.N2),
            repeat([0.0], solver1.x_len, solver1.y_len),
            repeat([0.0 + 0.0*im], solver1.N1r, solver1.N2),
            repeat([0.0], solver1.x_len, solver1.y_len),
            repeat([0.0 + 0.0*im], solver1.N1r, solver1.N2),
            repeat([0.0], solver1.x_len, solver1.y_len),
            repeat([0.0], solver1.x_len, solver1.y_len));
    end;
end;



function Adapt.adapt_structure(to, F_arr::Fourier_array)
    pad = Adapt.adapt(to, F_arr.pad);
    hat_padded = Adapt.adapt(to, F_arr.hat_padded);
    return Fourier_array{to}(pad, hat_padded);
end;


function Adapt.adapt_structure(to, solver1::solver)
    x_arr = Adapt.adapt(to{typeof(solver1.x_arr[1])}, solver1.x_arr);
    y_arr = Adapt.adapt(to{typeof(solver1.y_arr[1])}, solver1.y_arr);

    ω_arr = Adapt.adapt(to{typeof(solver1.ω_arr[1])}, solver1.ω_arr);
    ω_temp = Adapt.adapt(to{typeof(solver1.ω_temp[1])}, solver1.ω_temp);
    ω0 = Adapt.adapt(to{typeof(solver1.ω0[1])}, solver1.ω0);
    ω_hat_prev = Adapt.adapt(to{typeof(solver1.ω_hat_prev[1])}, solver1.ω_hat_prev);
    ω_hat_intermediate = Adapt.adapt(to{typeof(solver1.ω_hat_intermediate[1])}, solver1.ω_hat_intermediate);
    ω_hat_new = Adapt.adapt(to{typeof(solver1.ω_hat_new[1])}, solver1.ω_hat_new);
    
    source_hat = Adapt.adapt(to{typeof(solver1.source_hat[1])}, solver1.source_hat);

    #intermediate arrays
    u = Adapt.adapt_structure(to, solver1.u);
    v = Adapt.adapt_structure(to, solver1.v);
    dωdx = Adapt.adapt_structure(to, solver1.dωdx);
    dωdy = Adapt.adapt_structure(to, solver1.dωdy);
    udωdx = Adapt.adapt_structure(to, solver1.udωdx);
    vdωdy = Adapt.adapt_structure(to, solver1.vdωdy);
    
    G = Adapt.adapt(to{typeof(solver1.G[1])}, solver1.G);
    Ginv_1 = Adapt.adapt(to{typeof(solver1.Ginv_1[1])}, solver1.Ginv_1);

    conv_prev = Adapt.adapt(to{typeof(solver1.conv_prev[1])}, solver1.conv_prev);
    conv_intermediate = Adapt.adapt(to{typeof(solver1.conv_intermediate[1])}, solver1.conv_intermediate);

    mask = Adapt.adapt(to{typeof(solver1.mask[1])}, solver1.mask);

    # wavenumbers
    kx = Adapt.adapt(to{typeof(solver1.kx[1])}, solver1.kx);
    ky = Adapt.adapt(to{typeof(solver1.ky[1])}, solver1.ky);
    kxy2 = Adapt.adapt(to{typeof(solver1.kxy2[1])}, solver1.kxy2);

    rfft_plan_padded = CUDA.CUFFT.plan_rfft(u.pad);
    irfft_plan_padded = CUDA.CUFFT.plan_irfft(u.hat_padded, solver1.N1_padded);

    solver2 = solver{to}();

    solver2.dt = copy(solver1.dt);
    solver2.Re = copy(solver1.Re);
    solver2.n_iter = copy(solver1.n_iter);
    solver2.current_iter = copy(solver1.current_iter);
    solver2.x_len = copy(solver1.x_len);
    solver2.y_len = copy(solver1.y_len);
    solver2.x_arr = copy(solver1.x_arr);
    solver2.y_arr = copy(solver1.y_arr);
    solver2.n = copy(solver1.n);
    solver2.ω_arr = ω_arr;
    solver2.ω_temp = ω_temp;
    solver2.ω0 = ω0;
    solver2.ω_hat_prev = ω_hat_prev;
    solver2.ω_hat_intermediate = ω_hat_intermediate;
    solver2.ω_hat_new = ω_hat_new;
    solver2.source_hat = source_hat;
    solver2.u = u;
    solver2.v = v;
    solver2.dωdx = dωdx;
    solver2.dωdy = dωdy;
    solver2.udωdx = udωdx;
    solver2.vdωdy = vdωdy;
    solver2.G = G;
    solver2.Ginv_1 = Ginv_1;
    solver2.conv_prev = conv_prev;
    solver2.conv_intermediate = conv_intermediate;
    solver2.mask = mask;
    solver2.N1 = copy(solver1.N1);
    solver2.N1r = copy(solver1.N1r);
    solver2.N2 = copy(solver1.N2);
    solver2.N2_half = copy(solver1.N2_half);
    solver2.N1r_padded = copy(solver1.N1r_padded);
    solver2.N1_padded = copy(solver1.N1_padded);
    solver2.N2_padded = copy(solver1.N2_padded);
    solver2.kx = kx;
    solver2.ky = ky;
    solver2.kxy2 = kxy2;
    solver2.rfft_plan_padded = rfft_plan_padded;
    solver2.irfft_plan_padded = irfft_plan_padded;
    return solver2;
end;


function calc_conv(solver1::solver, ω_hat::CUDA.CuArray{ComplexF64, 2}, return_arr::CUDA.CuArray{ComplexF64, 2})
    
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


function update(solver1::solver)
    calc_conv(solver1, solver1.ω_hat_prev, solver1.conv_prev);
    solver1.ω_hat_intermediate .= solver1.Ginv_1 .* (solver1.ω_hat_prev .+ solver1.dt .* (-0.5 .* solver1.G .* solver1.ω_hat_prev .- solver1.conv_prev .+ solver1.source_hat));
    
    calc_conv(solver1, solver1.ω_hat_intermediate, solver1.conv_intermediate);

    solver1.ω_hat_new .= solver1.Ginv_1 .* (solver1.ω_hat_prev .+ solver1.dt .* (-0.5 .* solver1.G .* solver1.ω_hat_prev .- 0.5 .* (solver1.conv_prev .+ solver1.conv_intermediate) .+ solver1.source_hat));

    return nothing;
end

function iter(solver1::solver, iteration::Int64)
    update(solver1);

    CUDA.copyto!(solver1.ω_hat_prev, solver1.ω_hat_new);

    CUDA.CUBLAS.mul!(solver1.ω_temp, solver1.irfft_plan_padded, solver1.ω_hat_new);
    view(solver1.ω_arr, 1:solver1.x_len, 1:solver1.y_len, iteration) .= solver1.ω_temp;
    return nothing;
end;


function calc_solver_base(solver1::solver{Arraytype}) where Arraytype
    
    #initialiase Fourier series terms
    solver1.N1r_padded = div(solver1.x_len, 2) + 1;
    solver1.N1_padded = copy(solver1.x_len);
    solver1.N2_padded = copy(solver1.y_len);
    solver1.N2_half = div(solver1.N2, 2);
    
    #solver1.N2_idx2 = div(solver1.N2, 2)+1;
    N2_pad_idx = solver1.N2_padded - div(solver1.N2, 2)+1;
    
    N1_idx_arr = 1:solver1.N1r;
    N2_idx_arr1 = 1:div(solver1.N2, 2);
    N2_idx_arr2_padded = N2_pad_idx:solver1.N2_padded;
    
    temp_x = Array{Int64}([0:solver1.N1r_padded-1;]);
    temp_y = Array{Int64}([0:div(solver1.N2_padded, 2); -div(solver1.N2_padded, 2)+1:-1]);
    
    solver1.kx = Array{Int64}(repeat(temp_x, 1, solver1.N2_padded));
    solver1.ky = Array{Int64}(transpose(Array{Int64}(repeat(temp_y, 1, solver1.N1r_padded))));
    solver1.kxy2 = (solver1.kx).^2 + (solver1.ky).^2;
    solver1.kxy2[1, 1] = 1;
    
    solver1.mask = Array{Bool}(repeat([false], solver1.N1r_padded, solver1.N2_padded));
    view(solver1.mask, N1_idx_arr, N2_idx_arr1) .= true;
    view(solver1.mask, N1_idx_arr, N2_idx_arr2_padded) .= true;
    
end;


function init_F_array(solver1::solver, F_arr::Fourier_array)    
    F_arr.pad = repeat([convert(Float64, 0.0)], solver1.N1_padded, solver1.N2_padded);
    F_arr.hat = repeat([0.0 + 0.0*im], solver1.N1r, solver1.N2);
    F_arr.hat_padded = repeat([0.0 + 0.0*im], solver1.N1r_padded, solver1.N2_padded);
end;


#function init_solver(solver1::solver, dt::Float64, Re::Float64, n_iter::Int64, x_arr::CUDA.CuArray{Float64, 1}, y_arr::CUDA.CuArray{Float64, 1}, ω0::CUDA.CuArray{Float64, 2}, source_hat::CUDA.CuArray{ComplexF64, 2}, Forcing_freq::Int64)
function init_solver(solver1::solver, dt::Float64, Re::Float64, n_iter::Int64, x_arr::Union{Float64, arr_type}, y_arr::Union{Float64, arr_type}, 
        x_cutoff::Int64, y_cutoff::Int64, ω0::Union{Float64, arr_type}, source_hat::Union{Float64, arr_type}, Forcing_freq::Int64)
    @assert(size(x_arr, 2) == 1)
    @assert(size(y_arr, 2) == 1)
    
    solver1.dt = dt;
    solver1.Re = Re;
    solver1.n_iter = n_iter;
    solver1.current_iter = 0;
    solver1.x_len = size(x_arr, 1);
    solver1.y_len = size(y_arr, 1);
    solver1.x_arr = CUDA.CuArray{Float64}(repeat(x_arr, 1, solver1.y_len));
    solver1.y_arr = transpose(CUDA.CuArray{Float64}(repeat(y_arr, 1, solver1.x_len)));
    solver1.n = Forcing_freq;

    #initialise ω terms
    solver1.ω0 = copy(ω0);
    solver1.ω_hat_prev = CUDA.CUFFT.rfft(solver1.ω0);

    #initialiase Fourier series terms
    solver1.N1 = x_cutoff;
    solver1.N1r = div(x_cutoff, 2) + 1;
    solver1.N2 = y_cutoff;
    calc_solver_base(solver1);

    # initialise ω terms
    solver1.ω_arr = repeat([0.0], solver1.x_len, solver1.y_len, solver1.n_iter);
    solver1.ω_temp = repeat([0.0], solver1.x_len, solver1.y_len);
    solver1.ω_hat_intermediate = repeat([0.0 + 0.0*im], solver1.N1r_padded, solver1.N2_padded)
    solver1.ω_hat_new = repeat([0.0 + 0.0*im], solver1.N1r_padded, solver1.N2_padded);

    solver1.conv_prev = repeat([0.0 + 0.0*im], solver1.N1r_padded, solver1.N2_padded);
    solver1.conv_intermediate = repeat([0.0 + 0.0*im], solver1.N1r_padded, solver1.N2_padded);

    # intialise intermediate arrays
    solver1.u = Fourier_array(solver1.N1_padded, solver1.N2_padded, solver1.N1r_padded, solver1.N2_padded);
    solver1.v = Fourier_array(solver1.N1_padded, solver1.N2_padded, solver1.N1r_padded, solver1.N2_padded);
    solver1.dωdx = Fourier_array(solver1.N1_padded, solver1.N2_padded, solver1.N1r_padded, solver1.N2_padded);
    solver1.dωdy = Fourier_array(solver1.N1_padded, solver1.N2_padded, solver1.N1r_padded, solver1.N2_padded);
    solver1.udωdx = Fourier_array(solver1.N1_padded, solver1.N2_padded, solver1.N1r_padded, solver1.N2_padded);
    solver1.vdωdy = Fourier_array(solver1.N1_padded, solver1.N2_padded, solver1.N1r_padded, solver1.N2_padded);

    # initialise source term
    solver1.source_hat = copy(source_hat);

    # initialise FFT plans
    solver1.rfft_plan_padded = FFTW.plan_rfft(solver1.u.pad); #CUDA.CUFFT.plan_rfft(solver1.u.pad);
    solver1.irfft_plan_padded = FFTW.plan_irfft(solver1.u.hat_padded, solver1.N1_padded); #CUDA.CUFFT.plan_irfft(solver1.u.hat_padded, solver1.N1_padded);
    
    # initialise implicit terms
    solver1.G = solver1.kxy2 ./ solver1.Re;  #[(kx^2 + ky^2) / solver1.Re for kx in solver1.kx, ky in solver1.ky];
    solver1.Ginv_1 = 1 ./ (1 .+ (dt / 2) .* solver1.G);
end;


function calc_flow_measures(solver1::solver, measures::flow_measures, iter::Int64)
    CUDA.CUBLAS.mul!(solver1.ω_hat_new, solver1.rfft_plan, view(solver1.ω_arr, 1:solver1.x_len, 1:solver1.y_len, iter));
    #=
    for i in 1:solver1.N1r
        for j in 1:solver1.N2
            measures.u_hat[i, j] = im * solver1.ω_hat_new[i, j] * (solver1.ky[j] / (solver1.kx[i]^2 + solver1.ky[j]^2));
            measures.v_hat[i, j] = -im * solver1.ω_hat_new[i, j] * (solver1.kx[i] / (solver1.kx[i]^2 + solver1.ky[j]^2));

            measures.dudx_hat[i, j] = im * solver1.kx[i] * measures.u_hat[i, j];
            measures.dudy_hat[i, j] = im * solver1.ky[j] * measures.u_hat[i, j];
            measures.dvdx_hat[i, j] = im * solver1.kx[i] * measures.v_hat[i, j];
            measures.dvdy_hat[i, j] = im * solver1.ky[j] * measures.v_hat[i, j];
        end;
    end;
    =#

    measures.u_hat = im .* solver1.ω_hat_new .* solver1.ky ./ solver1.kxy2;
    measures.v_hat = -im .* solver1.ω_hat_new .* solver1.kx ./ solver1.kxy2;

    measures.dudx_hat = im .* solver1.kx .* measures.u_hat;
    measures.dudy_hat = im .* solver1.ky .* measures.u_hat;
    measures.dvdx_hat = im .* solver1.kx .* measures.v_hat;
    measures.dvdy_hat = im .* solver1.ky .* measures.v_hat;

    #measures.u_hat[1, 1] = 0;
    #measures.v_hat[1, 1] = 0;
    #measures.dudx_hat[1, 1] = 0;
    #measures.dudy_hat[1, 1] = 0;
    #measures.dvdx_hat[1, 1] = 0;
    #measures.dvdy_hat[1, 1] = 0;
    
    CUDA.CUBLAS.mul!(measures.u, solver1.irfft_plan, measures.u_hat);
    CUDA.CUBLAS.mul!(measures.v, solver1.irfft_plan, measures.v_hat);
    CUDA.CUBLAS.mul!(measures.dudx, solver1.irfft_plan, measures.dudx_hat);
    CUDA.CUBLAS.mul!(measures.dudy, solver1.irfft_plan, measures.dudy_hat);
    CUDA.CUBLAS.mul!(measures.dvdx, solver1.irfft_plan, measures.dvdx_hat);
    CUDA.CUBLAS.mul!(measures.dvdy, solver1.irfft_plan, measures.dvdy_hat);

    measures.D_arr = (measures.dudx).^2 + (measures.dudy).^2 + (measures.dvdx).^2 + (measures.dvdy).^2;
    measures.I_arr = measures.u .* sin.(solver1.n .* solver1.y_arr);

    
    measures.dissipation_rate[iter] = 1 / (4 * pi^2 * solver1.Re) * NumericalIntegration.integrate((solver1.x_arr, solver1.y_arr), measures.D_arr);
    measures.EIR[iter] = 1 / (2pi)^2 * NumericalIntegration.integrate((solver1.x_arr, solver1.y_arr), measures.I_arr);
end;

end # module kGPU_M
