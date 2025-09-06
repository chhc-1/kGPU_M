import CUDA
import AbstractFFTs
import FFTW
import Adapt

#include("typedefs.jl")
include("Fourier_array.jl")

#=
File containing mutable struct solver and the corresponding Adapt.adapt_structure function and external function to assign values to attributes of solver object
solver object contains all necessary information to run the simulation, 
=#

mutable struct solver{Arraytype<:arr_type, rfft_t<:rfft_type, irfft_t<:irfft_type}
    dt::Float64;
    Re::Float64;
    n_iter::Int64;
    current_iter::Int64;
    x_len::Int64;
    y_len::Int64;
    x_arr::Arraytype #CUDA.CuArray{Float64, 2};
    y_arr::Arraytype #CUDA.CuArray{Float64, 2};
    n::Int64; # Forcing frequency

    ω_arr::Union{Float64, Arraytype} # float array type containing all values of ω in real sapce #CUDA.CuArray{Float64, 3}; # contains real space ω values
    ω_temp::Union{Float64, Arraytype} # float array type containing temporary ω in real space #CUDA.CuArray{Float64, 2}; # temporary storage for ω values -> not necesary if can fix mul! not working on selected array size?
    ω0::Arraytype # float array type containing initial condition in real space #CUDA.CuArray{Float64, 2} # initial condition in real space
    ω_hat_prev::Arraytype # complex array type containing previous ω in Fourier space #CUDA.CuArray{ComplexF64, 2}; # previous ω in Fourier space
    ω_hat_intermediate::Arraytype # complex array containing intermediate ω in Fourier space #CUDA.CuArray{ComplexF64, 2}; # omega i+1 tilde in Fourier space
    ω_hat_new::Arraytype # complex array containing next ω in Fourier space  #CUDA.CuArray{ComplexF64, 2}; # next ω in Fourier space
    
    source_hat::Arraytype# complex array containing source terms in Fourier space

    #intermediate arrays
    u::Fourier_array{Arraytype};
    v::Fourier_array{Arraytype};
    dωdx::Fourier_array{Arraytype};
    dωdy::Fourier_array{Arraytype};
    udωdx::Fourier_array{Arraytype};
    vdωdy::Fourier_array{Arraytype};

    G::Arraytype #CUDA.CuArray{Float64, 2}; # Implicit operator in Fourier space
    Ginv_1::Arraytype #CUDA.CuArray{Float64, 2};

    conv_prev::Arraytype # complex array containing previous convection term # CUDA.CuArray{ComplexF64, 2}; # convection of previous time step
    conv_intermediate::Arraytype # complex array containing intermediate convection term # CUDA.CuArray{ComplexF64, 2} # convection of intermediate term

    mask::Arraytype # boolean array

    N1::Int64; # direction 1 size of fft array # necessary?
    N1r::Int64; # direction 1 size of rfft array # necessary?
    N2::Int64; # direction 2 size of rfft array # necessary?
    N2_half::Int64; # necessary?
    
    N1r_padded::Int64; # total size of now padded array
    N1_padded::Int64;
    N2_padded::Int64;

    #pad_scaling_factor::Float64; # scaling factor to transfer from padded to non-padded array  # necessary?
    # wavenumbers
    kx::Arraytype #CUDA.CuArray{Int64, 2};
    ky::Arraytype #CUDA.CuArray{Int64, 2};
    kxy2::Arraytype #CUDA.CuArray{Int64, 2}; # array containing kx^2 + ky^2 

    # FFT plans
    rfft_plan_padded::rfft_t;
    irfft_plan_padded::irfft_t;
    
    solver{Arraytype, rfft_t, irfft_t}() where {Arraytype, rfft_t, irfft_t} = new{Arraytype, rfft_t, irfft_t}();
end;


function Adapt.adapt_structure(to, solver1::solver) # cannot impose type restriction on "to" argument
    #println("in function");
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

    
    if to == CUDA.CuArray
        solver2 = solver{
        to, 
        CUDA.CUFFT.CuFFTPlan{ComplexF64, Float64, -1, false, 2, 2, CUDA.CuArray{ComplexF64, 2, CUDA.DeviceMemory}},
        AbstractFFTs.ScaledPlan{ComplexF64, CUDA.CUFFT.CuFFTPlan{Float64, ComplexF64, 1, false, 2, 2, CUDA.CuArray{ComplexF64, 2, CUDA.DeviceMemory}}, Float64}
        }();
        rfft_plan_padded = CUDA.CUFFT.plan_rfft(u.pad);
        irfft_plan_padded = CUDA.CUFFT.plan_irfft(u.hat_padded, solver1.N1_padded);
    elseif to == Array
        solver2 = solver{to, 
        FFTW.rFFTWPlan{Float64, -1, false, 2, Tuple{Int64, Int64}},
        AbstractFFTs.ScaledPlan{ComplexF64, FFTW.rFFTWPlan{ComplexF64, 1, false, 2, UnitRange{Int64}}, Float64}
        }();
        rfft_plan_padded = FFTW.plan_rfft(u.pad);
        irfft_plan_padded = FFTW.plan_irfft(u.hat_padded, solver1.N1_padded);
    end;    
    


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

#function init_solver(solver1::solver, dt::Float64, Re::Float64, n_iter::Int64, x_arr::CUDA.CuArray{Float64, 1}, y_arr::CUDA.CuArray{Float64, 1}, ω0::CUDA.CuArray{Float64, 2}, source_hat::CUDA.CuArray{ComplexF64, 2}, Forcing_freq::Int64)
function init_solver(solver1::solver{Arraytype, rfft_t, irfft_t}, dt::Float64, Re::Float64, n_iter::Int64, x_arr::arr_type, y_arr::arr_type, 
        x_cutoff::Int64, y_cutoff::Int64, ω0::arr_type, source_hat::arr_type, Forcing_freq::Int64) where {Arraytype, rfft_t, irfft_t}
    @assert(size(x_arr, 2) == 1)
    @assert(size(y_arr, 2) == 1)
    
    solver1.dt = dt;
    solver1.Re = Re;
    solver1.n_iter = n_iter;
    solver1.current_iter = 1;
    solver1.x_len = size(x_arr, 1);
    solver1.y_len = size(y_arr, 1);
    solver1.x_arr = Arraytype{Float64}(repeat(x_arr, 1, solver1.y_len));
    solver1.y_arr = transpose(Arraytype{Float64}(repeat(y_arr, 1, solver1.x_len)));
    solver1.n = Forcing_freq;

    #initialise ω terms
    solver1.ω0 = Arraytype{Float64}(ω0);
    if Arraytype == CUDA.CuArray
        solver1.ω_hat_prev = CUDA.CUFFT.rfft(solver1.ω0);
    else
        solver1.ω_hat_prev = FFTW.rfft(solver1.ω0);
    end;

    #initialiase Fourier series terms
    solver1.N1 = x_cutoff;
    solver1.N1r = div(x_cutoff, 2) + 1;
    solver1.N2 = y_cutoff;
    calc_solver_base(solver1);

    # initialise ω terms
    solver1.ω_arr = Arraytype{Float64}(repeat([0.0], solver1.x_len, solver1.y_len, solver1.n_iter));
    solver1.ω_temp = Arraytype{Float64}(repeat([0.0], solver1.x_len, solver1.y_len));
    solver1.ω_hat_intermediate = Arraytype{ComplexF64}(repeat([0.0 + 0.0*im], solver1.N1r_padded, solver1.N2_padded));
    solver1.ω_hat_new = Arraytype{ComplexF64}(repeat([0.0 + 0.0*im], solver1.N1r_padded, solver1.N2_padded));

    solver1.conv_prev = Arraytype{ComplexF64}(repeat([0.0 + 0.0*im], solver1.N1r_padded, solver1.N2_padded));
    solver1.conv_intermediate = Arraytype{ComplexF64}(repeat([0.0 + 0.0*im], solver1.N1r_padded, solver1.N2_padded));

    # intialise intermediate arrays
    solver1.u = Fourier_array{Arraytype}(solver1.N1_padded, solver1.N2_padded, solver1.N1r_padded, solver1.N2_padded);
    solver1.v = Fourier_array{Arraytype}(solver1.N1_padded, solver1.N2_padded, solver1.N1r_padded, solver1.N2_padded);
    solver1.dωdx = Fourier_array{Arraytype}(solver1.N1_padded, solver1.N2_padded, solver1.N1r_padded, solver1.N2_padded);
    solver1.dωdy = Fourier_array{Arraytype}(solver1.N1_padded, solver1.N2_padded, solver1.N1r_padded, solver1.N2_padded);
    solver1.udωdx = Fourier_array{Arraytype}(solver1.N1_padded, solver1.N2_padded, solver1.N1r_padded, solver1.N2_padded);
    solver1.vdωdy = Fourier_array{Arraytype}(solver1.N1_padded, solver1.N2_padded, solver1.N1r_padded, solver1.N2_padded);

    # initialise source term
    solver1.source_hat = Arraytype{ComplexF64}(source_hat);#convert(Arraytype, copy(source_hat));

    # initialise FFT plans
    #solver1.rfft_plan_padded = FFTW.plan_rfft(solver1.u.pad); #CUDA.CUFFT.plan_rfft(solver1.u.pad);
    #solver1.irfft_plan_padded = FFTW.plan_irfft(solver1.u.hat_padded, solver1.N1_padded); #CUDA.CUFFT.plan_irfft(solver1.u.hat_padded, solver1.N1_padded);
    if Arraytype == Array
        solver1.rfft_plan_padded = FFTW.plan_rfft(solver1.u.pad); #CUDA.CUFFT.plan_rfft(solver1.u.pad);
        solver1.irfft_plan_padded = FFTW.plan_irfft(solver1.u.hat_padded, solver1.N1_padded); #CUDA.CUFFT.plan_irfft(solver1.u.hat_padded, solver1.N1_padded);
    else
        solver1.rfft_plan_padded = CUDA.CUFFT.plan_rfft(solver1.u.pad);
        solver1.irfft_plan_padded = CUDA.CUFFT.plan_irfft(solver1.u.hat_padded, solver1.N1_padded);
    end;


    # initialise implicit terms
    solver1.G = solver1.kxy2 ./ solver1.Re;  #[(kx^2 + ky^2) / solver1.Re for kx in solver1.kx, ky in solver1.ky];
    solver1.Ginv_1 = 1 ./ (1 .+ (dt / 2) .* solver1.G);
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
    
    temp_x = Arraytype{Int64}([0:solver1.N1r_padded-1;]);

    if solver1.N2_padded % 2 == 0
        temp_y = Arraytype{Int64}([0:div(solver1.N2_padded, 2); -div(solver1.N2_padded, 2)+1:-1]);
    else
        temp_y = Arraytype{Int64}([0:div(solver1.N2_padded, 2); -div(solver1.N2_padded, 2):-1]);
    end;

    
    solver1.kx = Arraytype{Int64}(repeat(temp_x, 1, solver1.N2_padded));
    solver1.ky = Arraytype{Int64}(transpose(Array{Int64}(repeat(temp_y, 1, solver1.N1r_padded))));
    solver1.kxy2 = (solver1.kx).^2 + (solver1.ky).^2;
    CUDA.@allowscalar solver1.kxy2[1, 1] = 1;
    
    solver1.mask = Arraytype{Bool}(repeat([false], solver1.N1r_padded, solver1.N2_padded));
    view(solver1.mask, N1_idx_arr, N2_idx_arr1) .= true;
    view(solver1.mask, N1_idx_arr, N2_idx_arr2_padded) .= true;
    
end;