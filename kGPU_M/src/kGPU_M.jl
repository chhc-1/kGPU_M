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

#=

#-----------------


mutable struct Fourier_array{Arraytype<:arr_type}
    pad::Arraytype   #CUDA.CUDA.CuArray{Float64, 2}; # padded real array
    hat_padded::Arraytype   #CUDA.CUDA.CuArray{ComplexF64, 2}; #  padded Fourier space array

    Fourier_array{Arraytype}() where {Arraytype} = new{Arraytype}();

    function Fourier_array{Arraytype}(pad_arr::Arraytype, hat_padded_arr::Arraytype) where Arraytype
        new{Arraytype}(pad_arr, hat_padded_arr);
    end;     

    Fourier_array{Arraytype}(
        x_size_pad::Int64, y_size_pad::Int64, 
        N1r_pad::Int64, N2_pad::Int64) where Arraytype = 
    new{Arraytype}(
        Arraytype{Float64}(repeat([0.0], x_size_pad, y_size_pad)),
        Arraytype{ComplexF64}(repeat([0.0 + 0.0*im], N1r_pad, N2_pad)));
end;

function Adapt.adapt_structure(to, F_arr::Fourier_array)
    pad = Adapt.adapt(to, F_arr.pad);
    hat_padded = Adapt.adapt(to, F_arr.hat_padded);
    return Fourier_array{to}(pad, hat_padded);
end;

#--------------------------
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
    if to == CUDA.CuArray
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
    else
        CUDA.@allowscalar x_arr = Adapt.adapt(to{typeof(solver1.x_arr[1])}, solver1.x_arr);
        CUDA.@allowscalar y_arr = Adapt.adapt(to{typeof(solver1.y_arr[1])}, solver1.y_arr);

        CUDA.@allowscalar ω_arr = Adapt.adapt(to{typeof(solver1.ω_arr[1])}, solver1.ω_arr);
        CUDA.@allowscalar ω_temp = Adapt.adapt(to{typeof(solver1.ω_temp[1])}, solver1.ω_temp);
        CUDA.@allowscalar ω0 = Adapt.adapt(to{typeof(solver1.ω0[1])}, solver1.ω0);
        CUDA.@allowscalar ω_hat_prev = Adapt.adapt(to{typeof(solver1.ω_hat_prev[1])}, solver1.ω_hat_prev);
        CUDA.@allowscalar ω_hat_intermediate = Adapt.adapt(to{typeof(solver1.ω_hat_intermediate[1])}, solver1.ω_hat_intermediate);
        CUDA.@allowscalar ω_hat_new = Adapt.adapt(to{typeof(solver1.ω_hat_new[1])}, solver1.ω_hat_new);
        
        CUDA.@allowscalar source_hat = Adapt.adapt(to{typeof(solver1.source_hat[1])}, solver1.source_hat);

        #intermediate arrays
        CUDA.@allowscalar u = Adapt.adapt_structure(to, solver1.u);
        CUDA.@allowscalar v = Adapt.adapt_structure(to, solver1.v);
        CUDA.@allowscalar dωdx = Adapt.adapt_structure(to, solver1.dωdx);
        CUDA.@allowscalar dωdy = Adapt.adapt_structure(to, solver1.dωdy);
        CUDA.@allowscalar udωdx = Adapt.adapt_structure(to, solver1.udωdx);
        CUDA.@allowscalar vdωdy = Adapt.adapt_structure(to, solver1.vdωdy);
        
        CUDA.@allowscalar G = Adapt.adapt(to{typeof(solver1.G[1])}, solver1.G);
        CUDA.@allowscalar Ginv_1 = Adapt.adapt(to{typeof(solver1.Ginv_1[1])}, solver1.Ginv_1);

        CUDA.@allowscalar conv_prev = Adapt.adapt(to{typeof(solver1.conv_prev[1])}, solver1.conv_prev);
        CUDA.@allowscalar conv_intermediate = Adapt.adapt(to{typeof(solver1.conv_intermediate[1])}, solver1.conv_intermediate);

        CUDA.@allowscalar mask = Adapt.adapt(to{typeof(solver1.mask[1])}, solver1.mask);

        # wavenumbers
        CUDA.@allowscalar kx = Adapt.adapt(to{typeof(solver1.kx[1])}, solver1.kx);
        CUDA.@allowscalar ky = Adapt.adapt(to{typeof(solver1.ky[1])}, solver1.ky);
        CUDA.@allowscalar kxy2 = Adapt.adapt(to{typeof(solver1.kxy2[1])}, solver1.kxy2);
    end;


    
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
#--------------------

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
        #weight1 = Arraytype{Float64}(repeat([node_weighting], solver1.x_len, solver1.y_len));
        #view(weight1, 1:solver1.x_len, 1) .*= 0.5;
        #view(weight1, 1:solver1.x_len, solver1.y_len) .*= 0.5;
        #view(weight1, 1, 1:solver1.y_len) .*= 0.5;
        #view(weight1, solver1.x_len, 1:solver1.y_len) .*= 0.5;

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







function calc_flow_measures(solver1::solver{Arraytype, rfft_t, irfft_t}, measures::flow_measures, iter::Int64) where {Arraytype, rfft_t, irfft_t}
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

    measures.u_hat .= im .* solver1.ω_hat_new .* solver1.ky ./ solver1.kxy2;
    measures.v_hat .= -im .* solver1.ω_hat_new .* solver1.kx ./ solver1.kxy2;

    measures.dudx_hat .= im .* solver1.kx .* measures.u_hat;
    measures.dudy_hat .= im .* solver1.ky .* measures.u_hat;
    measures.dvdx_hat .= im .* solver1.kx .* measures.v_hat;
    measures.dvdy_hat .= im .* solver1.ky .* measures.v_hat;

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

function calc_flow_measures_iter(solver1::solver, measures::flow_measures, iter::Int64)
    #CUDA.CUBLAS.mul!(solver1.ω_hat_new, solver1.rfft_plan, view(solver1.ω_arr, 1:solver1.x_len, 1:solver1.y_len, iter));
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

    measures.u_hat .= im .* solver1.ω_hat_new .* solver1.ky ./ solver1.kxy2;
    measures.v_hat .= -im .* solver1.ω_hat_new .* solver1.kx ./ solver1.kxy2;

    measures.dudx_hat .= im .* solver1.kx .* measures.u_hat;
    measures.dudy_hat .= im .* solver1.ky .* measures.u_hat;
    measures.dvdx_hat .= im .* solver1.kx .* measures.v_hat;
    measures.dvdy_hat .= im .* solver1.ky .* measures.v_hat;

    #measures.u_hat[1, 1] = 0;
    #measures.v_hat[1, 1] = 0;
    #measures.dudx_hat[1, 1] = 0;
    #measures.dudy_hat[1, 1] = 0;
    #measures.dvdx_hat[1, 1] = 0;
    #measures.dvdy_hat[1, 1] = 0;
    
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
    
    #NumericalIntegration.integrate((solver1.x_arr, solver1.y_arr), measures.D_arr);
    #measures.dissipation_rate[iter] = 1 / (4 * pi^2 * solver1.Re) * NumericalIntegration.integrate((solver1.x_arr, solver1.y_arr), measures.D_arr);
    #measures.EIR[iter] = 1 / (2pi)^2 * NumericalIntegration.integrate((solver1.x_arr, solver1.y_arr), measures.I_arr);
end;
#-------------------
function calc_conv(solver1::solver, ω_hat::arr_type, return_arr::arr_type)

    solver1.dωdx.hat_padded .= im .* solver1.kx .* ω_hat .* solver1.mask;
    solver1.dωdy.hat_padded .= im .* solver1.ky .* ω_hat .* solver1.mask;

    solver1.u.hat_padded .= solver1.dωdy.hat_padded ./ solver1.kxy2;
    solver1.v.hat_padded .= -1 .* solver1.dωdx.hat_padded ./ solver1.kxy2;
    
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

#----------------------------------
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



=#
#---------------------------

#---------------




#=
function init_F_array(solver1::solver, F_arr::Fourier_array)    
    F_arr.pad = repeat([convert(Float64, 0.0)], solver1.N1_padded, solver1.N2_padded);
    F_arr.hat = repeat([0.0 + 0.0*im], solver1.N1r, solver1.N2);
    F_arr.hat_padded = repeat([0.0 + 0.0*im], solver1.N1r_padded, solver1.N2_padded);
end;
=#


end # module kGPU_M
