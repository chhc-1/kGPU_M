include("typedefs.jl")
include("Fourier_array.jl")
include("solver.jl")


function Adapt.adapt_structure(to, F_arr::Fourier_array)
    pad = Adapt.adapt(to, F_arr.pad);
    hat_padded = Adapt.adapt(to, F_arr.hat_padded);
    return Fourier_array{to}(pad, hat_padded);
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