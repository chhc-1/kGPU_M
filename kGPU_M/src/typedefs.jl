import AbstractFFTs
import CUDA
import FFTW

global complex_type = Union{ComplexF32, ComplexF64}
global float_type = Union{Float16, Float32, Float64}

global arr_type = Union{Array, CUDA.CuArray};
global rfft_type = Union{FFTW.rFFTWPlan{Float64, -1, false, 2, Tuple{Int64, Int64}}, 
                    CUDA.CUFFT.CuFFTPlan{ComplexF32, Float32, -1, false, 2, 2, CUDA.CuArray{ComplexF32, 2, CUDA.DeviceMemory}},
                    CUDA.CUFFT.CuFFTPlan{ComplexF64, Float64, -1, false, 2, 2, CUDA.CuArray{ComplexF64, 2, CUDA.DeviceMemory}}};
global irfft_type = Union{AbstractFFTs.ScaledPlan{ComplexF64, FFTW.rFFTWPlan{ComplexF64, 1, false, 2, UnitRange{Int64}}, Float64},
    #AbstractFFTs.ScaledPlan{ComplexF32, CUDA.CUFFT.CuFFTPlan{Float32, ComplexF32, 1, false, 2, 2, CUDA.CuArray{ComplexF32, 2, CUDA.DeviceMemory}}, Float32},
    AbstractFFTs.ScaledPlan{ComplexF64, CUDA.CUFFT.CuFFTPlan{Float64, ComplexF64, 1, false, 2, 2, CUDA.CuArray{ComplexF64, 2, CUDA.DeviceMemory}}, Float64}};

global CPU_rfft_type32 = FFTW.rFFTWPlan{Float32, -1, false, 2, Tuple{Int32, Int32}};
global CPU_rfft_type64 = FFTW.rFFTWPlan{Float64, -1, false, 2, Tuple{Int64, Int64}};
global CPU_irfft_type32 = AbstractFFTs.ScaledPlan{ComplexF32, FFTW.rFFTWPlan{ComplexF32, 1, false, 2, UnitRange{Int32}}, Float32};
global CPU_irfft_type64 = AbstractFFTs.ScaledPlan{ComplexF64, FFTW.rFFTWPlan{ComplexF64, 1, false, 2, UnitRange{Int64}}, Float64};

global GPU_rfft_type32 = CUDA.CUFFT.CuFFTPlan{ComplexF32, Float32, -1, false, 2, 2, CUDA.CuArray{ComplexF32, 2, CUDA.DeviceMemory}};
global GPU_rfft_type64 = CUDA.CUFFT.CuFFTPlan{ComplexF64, Float64, -1, false, 2, 2, CUDA.CuArray{ComplexF64, 2, CUDA.DeviceMemory}};
global GPU_irfft_type32 = AbstractFFTs.ScaledPlan{ComplexF64, CUDA.CUFFT.CuFFTPlan{Float32, ComplexF32, 1, false, 2, 2, CUDA.CuArray{ComplexF32, 2, CUDA.DeviceMemory}}, Float32}
global GPU_irfft_type64 = AbstractFFTs.ScaledPlan{ComplexF64, CUDA.CUFFT.CuFFTPlan{Float64, ComplexF64, 1, false, 2, 2, CUDA.CuArray{ComplexF64, 2, CUDA.DeviceMemory}}, Float64};
