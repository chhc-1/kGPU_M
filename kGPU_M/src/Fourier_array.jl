import CUDA
import Adapt

include("typedefs.jl")

#=
Fourier array type consisting of padded array in both real and Fourier space
File includes Fourier_array struct and the corresponding Adapt.adapt_structure
=#

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


