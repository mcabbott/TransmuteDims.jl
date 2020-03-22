#========== GPU etc ==========#

using GPUArrays
# https://github.com/JuliaGPU/GPUArrays.jl/blob/master/src/broadcast.jl

using Base.Broadcast
import Base.Broadcast: BroadcastStyle, Broadcasted, ArrayStyle

TransmuteGPU{AT} = TransmutedDimsArray{T,N,P,Q,AT} where {T,N,P,Q}

BroadcastStyle(::Type{<:TransmuteGPU{AT}}) where {AT<:GPUArray} =
    BroadcastStyle(AT)

GPUArrays.backend(::Type{<:TransmuteGPU{AT}}) where {AT<:GPUArray} =
    GPUArrays.backend(AT)

@inline function Base.copyto!(dest::TransmuteGPU, bc::Broadcasted{Nothing})
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    bc′ = Broadcast.preprocess(dest, bc)
    gpu_call(dest, (dest, bc′)) do state, dest, bc′
        let I = CartesianIndex(@cartesianidx(dest))
            @inbounds dest[I] = bc′[I]
        end
        return
    end

    return dest
end

@inline Base.copyto!(dest::TransmuteGPU, bc::Broadcasted{<:Broadcast.AbstractArrayStyle{0}}) =
    copyto!(dest, convert(Broadcasted{Nothing}, bc))

# https://github.com/JuliaGPU/GPUArrays.jl/blob/master/src/abstractarray.jl#L53
# display
Base.print_array(io::IO, X::TransmuteGPU{AT} where {AT <: GPUArray}) =
    Base.print_array(io, GPUArrays.cpu(X))

# show
Base._show_nonempty(io::IO, X::TransmuteGPU{AT} where {AT <: GPUArray}, prefix::String) =
    Base._show_nonempty(io, GPUArrays.cpu(X), prefix)
Base._show_empty(io::IO, X::TransmuteGPU{AT} where {AT <: GPUArray}) =
    Base._show_empty(io, GPUArrays.cpu(X))
Base.show_vector(io::IO, v::TransmuteGPU{AT} where {AT <: GPUArray}, args...) =
    Base.show_vector(io, GPUArrays.cpu(X), args...)

using Adapt
# https://github.com/JuliaGPU/Adapt.jl/blob/master/src/base.jl

function Adapt.adapt_structure(to, A::TransmutedDimsArray{T,N,P,Q,AT}) where {T,N,P,Q,AT}
    data = adapt(to, A.parent)
    TransmutedDimsArray{eltype(data),N,P,Q,typeof(data)}(data)
end
