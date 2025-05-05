module TD_GPU

#========== GPU etc ==========#

using TransmuteDims, GPUArraysCore, Adapt

# https://github.com/JuliaGPU/GPUArrays.jl/blob/master/src/host/broadcast.jl

using Base.Broadcast
import Base.Broadcast: BroadcastStyle, Broadcasted, ArrayStyle

TransmuteGPU{AT} = TransmutedDimsArray{T,N,P,Q,AT} where {T,N,P,Q}

BroadcastStyle(::Type{<:TransmuteGPU{AT}}) where {AT<:AbstractGPUArray} =
    BroadcastStyle(AT)

# https://github.com/JuliaGPU/GPUArrays.jl/blob/master/src/device/indexing.jl#L77
macro linearidx(A, grididx=1, ctxsym=:ctx)
    quote
        x = $(esc(A))
        i = linear_index($(esc(ctxsym)), $(esc(grididx)))
        i > length(x) && return
        i
    end
end
macro cartesianidx(A, grididx=1, ctxsym=:ctx)
    quote
        x = $(esc(A))
        i = @linearidx(x, $(esc(grididx)), $(esc(ctxsym)))
        @inbounds CartesianIndices(x)[i]
    end
end

@inline function Base.copyto!(dest::TransmuteGPU{AT}, bc::Broadcasted{Nothing}) where {AT<:AbstractGPUArray}
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    isempty(dest) && return dest
    bc′ = Broadcast.preprocess(dest, bc)

    function broadcast_kernel(ctx, dest, bc′, nelem)
        for i in 1:nelem
            I = @cartesianidx(dest, i)
            @inbounds dest[I] = bc′[I]
        end
        return
    end
    heuristic = launch_heuristic(backend(dest), broadcast_kernel, dest, bc′, 1)
    config = launch_configuration(backend(dest), heuristic, length(dest), typemax(Int))
    gpu_call(broadcast_kernel, dest, bc′, config.elements_per_thread;
             threads=config.threads, blocks=config.blocks)

    return dest
end

@inline function Base.copyto!(
        dest::TransmuteGPU{AT},
        bc::Broadcasted{<:Broadcast.AbstractArrayStyle{0}}
    )  where {AT<:AbstractGPUArray}
    copyto!(dest, convert(Broadcasted{Nothing}, bc))
end

using Adapt

# https://github.com/JuliaGPU/GPUArrays.jl/blob/master/src/host/abstractarray.jl#L49
# display
Base.print_array(io::IO, X::TransmuteGPU{AT} where {AT <: AbstractGPUArray}) =
    Base.print_array(io, adapt(ToArray(), X))

Base._show_nonempty(io::IO, X::TransmuteGPU{AT} where {AT <: AbstractGPUArray}, prefix::String) =
    Base._show_nonempty(io, adapt(ToArray(), X), prefix)
Base._show_empty(io::IO, X::TransmuteGPU{AT} where {AT <: AbstractGPUArray}) =
    Base._show_empty(io, adapt(ToArray(), X))
Base.show_vector(io::IO, v::TransmuteGPU{AT} where {AT <: AbstractGPUArray}, args...) =
    Base.show_vector(io, adapt(ToArray(), X), args...)

# https://github.com/JuliaGPU/Adapt.jl/blob/master/src/base.jl

function Adapt.adapt_structure(to, A::TransmutedDimsArray{T,N,P,Q,AT}) where {T,N,P,Q,AT}
    data = adapt(to, A.parent)
    TransmutedDimsArray{eltype(data),N,P,Q,typeof(data)}(data)
end

end  # module
