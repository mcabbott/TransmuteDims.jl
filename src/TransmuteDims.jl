module TransmuteDims

export TransmutedDimsArray, Transmute, transmutedims, transmutedims!

#=

TODO:
* how to allow efficient creation using names
* more Transmute{} shortcuts, != TransmutedDimsArray
* Efficient reductions? sum(parent) etc
* transmutedims as permutedims + reshape, on DenseArray

=#

#========== alla PermutedDimsArrays, mostly ==========#

struct TransmutedDimsArray{T,N,perm,iperm,AA<:AbstractArray,L} <: AbstractArray{T,N}
    parent::AA
end

"""
    TransmutedDimsArray(A, perm′) -> B

This is just like `PermutedDimsArray`, except that `perm′` need not be a permutation:
where it contains `0`, this inserts a trivial dimension into the output, size 1.
Any number outside `1:ndims(A)` is treated like `0`, fitting with `size(A,99) == 1`.

When `A` is a `PermutedDimsArray`, `TransmutedDimsArray` or a `Transpose{<:Number}`,
the constructor adjust adjust perm′ to work directly on `parent(A)`, and wrap that.

See also: [`transmutedims`](@ref), [`Transmute`](@ref).

# Examples
```jldoctest
julia> A = rand(3,5,4);

julia> B = TransmutedDimsArray(A, (3,0,1,2));

julia> size(B)
(4, 1, 3, 5)

julia> B[3,1,1,2] == A[1,2,3]
true
```
"""
TransmutedDimsArray(data::AbstractArray, perm) = Transmute{Tuple(perm)}(data)

perm(A::TransmutedDimsArray{T,N,P,Q} where {T,N,P,Q}) = P
iperm(A::TransmutedDimsArray{T,N,P,Q} where {T,N,P,Q}) = Q

"""
    Transmute{perm′}(A::AbstractArray)

Equivalent to `TransmutedDimsArray(A, perm′)`, but computes the inverse
(and performs sanity checks) at compile-time.
"""
struct Transmute{perm} end

Transmute{perm}(x) where {perm} = x

@generated function Transmute{perm}(data::A) where {A<:AbstractArray{T,M}} where {T,M,perm}
    perm_plus = sanitise_zero(perm, data)
    real_perm = filter(!iszero, perm_plus)
    length(real_perm) == M && isperm(real_perm) || throw(ArgumentError(
        string(real_perm, " is not a valid permutation of dimensions 1:", M,
            ". Obtained by filtering input ",perm)))

    N = length(perm_plus)
    iperm = invperm_zero(perm_plus, M)
    L = issorted(real_perm)

    :( TransmutedDimsArray{$T,$N,$perm_plus,$iperm,$A,$L}(data) )
end

using LinearAlgebra
const LazyTranspose = Union{Transpose{<:Number}, Adjoint{<:Real}}

@generated function Transmute{perm}(data::LazyTranspose) where {perm}
    new_perm = map(d -> d==1 ? 2 : d==2 ? 1 : d, perm)
    :( Transmute{$new_perm}(data.parent) )
end

LazyPermute{P} = Union{
    PermutedDimsArray{T,N,P} where {T,N},
    TransmutedDimsArray{T,N,P}  where {T,N} }

@generated function Transmute{perm}(data::LazyPermute{inner}) where {perm,inner}
    new_perm = map(d -> d==0 ? 0 : inner[d], sanitise_zero(perm, data))
    :( Transmute{$new_perm}(data.parent) )
end

Base.parent(A::TransmutedDimsArray) = A.parent

Base.size(A::TransmutedDimsArray{T,N,perm}) where {T,N,perm} =
    genperm_zero(size(parent(A)), perm)

Base.axes(A::TransmutedDimsArray{T,N,perm}) where {T,N,perm} =
    genperm_zero(axes(parent(A)), perm, Base.OneTo(1))

Base.unsafe_convert(::Type{Ptr{T}}, A::TransmutedDimsArray{T}) where {T} =
    Base.unsafe_convert(Ptr{T}, parent(A))

# It's OK to return a pointer to the first element, and indeed quite
# useful for wrapping C routines that require a different storage
# order than used by Julia. But for an array with unconventional
# storage order, a linear offset is ambiguous---is it a memory offset
# or a linear index?
Base.pointer(A::TransmutedDimsArray, i::Integer) = throw(ArgumentError(
    "pointer(A, i) is deliberately unsupported for TransmutedDimsArray"))

Base.strides(A::TransmutedDimsArray{T,N,perm}) where {T,N,perm} =
    genperm_zero(strides(parent(A)), perm, 0)

Base.dataids(A::TransmutedDimsArray) = Base.dataids(A.parent)

Base.unaliascopy(A::TransmutedDimsArray) = typeof(A)(Base.unaliascopy(A.parent))

Base.IndexStyle(A::TransmutedDimsArray{T,N,P,Q,S,L}) where {T,N,P,Q,S,L} =
    L ? IndexLinear() : IndexCartesian()
# Base.IndexStyle(A::TransmutedDimsArray{T,N,P,Q,S,L}) where {T,N,P,Q,S,L} = IndexCartesian()

@inline function Base.getindex(A::TransmutedDimsArray{T,N,perm,iperm}, I::Vararg{Int,N}) where {T,N,perm,iperm}
    @boundscheck checkbounds(A, I...)
    @inbounds val = getindex(A.parent, genperm_zero(I, iperm)...)
    val
end
@inline function Base.getindex(A::TransmutedDimsArray{T,N,P,Q,S,true}, i::Int) where {T,N,P,Q,S}
    @boundscheck checkbounds(A, i)
    getindex(A.parent, i)
end

@inline function Base.setindex!(A::TransmutedDimsArray{T,N,perm,iperm}, val, I::Vararg{Int,N}) where {T,N,perm,iperm}
    @boundscheck checkbounds(A, I...)
    @inbounds setindex!(A.parent, val, genperm_zero(I, iperm)...)
    val
end
@inline function Base.setindex!(A::TransmutedDimsArray{T,N,P,Q,S,true}, val, i::Int) where {T,N,P,Q,S}
    @boundscheck checkbounds(A, i)
    @inbounds setindex!(A.parent, val, i)
    val
end

# Not entirely sure this is a good idea, but passing KW along...
Base.@propagate_inbounds Base.getindex(A::TransmutedDimsArray; kw...) =
    getindex(A.parent; kw...)
Base.@propagate_inbounds Base.setindex!(A::TransmutedDimsArray, val; kw...) =
    setindex!(A.parent, val; kw...)


@inline genperm_zero(I::Tuple, perm::Dims{N}, gap=1) where {N} =
    ntuple(d -> perm[d]==0 ? gap : I[perm[d]], Val(N))

@inline invperm_zero(P::Tuple, M::Int) = ntuple(d -> findfirst(isequal(d),P), M)

@inline sanitise_zero(P::Tuple, A) = map(i -> i in Base.OneTo(ndims(A)) ? i : 0, P)

# https://github.com/JuliaLang/julia/pull/32968
filter(f, xs::Tuple) = Base.afoldl((ys, x) -> f(x) ? (ys..., x) : ys, (), xs...)
filter(f, t::Base.Any16) = Tuple(filter(f, collect(t)))
filter(args...) = Base.filter(args...)

transmute_str = """
    transmutedims(A, perm′)
    transmutedims!(dst, src, perm′)

Variants of `permutedims` / `permutedims!` which allow generalised permutations
like `TransmutedDimsArray`.

See also: [`TransmutedDimsArray`](@ref), [`Transmute`](@ref).
"""

@doc transmute_str
transmutedims(data::AbstractArray, perm) = collect(Transmute{Tuple(perm)}(data))

@doc transmute_str
transmutedims!(dst::AbstractArray, src::AbstractArray, perm) = copyto!(dst, Transmute{Tuple(perm)}(src))

function Base.showarg(io::IO, A::TransmutedDimsArray{T,N,perm}, toplevel) where {T,N,perm}
    print(io, "TransmutedDimsArray(")
    Base.showarg(io, parent(A), false)
    print(io, ", ", perm, ')')
    toplevel && print(io, " with eltype ", eltype(A))
end

#========== GPU etc ==========#

using GPUArrays
# https://github.com/JuliaGPU/GPUArrays.jl/blob/master/src/broadcast.jl

using Base.Broadcast
import Base.Broadcast: BroadcastStyle, Broadcasted, ArrayStyle

TransmuteGPU{AT} = TransmutedDimsArray{T,N,P,Q,AT,L} where {T,N,P,Q,L}

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

function Adapt.adapt_structure(to, A::TransmutedDimsArray{T,N,P,Q,AT,L}) where {T,N,P,Q,AT,L}
    data = adapt(to, A.parent)
    TransmutedDimsArray{eltype(data),N,P,Q,typeof(data),L}(data)
end

#========== Other Base functions ==========#

Base.dropdims(A::TransmutedDimsArray; dims) = _dropdims(A, dims...)

_dropdims(A) = A
function _dropdims(A::TransmutedDimsArray{T,N,P}, d::Int, dims...) where {T,N,P}
    if P[d]==0
        perm = ntuple(n -> n<d ? P[n] : P[n+1], N-1)
        newdims = map(n -> n<d ? n : n-1, dims)
        _dropdims(Transmute{perm}(A.parent), newdims...)
    else
        perm = ntuple(N-1) do n
            Pn = n<d ? P[n] : P[n+1]
            Pn<d ? Pn : Pn-1
        end
        newdims = map(n -> n<d ? n : n-1, dims)
        _dropdims(Transmute{perm}(dropdims(A.parent; dims=P[d])), newdims...)
    end
end

#=

@btime        dropdims($(Transmute{(2,0,1)}(rand(3,3))), dims=2)
@code_warntype dropdims(Transmute{(2,0,1)}(rand(3,3)), dims=2)

@btime        dropdims($(Transmute{(3,2,1)}(rand(3,1,3))), dims=2)
@code_warntype dropdims(Transmute{(3,2,1)}(rand(3,1,3)), dims=2)

=#

#=
# piracy!

for N=3:15
    @eval Base.transpose(x::AbstractArray{<:Number,$N}; dims) = _transpose(x, Tuple(dims))
    # @eval Base.transpose(x::AbstractArray{<:Number,$N}; dims) = _transpose(x, Val(Tuple(dims)))
end

@btime transpose($(rand(2,2,2,2)), dims=(2,3)) # 3.686 μs with Val, 259.155 ns without!

@btime TransmuteDims._transpose($(rand(2,2,2,2)), (2,3))      # 262.377 ns
@btime TransmuteDims._transpose($(rand(2,2,2,2)), Val((2,3))) #   5.617 ns

=#

function _transpose(x::AbstractArray{<:Number}, dims::Tuple{Int,Int})
    lo, hi = minimum(dims), maximum(dims)
    # hi <= ndims(x) || error("dims too large")
    perm = ntuple(d -> d==lo ? hi : d==hi ? lo : d, ndims(x))
    # TransmutedDimsArray{eltype(x),ndims(x),perm,perm,typeof(x),IndexStyle(x)===IndexLinear()}(x)
    Transmute{perm}(x)
end

@generated function _transpose(x::AbstractArray, ::Val{dims}) where {dims}
    lo, hi = minimum(dims), maximum(dims)
    perm = ntuple(d -> d==lo ? hi : d==hi ? lo : d, ndims(x))
    :( Transmute{$perm}(x) )
end

#========== Reductions ==========#

Base.sum(A::TransmutedDimsArray) = sum(A.parent)

end
