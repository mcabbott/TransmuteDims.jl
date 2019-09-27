module TransmuteDims

export TransmutedDimsArray, Transmute, transmutedims, transmutedims!

struct TransmutedDimsArray{T,N,perm,iperm,AA<:AbstractArray,L} <: AbstractArray{T,N}
    parent::AA
end

"""
    TransmutedDimsArray(A, perm′) -> B

This is just like `PermutedDimsArray`, except that `perm′` need not be a permutation:
where it contains `0`, this inserts a trivial dimension into the output, size 1.
Any number outside `1:ndims(A)` is treated like `0`, fitting with `size(A,99) == 1`.

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

"""
    Transmute{perm′}(A)

Equivalent to `TransmutedDimsArray(A, perm′)`, but computes the inverse
(and performs sanity checks) at compile-time.
"""
struct Transmute{perm} end

@generated function Transmute{perm}(data::A) where {A<:AbstractArray{T,M}} where {T,M,perm}

    perm_plus = sanitise(perm, M)
    real_perm = filter(!iszero, perm_plus)
    length(real_perm) == M && isperm(real_perm) || throw(ArgumentError(
        string(real_perm, " is not a valid permutation of dimensions 1:", M)))

    N = length(perm_plus)
    iperm = invperm_zero(perm, M)
    L = issorted(real_perm)

    :( TransmutedDimsArray{$T,$N,$perm_plus,$iperm,$A,$L}(data) )
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

Base.IndexStyle(A::TransmutedDimsArray{T,N,P,Q,S,L}) where {T,N,P,Q,S,L} =
    L ? IndexLinear() : IndexCartesian()

@inline function Base.getindex(A::TransmutedDimsArray{T,N,perm,iperm}, I::Vararg{Int,N}) where {T,N,perm,iperm}
    @boundscheck checkbounds(A, I...)
    @inbounds val = getindex(A.parent, genperm_zero(I, iperm)...)
    val
end

@inline Base.getindex(A::TransmutedDimsArray{T,N,P,Q,S,true}, i::Int) where {T,N,P,Q,S} =
    getindex(A.parent, i)

@inline function Base.setindex!(A::TransmutedDimsArray{T,N,perm,iperm}, val, I::Vararg{Int,N}) where {T,N,perm,iperm}
    @boundscheck checkbounds(A, I...)
    @inbounds setindex!(A.parent, val, genperm_zero(I, iperm)...)
    val
end

@inline Base.setindex!(A::TransmutedDimsArray{T,N,P,Q,S,true}, val, i::Int) where {T,N,P,Q,S} =
    setindex!(A.parent, val, i)

@inline genperm_zero(I::Tuple, perm::Dims{N}, gap=1) where {N} =
    ntuple(d -> perm[d]==0 ? gap : I[perm[d]], Val(N))

@inline invperm_zero(P::Tuple, M::Int) = ntuple(d -> findfirst(isequal(d),P), M)

@inline sanitise(P::Tuple, M::Int) = map(i -> i in Base.OneTo(M) ? i : 0, P)

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

#=
TODO:
* Efficient reductions
* transmutedims as permutedims + reshape
=#

end
