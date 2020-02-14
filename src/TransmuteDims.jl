module TransmuteDims

export transmutedims, transmutedims!, Transmute, transmute, TransmutedDimsArray

using Compat # v3.1, for filter(f, Tuple)

#=

TODO:
* how to allow efficient creation using names
* Efficient reductions? sum(parent) etc

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

See also: [`Transmute`](@ref) (faster constructor),
[`transmutedims`](@ref) (eager version)

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
function TransmutedDimsArray(data::AT, perm) where {AT <: AbstractArray{T,M}} where {T,M}
    #=
    P = map(n -> n in 1:M ? n : 0, Tuple(perm))
    for d in 1:M
        count(isequal(d), P) == 1 || throw(ArgumentError(
            "Every number in 1:$M must occur exactly once in (sanitised) transmutation $P, but $d appears $(count(isequal(d), P)) times."))
    end
    Q = invperm_zero(P, M)
    L = IndexStyle(data) === IndexLinear() &&
        all(Q[d] < Q[d+1] for d in 1:length(Q)-1)
    TransmutedDimsArray{T,length(perm),P,Q,AT,L}(data)
    =#
    transmute(data, perm)
end

perm(A::TransmutedDimsArray{T,N,P,Q} where {T,N,P,Q}) = P
iperm(A::TransmutedDimsArray{T,N,P,Q} where {T,N,P,Q}) = Q

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
    @inbounds val = ifelse(off_diag(Val(perm), I), zero(T), getindex(A.parent, genperm_zero(I, iperm)...))
    # @inbounds val = off_diag(Val(perm), I) ? zero(T) : getindex(A.parent, genperm_zero(I, iperm)...)
    # not sure which is better
    val
end
@inline function Base.getindex(A::TransmutedDimsArray{T,N,P,Q,S,true}, i::Int) where {T,N,P,Q,S}
    @boundscheck checkbounds(A, i)
    getindex(A.parent, i)
end

@inline function Base.setindex!(A::TransmutedDimsArray{T,N,perm,iperm}, val, I::Vararg{Int,N}) where {T,N,perm,iperm}
    @boundscheck checkbounds(A, I...)
    if off_diag(Val(perm), I)
        iszero(val) || throw(ArgumentError(
            "cannot set off-diagonal entry $I to a nonzero value ($val)"))
    end
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

@inline function off_diag(P::Tuple, I::Tuple)
    for a in 1:length(P)
        for b in 1:length(P)
            P[a] == P[b] || continue
            I[a] == I[b] || return true
        end
    end
    false
end
@inline @generated function off_diag(::Val{P}, I::Tuple) where {P}
    out = []
    for a in 1:length(P), b in 1:length(P)
        P[a] == P[b] && push!(out,  :( I[$a] != I[$b] ))
    end
    Expr(:call, :|, out...)
end

function Base.showarg(io::IO, A::TransmutedDimsArray{T,N,perm}, toplevel) where {T,N,perm}
    print(io, "TransmutedDimsArray(")
    Base.showarg(io, parent(A), false)
    print(io, ", ", perm, ')')
    toplevel && print(io, " with eltype ", eltype(A))
end

#========== Constructors Transmute{} and transmute() ==========#

"""
    Transmute{perm′}(A::AbstractArray)
    transmute(A, Val(perm′))

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

transmute(data::AbstractArray, ::Val{perm}) where {perm} = Transmute{perm}(data)

"""
    transmute(A, perm′)

Equivalent to `TransmutedDimsArray(A, perm′)`,
but with some effort into making constant propagation work.
"""
function transmute(data::AT, perm) where {AT <: AbstractArray{T,M}} where {T,M}
    P,Q,L = _transmute_calc(data, Tuple(perm))

    # P = map(n -> n isa Int && 0<n<=M ? n : 0, perm)
    # P = sanitise_p(data, perm)

    # Q = ps_inverse(data, P)

    # L = Base.afoldl(<, Q...) # this is wrong

    TransmutedDimsArray{T,length(perm),P,Q,AT,L}(data)
end


#=
sanitise_p(A::AbstractArray{T,M}, tup::Tuple{Vararg{Any,N}}) where {T,M,N} =
     map(n -> n isa Int && 0<n<=M ? n : 0, tup)

ps_inverse(A::AbstractArray{T,M}, P::Tuple{Vararg{Any,N}}) where {T,M,N} =
    ntuple(M) do d
        # c = ntuple(n -> P[n]==d ? 1 : 0, N)
        # sum(c)==1 || error("Every number in 1:$N must occur exactly once in (sanitised) trasmutation $P, but one appears $(sum(c)) times.")

        w = ntuple(n -> P[n]==d ? n : 0, N)
        sum(w)
    end
=#

function _transmute_calc(A::AbstractArray{T,M}, tup::Tuple{Vararg{Any,N}}) where {T,M,N}
# function _transmute_calc(A::AbstractArray{T,M}, tup::NTuple{N, Int}) where {T,M,N}
    # Sanitise input "perm"
    P = map(n -> n isa Int && 0<n<=M ? n : 0, tup)

    # Calculate sort-of inverse
    Q = ntuple(M) do d
        c = ntuple(n -> P[n]==d ? 1 : 0, N)
        sum(c)==1 || throw(ArgumentError(
            "Every number in 1:$N must occur exactly once in (sanitised) trasmutation $P, but one appears $(sum(c)) times."))

        w = ntuple(n -> P[n]==d ? n : 0, N)
        sum(w)
    end

    # Linear indexing?
    L = (IndexStyle(A) === IndexLinear()) & are_increasing(Q...)

    return P,Q,L
end
# @btime (()-> TransmuteDims._transmute_calc($(ones(1,1,1)), (1,5,3,2,4)) )()  # 0.029 ns, 0 allocations
# @btime (()-> TransmuteDims._transmute_calc($(ones(1,1,1)), (2,nothing,3,1,0)) )()

# @btime transmute($(ones(1,1,1)), (0,2,3,1));  # slow!
# @btime (() -> transmute($(ones(1,1,1)), (0,2,3,1)))();  # slow!
# @code_warntype (() -> TransmutedDimsArray(ones(1,1,1), (0,2,3,1)))() # TDA{Float64,4,_A,_B,Array{...

are_increasing(x, y, zs...) = (x<y) & are_increasing(y, zs...)
are_increasing(x) = true

#========== transmutedims() uses permutedims() ==========#

_transmutedims_doc = """
    transmutedims(A, perm′)
    transmutedims!(dst, src, perm′)

These are just like `permutedims` / `permutedims!`, except that `perm′` need not
be a permutation of `1:ndims(A)`. Any number outside this range inserts a trivial
dimension into the output, size 1, fitting with `size(A,99) == 1`.

They are implemented just as permutation + `reshape`.

See also: [`TransmutedDimsArray`](@ref), [`Transmute`](@ref).

# Examples
```jldoctest
julia> A = rand(3,5,4);

julia> B = transmutedims(A, (3,4,1,2)); # A valid permutation of 1:4

julia> size(B)
(4, 1, 3, 5)

julia> B[3,1,1,2] == A[1,2,3]
true

julia> B == permutedims(reshape(A,size(A)...,1), (3,4,1,2))
true

julia> C = zeros(axes(B));

julia> transmutedims!(C, A, (3,0,1,2)); # OK to replace 4 with 0

julia> C == B
true
```
"""

@doc _transmutedims_doc
function transmutedims(src::AbstractArray{T,N}, perm) where {T,N}
    length(perm) >= ndims(src) || throw(ArgumentError("length(perm) is less than ndims(src)"))
    safe_perm = filter(d -> d in 1:N, Tuple(perm))#::NTuple{N, Int}
    final_ax = map(d -> d in 1:N ? axes(src, d) : Base.OneTo(1), Tuple(perm))
    if safe_perm == 1:ndims(src)
        reshape(copy(src), final_ax...)
    else
        reshape(permutedims(src, safe_perm), final_ax...)
    end
end

@doc _transmutedims_doc
function transmutedims!(dst::AbstractArray, src::AbstractArray{T,N}, perm::Tuple) where {T,N}
    length(perm) == ndims(dst) || throw(ArgumentError("length(perm) does not match ndims(dst)"))
    safe_perm = filter(d -> d in 1:N, Tuple(perm))
    permed_ax = map(d -> axes(src, d), safe_perm)
    for (i,d) in enumerate(perm)
        if d in 1:N
            axes(dst, i) == axes(src, d) || throw(DimensionMismatch("destination tensor of incorrect size"))
        end
    end
    if safe_perm == 1:ndims(src)
        copyto!(dst, src)
    else
        permutedims!(reshape(dst, permed_ax...), src, safe_perm)
    end
    dst
end

#========== The rest ==========#

include("gpu.jl")

include("base.jl")

end
