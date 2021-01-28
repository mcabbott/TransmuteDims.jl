module TransmuteDims

export TransmutedDimsArray, transmute

#========== alla PermutedDimsArrays, mostly ==========#

struct TransmutedDimsArray{T,N,perm,iperm,AA<:AbstractArray} <: AbstractArray{T,N}
    parent::AA
end

"""
    TransmutedDimsArray(A, perm′) -> B

This is just like `PermutedDimsArray`, except that `perm′` need not be a permutation:

* Where it contains `0`, this inserts a trivial dimension into the output, size 1.
  Any number outside `1:ndims(A)` is treated like `0`, fitting with `size(A,99) == 1`.

* When number appears twice in `perm′`, then this works like `Diagonal` in those dimensions.

By calling the type directly you get the simplest constructor.
Calling instead [`transmute`](@ref) gives one which un-wraps nested objects,
picks simpler alternatives, etc.

# Examples
```jldoctest
julia> TransmutedDimsArray('a':'c', (2,1)) # fails!!

julia> TransmutedDimsArray([3,5,7,11], (2,1))  # like transpose
1×4 transmute(::Vector{Int64}, (0, 1)) with eltype Int64:
 3  5  7  11

julia> A = rand(10, 20, 30);

julia> B = TransmutedDimsArray(A, (3,0,1,2));  # like reshape + permute

julia> size(B)
(30, 1, 10, 20)

julia> B[3,1,1,2] == A[1,2,3] == A[1,2,3,1]  # implicit trailing dimensions
true

julia> B == TransmutedDimsArray(A, (3,4,1,2)) == permutedims(A[:,:,:,:], (3,4,1,2))
true

julia> TransmutedDimsArray(reshape(1:6,3,2), (1,1,2))  # generalised Diagonal
3×3×2 transmute(reshape(::UnitRange{Int64}, 3, 2), (1, 1, 2)) with eltype Int64:
[:, :, 1] =
 1  ⋅  ⋅
 ⋅  2  ⋅
 ⋅  ⋅  3

[:, :, 2] =
 4  ⋅  ⋅
 ⋅  5  ⋅
 ⋅  ⋅  6
```
"""
function TransmutedDimsArray(data::AT, perm) where {AT <: AbstractArray{T,M}} where {T,M}
    P = map(n -> n in 1:M ? n : 0, Tuple(perm))
    for d in 1:M
        count(isequal(d), P) >= 1 || throw(ArgumentError(
            "Every number in 1:$M must appear at least once in trasmutation $P, but $d is missing"))
    end
    Q = invperm_zero(P, M)
    TransmutedDimsArray{T,length(perm),P,Q,AT}(data)
end

getperm(A::TransmutedDimsArray{T,N,P,Q}) where {T,N,P,Q} = P
getinvperm(A::TransmutedDimsArray{T,N,P,Q}) where {T,N,P,Q} = Q

Base.parent(A::TransmutedDimsArray) = A.parent

Base.size(A::TransmutedDimsArray{T,N,perm}) where {T,N,perm} =
    genperm_zero(size(parent(A)), perm)

Base.axes(A::TransmutedDimsArray{T,N,perm}) where {T,N,perm} =
    genperm_zero(axes(parent(A)), perm, Base.OneTo(1))

Base.similar(A::TransmutedDimsArray, T::Type, dims::Base.Dims) = similar(parent(A), T, dims)

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

# @generated function Base.IndexStyle(::Type{TT}) where {TT<:TransmutedDimsArray{T,N,P,Q,AT}} where {T,N,P,Q,AT}
#     if IndexStyle(AT) == IndexLinear() && increasing_or_zero(P)
#         :(IndexLinear())
#     else
#         :(IndexCartesian())
#     end
# end

@inline function Base.getindex(A::TransmutedDimsArray{T,N,perm,iperm}, I::Vararg{Int,N}) where {T,N,perm,iperm}
    @boundscheck checkbounds(A, I...)
    val = @inbounds getindex(A.parent, genperm_zero(I, iperm)...)
    if unique_or_zero(Val(perm))
        val
    else
        ifelse(is_off_diag(Val(perm), I), zero(T), val)
    end
end

# @inline function Base.getindex(A::TransmutedDimsArray{T,N,perm,iperm}, i::Int) where {T,N,perm,iperm}
#     @boundscheck checkbounds(A, i)
#     if IndexStyle(A) == IndexLinear() || @warn "wtf"
#     val = @inbounds getindex(A.parent, i)
# end

# @inline function Base.getindex(A::TransmutedDimsArray{T,N,P,Q,S,true}, i::Int) where {T,N,P,Q,S}
#     @boundscheck checkbounds(A, i)
#     getindex(A.parent, i)
# end # plus one more method to resolve an ambiguity:
# @inline function Base.getindex(A::TransmutedDimsArray{T,1,P,Q,S,true}, i::Int) where {T,P,Q,S}
#     @boundscheck checkbounds(A, i)
#     getindex(A.parent, i)
# end

@inline function Base.setindex!(A::TransmutedDimsArray{T,N,perm,iperm}, val, I::Vararg{Int,N}) where {T,N,perm,iperm}
    @boundscheck checkbounds(A, I...)
    if !unique_or_zero(Val(perm)) && is_off_diag(Val(perm), I)
        iszero(val) || throw(ArgumentError(
            "cannot set off-diagonal entry $I to a nonzero value ($val)"))
    end
    @inbounds setindex!(A.parent, val, genperm_zero(I, iperm)...)
    val
end

# @inline function Base.setindex!(A::TransmutedDimsArray{T,N,perm,iperm}, val, i::Int) where {T,N,perm,iperm}
#     @boundscheck checkbounds(A, i)
#     IndexStyle(A) == IndexLinear() || @warn "wtf"
#     @inbounds setindex!(A.parent, val, i)
#     val
# end

# @inline function Base.setindex!(A::TransmutedDimsArray{T,N,P,Q,S,true}, val, i::Int) where {T,N,P,Q,S}
#     @boundscheck checkbounds(A, i)
#     @inbounds setindex!(A.parent, val, i)
#     val
# end

# Not entirely sure this is a good idea, but passing KW along...
# Base.@propagate_inbounds Base.getindex(A::TransmutedDimsArray; kw...) =
#     getindex(A.parent; kw...)
# Base.@propagate_inbounds Base.setindex!(A::TransmutedDimsArray, val; kw...) =
#     setindex!(A.parent, val; kw...)


@inline genperm_zero(I::Tuple, perm::Dims{N}, gap=1) where {N} =
    ntuple(d -> perm[d]==0 ? gap : I[perm[d]], Val(N))

@inline invperm_zero(P::Tuple, M::Int) = ntuple(d -> findfirst(isequal(d),P), M)

@inline sanitise_zero(P::Tuple, A) = map(i -> i in Base.OneTo(ndims(A)) ? i : 0, P)

@inline function is_off_diag(P::Tuple, I::Tuple)
    for a in 1:length(P)
        for b in 1:length(P)
            P[a] == P[b] || continue
            I[a] == I[b] || return true
        end
    end
    false
end
@inline @generated function is_off_diag(::Val{P}, I::Tuple) where {P}
    out = []
    for a in 1:length(P), b in 1:length(P)
        P[a] == P[b] && push!(out,  :( I[$a] != I[$b] ))
    end
    Expr(:call, :|, out...)
end

#========== Printing ==========#

function Base.showarg(io::IO, A::TransmutedDimsArray{T,N,perm}, toplevel) where {T,N,perm}
    print(io, "transmute(")
    Base.showarg(io, parent(A), false)
    print(io, ", ", perm, ')')
    toplevel && print(io, " with eltype ", eltype(A))
end

# This is from  julia/stdlib/v1.3/LinearAlgebra/src/diagonal.jl:109
# function Base.replace_in_print_matrix(A::Diagonal,i::Integer,j::Integer,s::AbstractString)
function Base.replace_in_print_matrix(A::TransmutedDimsArray,
        i::Integer,j::Integer,s::AbstractString)
    is_off_diag(getperm(A), (i,j)) ? Base.replace_with_centered_mark(s) : s
end
function Base.replace_in_print_matrix(A::SubArray{<:Any,2,<:TransmutedDimsArray},
        i::Integer,j::Integer,s::AbstractString) where {S}
    ijk = (i, j, A.indices[3:end]...)
    is_off_diag(getperm(A.parent), ijk) ? Base.replace_with_centered_mark(s) : s
end

#========== Types ==========#

using LinearAlgebra

LazyTranspose{AT} = Union{
    Transpose{T,AT} where T<:Number,
    Adjoint{T,AT} where T<:Real,
}

getperm(A::LazyTranspose{<:AbstractVector}) = (0,1)
getperm(A::LazyTranspose{<:AbstractMatrix}) = (2,1)
getinvperm(A::LazyTranspose{<:AbstractVector}) = (2,)
getinvperm(A::LazyTranspose{<:AbstractMatrix}) = (2,1)

LazyPermute{P,AT} = Union{
    PermutedDimsArray{T,N,P,Q,AT} where {T,N,Q},
    TransmutedDimsArray{T,N,P,Q,AT} where {T,N,Q},
}

getperm(A::PermutedDimsArray{T,N,P}) where {T,N,P} = P
getinvperm(A::PermutedDimsArray{T,N,P,Q}) where {T,N,P,Q} = Q

#========== New constructor transmute() ==========#

"""
    transmute(A, perm′)
    transmute(A, Val(perm′))

Similar to `TransmutedDimsArray(A, perm′)`, but:

* When `A` is a `PermutedDimsArray`, `Transpose{<:Number}`, `Adjoint{<:Real}`, or `Diagonal`,
  the constructor will adjust `perm′` to work directly on `parent(A)`.

* When the permutation simply inserts a trivial dimension, it may `reshape` instead of wrapping.

* If the permutation is wrapped in `Val`, then computing its inverse (and performing sanity checks)
  can be done at compile-time.

# Examples
```jldoctest; setup=:(using Random; Random.seed!(42);)
julia> A = transpose(rand(Int8, 4, 2))
2×4 transpose(::Matrix{Int8}) with eltype Int8:
 -112  -108   -92   -54
  106   -75  -112  -105

julia> B = transmute(A, (3,2,1))  # unwraps transpose, and then reshapes
1×4×2 Array{Int8, 3}:
[:, :, 1] =
 -112  -108  -92  -54

[:, :, 2] =
 106  -75  -112  -105

julia> B == TransmutedDimsArray(A, (0,2,1))  # same values, different type
true

julia> transmute(A, (2,2,0,1))
4×4×1×2 transmute(::Matrix{Int8}, (1, 1, 0, 2)) with eltype Int8:
[:, :, 1, 1] =
 -112     ⋅    ⋅    ⋅
    ⋅  -108    ⋅    ⋅
    ⋅     ⋅  -92    ⋅
    ⋅     ⋅    ⋅  -54

[:, :, 1, 2] =
 106    ⋅     ⋅     ⋅
   ⋅  -75     ⋅     ⋅
   ⋅    ⋅  -112     ⋅
   ⋅    ⋅     ⋅  -105

julia> ans == transmute(B, (2,2,1,3))
true
```
"""
transmute(data::AbstractArray, perm) = _transmute(data, Tuple(perm))

# First dispatch is on perm, second is _transmute(::SomeArray, ::Any)

function _transmute(data::AT, perm) where {AT <: AbstractArray{T,M}} where {T,M}
    P,Q = _calc_perms(data, Tuple(perm))
    if P == (2,1) && T<:Number
        transpose(data)
    elseif P == (1,1)
        Diagonal(data)
    elseif P == 1:length(P)
        data
    else
        TransmutedDimsArray{T,length(P),P,Q,AT}(data)
    end
end

# Reshaping instead of wrapping:

function _transmute(data::AT, perm) where {AT <: DenseArray{T,M}} where {T,M}
    P = sanitise_zero(perm, data)
    if increasing_or_zero(P)
        for d in 1:M
            count(isequal(d), P) >= 1 || throw(ArgumentError(
                "Every number in 1:$M must appear at least once in trasmutation $P, but $d is missing"))
        end
        S = map(d -> d==0 ? Base.OneTo(1) : axes(data,d), P)
        reshape(data, S)
    else
        invoke(_transmute, Tuple{AbstractArray, Any}, data, P)
    end
end

# Unwapping of other transpose etc:

function _transmute(data::LazyTranspose{<:AbstractVector}, perm)
    new_perm = map(d -> d==1 ? 0 : d==2 ? 1 : d, perm)
    _transmute(parent(data), new_perm)
end

function _transmute(data::LazyTranspose{<:AbstractMatrix}, perm)
    new_perm = map(d -> d==1 ? 2 : d==2 ? 1 : d, perm)
    _transmute(parent(data), new_perm)
end

function _transmute(data::LazyPermute{inner}, perm) where {inner}
    new_perm = map(d -> d==0 ? 0 : inner[d], sanitise_zero(perm, data))
    _transmute(parent(data), new_perm)
end

function _transmute(data::Diagonal, perm)
    new_perm = map(d -> d==1 ? 1 : d==2 ? 1 : 0, perm)
    _transmute(parent(data), new_perm)
end

#========== Version with Val(perm) ==========#

@generated function transmute(data::AbstractArray, ::Val{perm}) where {perm}
    _trex(:data, data, perm)
end

# Identical list of cases:

function _trex(ex, AT, perm) #  ::Type{AT}, perm) where {AT}
    T = eltype(AT)
    P,Q = _calc_perms(AT, perm)

    if P == (2,1) && T<:Number
        :(transpose($ex))
    elseif P == (1,1)
        :(Diagonal($ex))
    elseif P == 1:length(P)
        ex
    else
        :(TransmutedDimsArray{$T,$(length(P)),$P,$Q,$AT}($ex))
    end
end

function _trex(ex, ::Type{AT}, perm) where {AT <: DenseArray{T,M}} where {T,M}
    P = sanitise_zero(perm, AT)
    if increasing_or_zero(P)
        for d in 1:M
            count(isequal(d), P) >= 1 || throw(ArgumentError(
                "Every number in 1:$M must appear at least once in trasmutation $P, but $d is missing"))
        end
        S = map(d -> d==0 ? :(Base.OneTo(1)) : :(axes($ex,$d)), P)
        :(reshape($ex, ($(S...),)))
    else
        invoke(_trex, Tuple{Any, Any, Any}, ex, AT, P)
    end
end

function _trex(ex, ::Type{AT}, perm) where {AT<:LazyTranspose{PT}} where {PT<:AbstractVector}
    new_perm = map(d -> d==1 ? 0 : d==2 ? 1 : d, perm)
    _trex(:(parent($ex)), PT, new_perm)
end

function _trex(ex, ::Type{AT}, perm) where {AT<:LazyTranspose{PT}} where {PT<:AbstractMatrix}
    new_perm = map(d -> d==1 ? 2 : d==2 ? 1 : d, perm)
    _trex(:(parent($ex)), PT, new_perm)
end

function _trex(ex, ::Type{AT}, perm) where {AT<:LazyPermute{inner,PT}} where {inner,PT}
    new_perm = map(d -> d==0 ? 0 : inner[d], sanitise_zero(perm, AT))
    _trex(:(parent($ex)), PT, new_perm)
end

function _trex(ex, ::Type{AT}, perm) where {AT<:Diagonal{T,PT}} where {T,PT}
    new_perm = map(d -> d==1 ? 1 : d==2 ? 1 : 0, perm)
    _trex(:(parent($ex)), PT, new_perm)
end

#========== Utils ==========#

@inline function increasing_or_zero(perm)
    prev = 0
    for d in perm
        d == 0 && continue
        d <= prev && return false
        prev = max(prev, d)
    end
    return true
end
@generated increasing_or_zero(::Val{perm}) where {perm} = increasing_or_zero(perm)

@inline function unique_or_zero(perm)
    for i in 1:length(perm)
        d = perm[i]
        d == 0 && continue
        d in perm[i+1:end] && return false
    end
    return true
end
@generated unique_or_zero(::Val{perm}) where {perm} = unique_or_zero(perm)
# unique_or_zero(perm::Tuple) = _unique_or_zero(perm...)
# _unique_or_zero(q) = true
# function _unique_or_zero(p, qs...) # This works but isn't great
#     rest = _unique_or_zero(qs...)
#     iszero(p) && return rest
#     p in qs && return false
#     return rest
# end

_calc_perms(data::AbstractArray, tup::Tuple) = _calc_perms(typeof(data), tup)

function _calc_perms(::Type{AT}, tup::Tuple) where {AT<:AbstractArray{T,M}} where {T,M}
    N = length(tup)

    # Sanitise input "perm"
    P = map(n -> n isa Int && 0<n<=M ? n : 0, tup)

    # Calculate sort-of inverse
    Q = ntuple(M) do d
        c = ntuple(n -> P[n]==d ? 1 : 0, N)
        sum(c)>=1 || throw(ArgumentError(
            "Every number in 1:$M must appear at least once in trasmutation $P, but $d is missing"))

        w = ntuple(n -> P[n]==d ? n : 0, N)
        maximum(w)
    end

    return P,Q
end

#========== The rest ==========#

include("gpu.jl")

include("base.jl")

end
