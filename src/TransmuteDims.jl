module TransmuteDims

export TransmutedDimsArray, transmute

#========== alla PermutedDimsArrays, mostly ==========#

struct TransmutedDimsArray{T,N,perm,iperm,AA<:AbstractArray} <: AbstractArray{T,N}
    parent::AA
end

"""
    TransmutedDimsArray(A, perm⁺)

This is just like `PermutedDimsArray`, except that `perm⁺` need not be a permutation:

* Where it contains `0`, this inserts a trivial dimension into the output, size 1.
  Any number outside `1:ndims(A)` is treated like `0`, fitting with `size(A,99) == 1`.

* Any number omitted from `perm⁺` is a dimension to be dropped, which must be of size 1.

* When a number appears twice in `perm⁺`, the result works like `Diagonal` in those dimensions.

By calling the type directly you get the simplest constructor.
Calling instead [`transmute(A, perm⁺)`](@ref) gives one which un-wraps nested objects,
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
    P = sanitise_zero(perm, Val(M))
    Q = invperm_zero(P, size(data))
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

# # The cases in which linear indexing would be desirable are those in which transmute() just reshapes.
# function Base.IndexStyle(::Type{TT}) where {TT<:TransmutedDimsArray{T,N,P,Q,AT}} where {T,N,P,Q,AT}
#     if IndexStyle(AT) === IndexLinear() && increasing_or_zero(Val(P))
#         IndexLinear()
#     else
#         IndexCartesian()
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

@inline function Base.setindex!(A::TransmutedDimsArray{T,N,perm,iperm}, val, I::Vararg{Int,N}) where {T,N,perm,iperm}
    @boundscheck checkbounds(A, I...)
    if !unique_or_zero(Val(perm)) && is_off_diag(Val(perm), I)
        iszero(val) || throw(ArgumentError(
            "cannot set off-diagonal entry $I to a nonzero value ($val)"))
    else
        @inbounds setindex!(A.parent, val, genperm_zero(I, iperm)...)
    end
    val
end

# @inline function Base.setindex!(A::TransmutedDimsArray{T,N,perm,iperm}, val, i::Int) where {T,N,perm,iperm}
#     @boundscheck checkbounds(A, i)
#     IndexStyle(A) == IndexLinear() || @warn "wtf"
#     @inbounds setindex!(A.parent, val, i)
#     val
# end

# # Not entirely sure this is a good idea, but passing KW along...
# Base.@propagate_inbounds Base.getindex(A::TransmutedDimsArray; kw...) =
#     getindex(A.parent; kw...)
# Base.@propagate_inbounds Base.setindex!(A::TransmutedDimsArray, val; kw...) =
#     setindex!(A.parent, val; kw...)

#========== Utils ==========#

@inline function sanitise_zero(perm, ::Val{M}) where {M}
    map(n -> n isa Integer && 0<n<=M ? Int(n) : 0, Tuple(perm))
end

@inline function genperm_zero(I::Tuple, perm::Dims{N}, gap=1) where {N}
    ntuple(d -> perm[d]==0 ? gap : I[perm[d]], Val(N))
end

@inline function invperm_zero(P::NTuple{N,Int}, S::NTuple{M,Integer}) where {N,M}
    Q = ntuple(M) do d
        w = ntuple(n -> P[n]==d ? n : 0, N)
        x = max_zero(w...)
        x >= 1 || S[d]==1 || throw(ArgumentError(
            "dimension $d is missing from trasmutation $P, which is not allowed when size(A, $d) = $(S[d]) != 1"))
        x
    end
end

@inline max_zero() = 0
@inline max_zero(xs...) = max(xs...)

@inline function is_off_diag(P::Tuple, I::Tuple)
    for a in 1:length(P)
        for b in 1:length(P)
            P[a] == P[b] || continue
            I[a] == I[b] || return true
        end
    end
    return false
end
@inline @generated function is_off_diag(::Val{P}, I::Tuple) where {P}  # for getindex
    out = []
    for a in 1:length(P), b in 1:length(P)
        P[a] == P[b] && push!(out,  :( I[$a] != I[$b] ))
    end
    Expr(:call, :|, out...)
end

@inline function increasing_or_zero(tup::Tuple, prev=0)  # strictly increasing!
    d = first(tup)
    (d != 0) & (d <= prev) && return false
    return increasing_or_zero(Base.tail(tup), max(d, prev))
end
@inline increasing_or_zero(::Tuple{}, prev=0) = true

@inline function unique_or_zero(perm)  # only generated version is called
    for i in 1:length(perm)
        d = perm[i]
        d == 0 && continue
        d in perm[i+1:end] && return false
    end
    return true
end
@generated unique_or_zero(::Val{perm}) where {perm} = unique_or_zero(perm)

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

may_reshape(::Type) = false
may_reshape(::Type{<:DenseArray}) = true

#========== New constructor transmute() ==========#

"""
    transmute(A, perm⁺)
    transmute(A, Val(perm⁺))

Gives a result `== TransmutedDimsArray(A, perm⁺)`, but:

* When `A` is a `PermutedDimsArray`, `Transpose{<:Number}`, `Adjoint{<:Real}`, or `Diagonal`,
  it will un-wrap this, and adjust `perm⁺` to work directly on `parent(A)`.

* When the permutation simply inserts or removes a trivial dimension, it may `reshape`
  instead of wrapping. This is controlled by `may_reshape(::Type)`.

* When possible, it prefers to create a `Transpose` or `Diagonal`, instead of `TransmutedDimsArray`.

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

julia> TransmuteDims.may_reshape(typeof(A))  # avoids reshaping wrappers
false

julia> TransmuteDims.may_reshape(typeof(B))
true
```
"""
function transmute end

# @static if VERSION > v"1.7.0-DEV.400" # not exactly right!
#     @inline Base.@aggressive_constprop transmute(data::AbstractArray, perm) = _transmute(data, sanitise_zero(perm, Val(ndims(data))))
# else
    @inline transmute(data::AbstractArray, perm) = _transmute(data, sanitise_zero(perm, Val(ndims(data))))
# end

# First dispatch is on perm, second is _transmute(::SomeArray, ::Any),
# un-wrapping all before running this:

@inline function _transmute(data::AT, P) where {AT <: AbstractArray{T,M}} where {T,M}
    Q = invperm_zero(P, size(data))
    if may_reshape(AT) && increasing_or_zero(P)
        S = map(d -> d==0 ? Base.OneTo(1) : axes(data,d), P)
        reshape(data, S)
    elseif P == (2,1) && T<:Number
        transpose(data)
    elseif P == (1,1)
        Diagonal(data)
    elseif P == 1:length(P)  # TODO: is this right? Should it be earlier?
        data
    else
        TransmutedDimsArray{T,length(P),P,Q,AT}(data)
    end
end

@inline function _transmute(data::LazyTranspose{<:AbstractVector}, perm)
    new_perm = map(d -> d==1 ? 0 : d==2 ? 1 : d, perm)
    _transmute(parent(data), new_perm)
end

@inline function _transmute(data::LazyTranspose{<:AbstractMatrix}, perm)
    new_perm = map(d -> d==1 ? 2 : d==2 ? 1 : d, perm)
    _transmute(parent(data), new_perm)
end

@inline function _transmute(data::LazyPermute{inner}, perm) where {inner}
    new_perm = map(d -> d==0 ? 0 : inner[d], perm)
    _transmute(parent(data), new_perm)
end

@inline function _transmute(data::Diagonal, perm)
    new_perm = map(d -> d==1 ? 1 : d==2 ? 1 : 0, perm)
    _transmute(parent(data), new_perm)
end

#========== Version with Val(perm) ==========#

@generated function transmute(data::AbstractArray, ::Val{perm}) where {perm}
    _trex(:data, data, sanitise_zero(perm, Val(ndims(data))))
end

# Identical list of cases:

function _trex(ex, AT, P)
    @gensym sym
    T = eltype(AT)
    Q, checks = invperm_zero(P, ndims(AT), sym)

    noreshape = if P == (2,1) && T<:Number
        :(transpose($sym))
    elseif P == (1,1)
        :(Diagonal($sym))
    elseif P == 1:length(P)
        sym
    else
        :(TransmutedDimsArray{$T,$(length(P)),$P,$Q,$AT}($sym))
    end

    if increasing_or_zero(P)
        S = map(d -> d==0 ? :(Base.OneTo(1)) : :(axes($sym,$d)), P)
        withreshape = :(reshape($sym, ($(S...),)))
        quote
            $sym = $ex
            $(checks...)
            if may_reshape($AT) # check this trait at run-time
                $withreshape
            else
                $noreshape
            end
        end
    else
        quote
            $sym = $ex
            $(checks...)
            $noreshape
        end
    end
end

function invperm_zero(P::NTuple{N,Int}, M::Int, sym::Symbol) where {N}
    checks = []
    Q = ntuple(M) do d
        w = ntuple(n -> P[n]==d ? n : 0, N)
        str = "dimension $d is missing from trasmutation $P, which is not allowed when size(A, $d) = != 1"
        x = max_zero(w...)
        x >= 1 || push!(checks, quote
            size($sym,$d)==1 || throw(ArgumentError($str))
        end)
        x
    end
    Q, checks
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
    new_perm = map(d -> d==0 ? 0 : inner[d], perm)
    _trex(:(parent($ex)), PT, new_perm)
end

function _trex(ex, ::Type{AT}, perm) where {AT<:Diagonal{T,PT}} where {T,PT}
    new_perm = map(d -> d==1 ? 1 : d==2 ? 1 : 0, perm)
    _trex(:(parent($ex)), PT, new_perm)
end

#========== The rest ==========#

# include("gpu.jl") # this takes loading from 0.4s to 1.5s. And is probably outdated.

include("base.jl")

end
