module TransmuteDims

export transmutedims, transmutedims!, Transmute, transmute

using Compat # v3.1, for filter(f, Tuple)

#=

TODO:
* Move linear indexing story from type to IndexStyle
* Simplify to transmute(A, p) & transmute(A, val(p)) only.
* Unwrapping for transmute(A, p) too
* Efficient creation using names?
* Efficient reductions? sum(parent) etc
* Add a function such as transpose, adjoint, conj... to type + getindex? Hmm.

=#

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

julia> transmute(reshape(1:6,:,2), (1,1,2))
3×3×2 TransmutedDimsArray(reshape(::UnitRange{Int64}, 3, 2), (1, 1, 2)) with eltype Int64:
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
            "Every number in 1:$M must appear at least once in trasmutation $P, bud $d is missing"))
    end
    Q = invperm_zero(P, M)
    # L = IndexStyle(data) === IndexLinear() &&
    #     allunique(filter(!iszero, P)) &&
    #     issorted(Q)
    TransmutedDimsArray{T,length(perm),P,Q,AT}(data)
end

perm(A::TransmutedDimsArray{T,N,P,Q}) where {T,N,P,Q} = P
iperm(A::TransmutedDimsArray{T,N,P,Q}) where {T,N,P,Q} = Q

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

# Base.IndexStyle(A::TransmutedDimsArray{T,N,P,Q,S,L}) where {T,N,P,Q,S,L} =
#     L ? IndexLinear() : IndexCartesian()
# Base.IndexStyle(A::TransmutedDimsArray{T,N,P,Q,S,L}) where {T,N,P,Q,S,L} = IndexCartesian()

@inline function Base.getindex(A::TransmutedDimsArray{T,N,perm,iperm}, I::Vararg{Int,N}) where {T,N,perm,iperm}
    @boundscheck checkbounds(A, I...)
    @inbounds val = ifelse(off_diag(Val(perm), I), zero(T), getindex(A.parent, genperm_zero(I, iperm)...))
    # @inbounds val = off_diag(Val(perm), I) ? zero(T) : getindex(A.parent, genperm_zero(I, iperm)...)
    # not sure which is better
    val
end

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
    if off_diag(Val(perm), I)
        iszero(val) || throw(ArgumentError(
            "cannot set off-diagonal entry $I to a nonzero value ($val)"))
    end
    @inbounds setindex!(A.parent, val, genperm_zero(I, iperm)...)
    val
end
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
    print(io, "transmute(")
    Base.showarg(io, parent(A), false)
    print(io, ", ", perm, ')')
    toplevel && print(io, " with eltype ", eltype(A))
end

# This is from  julia/stdlib/v1.3/LinearAlgebra/src/diagonal.jl:109
# function Base.replace_in_print_matrix(A::Diagonal,i::Integer,j::Integer,s::AbstractString)
function Base.replace_in_print_matrix(A::TransmutedDimsArray,
        i::Integer,j::Integer,s::AbstractString)
    off_diag(perm(A), (i,j)) ? Base.replace_with_centered_mark(s) : s
end
function Base.replace_in_print_matrix(A::SubArray{<:Any,2,<:TransmutedDimsArray},
        i::Integer,j::Integer,s::AbstractString) where {S}
    ijk = (i, j, A.indices[3:end]...)
    off_diag(perm(A.parent), ijk) ? Base.replace_with_centered_mark(s) : s
end

function Base.show_nd(io::IO, A::TransmutedDimsArray, print_matrix::Function, label_slices::Bool)
    allunique(filter(!iszero, perm(A))) &&
        Core.invoke(Base.show_nd, Tuple{IO, AbstractArray, Function, Bool}, io, A, print_matrix, label_slices)
    # Only run this version on generalised diagonal things?

    ii = ntuple(d->firstindex(A, d+2), ndims(A)-2)
    println(io, "[:, :, ", join(ii, ", "), "] =")
    print_matrix(io, view(A, :, :, ii...))

    prod(size(A)[3:end]) == 1 && return nothing
    println(io, "\n")

    jj = ntuple(d->lastindex(A, d+2), ndims(A)-2)
    println(io, "[:, :, ", join(jj, ", "), "] =")
    print_matrix(io, view(A, :, :, jj...))
end


#========== Constructors Transmute{} and transmute() ==========#

"""
    Transmute{perm′}(A::AbstractArray)
    transmute(A, Val(perm′))

Equivalent to `TransmutedDimsArray(A, perm′)`, but computes the inverse
(and performs sanity checks) at compile-time.

When `A` is a `PermutedDimsArray`, `TransmutedDimsArray` or a `Transpose{<:Number}`,
the constructor adjust adjust perm′ to work directly on `parent(A)`, and wrap that.

"""
struct Transmute{perm} end

Transmute{perm}(x) where {perm} = x

@generated function Transmute{perm}(data::A) where {A<:AbstractArray{T,M}} where {T,M,perm}
    perm_plus = sanitise_zero(perm, data)
    for d in 1:M
        count(isequal(d), perm_plus) >= 1 || throw(ArgumentError(
            "Every number in 1:$M must appear at least once in trasmutation $perm_plus, bud $d is missing"))
    end

    iperm = invperm_zero(perm_plus, M)

    N = length(perm_plus)

    # L = issorted(iperm) &&
    #     allunique(filter(!iszero, perm_plus)) &&
    #     IndexStyle(data) === IndexLinear()

    :( TransmutedDimsArray{$T,$N,$perm_plus,$iperm,$A}(data) )
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
    P,Q = _transmute_calc(data, Tuple(perm))

    # P = map(n -> n isa Int && 0<n<=M ? n : 0, perm)
    # P = sanitise_p(data, perm)

    # Q = ps_inverse(data, P)

    TransmutedDimsArray{T,length(perm),P,Q,AT}(data)
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
        sum(c)>=1 || throw(ArgumentError(
            "Every number in 1:$M must appear at least once in trasmutation $P, bud $d is missing"))

        w = ntuple(n -> P[n]==d ? n : 0, N)
        maximum(w)
    end

    # Linear indexing?
    # L = (IndexStyle(A) === IndexLinear()) & all_unique(filter(!iszero, P)...) & are_increasing(Q...)

    return P,Q
end
# @btime (()-> TransmuteDims._transmute_calc($(ones(1,1,1)), (1,5,3,2,4)) )()  # 0.029 ns, 0 allocations
# @btime (()-> TransmuteDims._transmute_calc($(ones(1,1,1)), (2,nothing,3,1,0)) )()

# @btime transmute($(ones(1,1,1)), (0,2,3,1));  # slow!
# @btime (() -> transmute($(ones(1,1,1)), (0,2,3,1)))();  # slow!
# @code_warntype (() -> TransmutedDimsArray(ones(1,1,1), (0,2,3,1)))() # TDA{Float64,4,_A,_B,Array{...

are_increasing(x, y, zs...) = (x<y) & are_increasing(y, zs...)
are_increasing(x) = true

all_unique(p, qs...) = !(p in qs) & all_unique(qs...)
all_unique(x) = true

# @btime allunique((1,2,3))  # 135.309 ns (4 allocations: 480 bytes)
# @btime all_unique((1,2,3)) #   1.975 ns (0 allocations: 0 bytes)

#========== transmutedims() uses permutedims() ==========#

_transmutedims_doc = """
    transmutedims(A, perm′)
    transmutedims!(dst, src, perm′)

These are just like `permutedims` / `permutedims!`, except that `perm′` need not
be a permutation of `1:ndims(A)`. Any number outside this range inserts a trivial
dimension into the output, size 1, fitting with `size(A,99) == 1`.
And any repeated number places that dimension along a diagonal.

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

julia> transmutedims(reshape((1:10),2,:), (1,2,1))
2×5×2 Array{Int64,3}:
[:, :, 1] =
 1  3  5  7  9
 0  0  0  0  0

[:, :, 2] =
 0  0  0  0   0
 2  4  6  8  10
```
"""

@doc _transmutedims_doc
function transmutedims(src::AbstractArray{T,N}, perm) where {T,N}
    length(perm) >= ndims(src) || throw(ArgumentError("length(perm) is less than ndims(src)"))
    safe_perm = filter(d -> d in 1:N, Tuple(perm))#::NTuple{N, Int}
    final_ax = map(d -> d in 1:N ? axes(src, d) : Base.OneTo(1), Tuple(perm))
    if safe_perm == 1:ndims(src)
        reshape(copy(src), final_ax...)
    elseif all_unique(filter(!iszero, safe_perm)...)
        reshape(permutedims(src, safe_perm), final_ax...)
    else
        collect(transmute(src, perm))
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
    elseif all_unique(filter(!iszero, safe_perm)...)
        permutedims!(reshape(dst, permed_ax...), src, safe_perm)
    else
        copyto!(dst, transmute(src, perm))
    end
    dst
end

#========== The rest ==========#

include("gpu.jl")

include("base.jl")

end
