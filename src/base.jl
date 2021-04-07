#========== tuples ==========#

struct TupleVector{T,DT,L} <: AbstractVector{T}
    data::DT
    TupleVector(tup::DT) where {DT <: Tuple} =
        new{mapreduce(typeof, Base.promote_typejoin, tup), DT, length(tup)}(tup)
end

Base.size(v::TupleVector{T,DT,L}) where {T,DT,L} = (L,)
Base.@propagate_inbounds Base.getindex(v::TupleVector, i::Integer) = getindex(v.data, i)

Core.Tuple(v::TupleVector) = v.data
Base.iterate(v::TupleVector) = iterate(v.data)
Base.iterate(v::TupleVector, state) = iterate(v.data, state)

transmute(tup::Tuple, perm) = transmute(TupleVector(tup), perm)
transmute(tup::Tuple, ::Val{perm}) where {perm} = transmute(TupleVector(tup), Val(perm))

transmutedims(tup::Tuple, perm=(2,1)) = collect(transmute(tup, perm))

function Base.showarg(io::IO, v::TupleVector{T,DT}, toplevel) where {T,DT}
    if all(==(T), DT.parameters)
        toplevel || print(io, "::")
        print(io, "TupleVector{", T, "}")
    else
        print(io, "TupleVector(::", DT, ")")
        toplevel && print(io, " with eltype ", T)
    end
end

#========== dropdims ==========#

# Especially when dropping a trivial dimension, we don't want to produce
# reshape(TransmutedDimsArray(::Array ...

Base.dropdims(A::TransmutedDimsArray; dims) = _dropdims(A, dims...)

_dropdims(A) = A
function _dropdims(A::TransmutedDimsArray{T,N,P}, d::Int, dims...) where {T,N,P}
    if P[d]==0
        perm = ntuple(n -> n<d ? P[n] : P[n+1], N-1)
        newdims = map(n -> n<d ? n : n-1, dims)
        _dropdims(transmute(A.parent, perm), newdims...)
    else
        perm = ntuple(N-1) do n
            Pn = n<d ? P[n] : P[n+1]
            Pn<d ? Pn : Pn-1
        end
        newdims = map(n -> n<d ? n : n-1, dims)
        _dropdims(transmute(dropdims(A.parent; dims=P[d]), perm), newdims...)
    end
end

#=

@btime        dropdims($(transmute(rand(3,3), (2,0,1))), dims=2)
@code_warntype dropdims(transmute(rand(3,3), (2,0,1)), dims=2)

@btime        dropdims($(transmute(rand(3,1,3), (3,2,1))), dims=2)
@code_warntype dropdims(transmute(rand(3,1,3), (3,2,1)), dims=2)

=#

#========== reshape ==========#

function Base.vec(A::TransmutedDimsArray{T,N,P}) where {T,N,P}
    if increasing_or_zero(P)  # the case which transmute() will avoid creating
        vec(A.parent)
    else
        reshape(A, length(A))
    end
end

#========== transpose, etc ==========#

Base.transpose(A::TransmutedDimsArray{<:Number, 1}) = transmute(A, Val((2,1)))
Base.transpose(A::TransmutedDimsArray{<:Number, 2}) = transmute(A, Val((2,1)))
Base.adjoint(A::TransmutedDimsArray{<:Real, 1}) = transmute(A, Val((2,1)))
Base.adjoint(A::TransmutedDimsArray{<:Real, 2}) = transmute(A, Val((2,1)))

Base.PermutedDimsArray(A::TransmutedDimsArray, perm) = transmute(A, perm)

#========== reductions ==========#

# Same strategy as in https://github.com/JuliaLang/julia/pull/39513

function Base.mapreducedim!(f, op, B::AbstractArray, A::TransmutedDimsArray{T,N,P,Q}) where {T,N,P,Q}
    if unique_or_zero(Val(P))
        # any dense transmutation
        Base.mapreducedim!(f, op, transmute(B, Q), parent(A))  # using Val(Q) changes nothing
    else
        # default next step
        Base._mapreducedim!(f, op, B, A)
    end
    B
end

if VERSION > v"1.6-"
    Base._mapreduce_dim(f, op, init::Base._InitialValue, A::TransmutedDimsArray, dims::Colon) =
        _mapreduce_scalar(f, op, init, A, dims)
else
    Base._mapreduce_dim(f, op, init::NamedTuple{()}, A::TransmutedDimsArray, dims::Colon) =
        _mapreduce_scalar(f, op, init, A, dims)
end

@inline function _mapreduce_scalar(f, op, init, A::TransmutedDimsArray{T,N,P}, dims::Colon) where {T,N,P}
    if dims === Colon() && f === identity && op === Base.add_sum
        # safe & easy
        Base._mapreduce_dim(f, op, init, parent(A), dims)
    elseif unique_or_zero(Val(P))
        # any dense transmutation
        Base._mapreduce_dim(f, op, init, parent(A), dims)
    elseif op === Base.add_sum && iszero(f(zero(T)))
        # like sum(::Diagonal)
        Base._mapreduce_dim(f, op, init, parent(A), dims)
    else
        # default next step
        Base._mapreduce(f, op, IndexStyle(A), A)
    end
end

#========== copyto! ==========#

function Base.copyto!(dst::AbstractArray, src::TransmutedDimsArray)
    if axes(dst) == axes(src)
        copy!(dst, src)
    elseif length(dst) == length(src)
        copy!(reshape(dst, axes(src)), src)  # could save a reshape when increasing_or_zero(P)
    elseif length(dst) < length(src)
        throw(BoundsError(dst, lastindex(src)))
    else
        throw(BoundsError(src, lastindex(dst)))
    end
    dst
end

# @propagate_inbounds
function Base.copy!(dst::AbstractArray, src::TransmutedDimsArray{T,N,P,Q}) where {T,N,P,Q}
    @boundscheck axes(dst) == axes(src) || throw(ArgumentError("arrays must have the same axes for copy! (consider using copyto!"))
    if increasing_or_zero(P)  # just a reshape
        copyto!(dst, parent(src))
    else
        if unique_or_zero(P)
            _densecopy_permuted!(dst, parent(src), Val(P))
            # this is happy to reshape... should it be limited to
        else
            fill!(dst, zero(T))  # Diagonal-like
            _copy_into!(dst, parent(src), Val(P))
        end
    end
    dst
end

# For Arrays, this dispatches to use Strided.jl version. Second best:
@generated function _densecopy_permuted!(dst::DenseArray, src::AbstractArray, val::Val{P}) where {P}
    Pminus = filter(!=(0), collect(P))
    if 0 in P
        SB = [:(axes(src,$p)) for p in Pminus]
        Bex = :(reshape(dst, ($(SB...),)))
    else
        Bex = :dst
    end
    if sort(Pminus) == 1:ndims(src)
        Aex = :src
        perm = Tuple(Pminus)
    else
        SA = [:(axes(src,$d)) for d in 1:ndims(src) if d in Pminus]
        Aex = :(reshape(src, ($(SA...),)))
        perm = Tuple(sortperm(Pminus))
    end
    :(permutedims!($Bex, $Aex, $perm); nothing)
end

# Fallback option:
_densecopy_permuted!(dst::AbstractArray, src::AbstractArray, val::Val) =
    _copy_into!(dst, src, val)

function _copy_into!(dst::AbstractArray, parent::AbstractArray, ::Val{P}) where {P}
    @inbounds @simd for I in CartesianIndices(parent)
        J = CartesianIndex(map(p -> p==0 ? 1 : I[p], P))
        dst[J] = parent[I]
    end
    nothing
end


#========== view ==========#

function Base.view(A::TransmutedDimsArray{T,N,P,Q}, inds::Vararg{Union{Int,Colon},N}) where {T,N,P,Q}
    if _is_simple(inds, P)
        parent_inds = genperm_zero(inds, Q, missing)
        view(parent(A), parent_inds...)
    else
        view(A, Base.to_indices(A, inds)...)
    end
end

# Only allow one colon, and P there is not zero
@inline function _is_simple(inds::Tuple, P::NTuple{N,Int}) where {N}
    sum(map(i -> Int(i isa Colon), inds)) == 1 || return false
    n = sum(ntuple(d -> inds[d] isa Colon ? d : 0, N))
    return P[n] != 0
end
# @btime TransmuteDims._is_simple((1,2,:,3), (1,0,2,3)) # 0.041 ns


# @inline function _is_simple(inds, P::Tuple)
#     count(i -> i isa Colon, inds) == 1 || return false
#     n = findfirst(i -> i isa Colon, inds)
#     return P[n] != 0
# end
# @btime _is_simple((1,2,:,3), (1,0,2,3)) # 58.365 ns


#========== the end. ==========#
