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

# @propagate_inbounds
function Base.copy!(dst::AbstractArray, src::TransmutedDimsArray{T,N,P,Q}) where {T,N,P,Q}
    @boundscheck axes(dst) == axes(src) || throw(ArgumentError("arrays must have the same axes for copy! (consider using copyto!"))
    if increasing_or_zero(P)  # just a reshape
        copyto!(dst, parent(src))
    else
        if unique_or_zero(P)
            _densecopy_permuted!(dst, parent(src), Val(P))
        else
            fill!(dst, zero(T))  # Diagonal-like
            _copy_into!(dst, parent(src), Val(P))
        end
    end
    dst
end

@generated function _densecopy_permuted!(dst::AbstractArray, src::AbstractArray, val::Val{P}) where {P}
    Pminus = filter(!=(0), collect(P))
    if 0 in P
        SB = [:(axes(src,$p)) for p in Pminus]
        Bex = :(reshape(dst, ($(SB...),)))
    else
        Bex = :dst
    end
    if sort(Pminus) == 1:ndims(src)
        Aex = :src
        perm = Pminus
    else
        SA = [:(axes(src,$d)) for d in 1:ndims(src) if d in Pminus]
        Aex = :(reshape(src, ($(SA...),)))
        perm = Tuple(sortperm(Pminus))
    end
    :(permutedims!($Bex, $Aex, $perm))
end

function _copy_into!(dst::AbstractArray, parent::AbstractArray, ::Val{P}) where {P}
    @inbounds @simd for I in CartesianIndices(parent)
        J = CartesianIndex(map(p -> p==0 ? 1 : I[p], P))
        dst[J] = parent[I]
    end
end

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

#========== the end. ==========#
