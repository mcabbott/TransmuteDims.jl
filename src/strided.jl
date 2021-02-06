#=

StridedView allows permutations, and gap dimensions, but not Diagonal-like objects.
Instead of storing a permutation in the type, it stores the strides in a field.

Making @strided transmute(A, perm) return another StridedView will make TensorCast work,
but is unlike the other _transmute methods, which aim to unwrap.

=#

using Strided
using Strided: StridedView, UnsafeStridedView # @unsafe_strided

@inline function _transmute(A::StridedView{T,M}, P) where {T,M}
    Q = invperm_zero(P, size(A))  # also checks size of dropped dimensions
    N = length(P)
    if M == N && P == ntuple(identity, N)  # trivial case
        data
    elseif unique_or_zero(P)
        sz = map(d -> d==0 ? 1 : size(A,d), P)
        st = map(d -> d==0 ? 0 : stride(A,d), P)
        StridedView(A.parent, sz, st, A.offset, A.op)
    elseif P == (1,1)
        Diagonal(A.parent)
    else
        TransmutedDimsArray{T,N,P,Q,typeof(A)}(A)
    end
end

@inline function Strided.StridedView(A::TransmutedDimsArray{T,N,P,Q}) where {T,N,P,Q}
    _transmute(StridedView(parent(A)), P)
end

#=

The eager transmutedims(A, perm) should also use this?

=#

@inline function _densecopy_permuted!(dst::Array{T}, A::StridedArray{TA}, ::Val{P}) where {T,TA,P}
    sz = map(d -> d==0 ? 1 : size(A,d), P)
    st = map(d -> d==0 ? 0 : stride(A,d), P)
    if isbitstype(T) && isbitstype(TA)
        _A = UnsafeStridedView(pointer(A), sz, st, 0, identity)
        _B = UnsafeStridedView(pointer(dst), size(dst), strides(dst), 0, identity)
        _densecopy_strided!(_B, _A)
    else
        _densecopy_strided!(dst, StridedView(A, sz, st, 0, identity))
    end
    nothing
end

@inline function _densecopy_strided!(dst, src)
    T = eltype(dst)
    LinearAlgebra.axpby!(one(T), src, zero(T), dst)
end


#         if isbitstype(eltype(A)) && isbitstype(eltype(C))
#             @unsafe_strided A C _add!(α, A, β, C, (indCinA...,))
#         else
#             _add!(α, StridedView(A), β, StridedView(C), (indCinA...,))
#         end

# _add!(α, A::AbstractStridedView{<:Any,N},
#         β, C::AbstractStridedView{<:Any,N}, indCinA::IndexTuple{N}) where N =
#     LinearAlgebra.axpby!(α, permutedims(A, indCinA), β, C)
