#=

StridedView allows permutations, and gap dimensions, but not Diagonal-like objects.
Instead of storing a permutation in the type, it stores the strides in a field.

Making @strided transmute(A, perm) return another StridedView will make TensorCast work,
but is unlike the other _transmute methods, which aim to unwrap.

=#

using Strided

@inline function _transmute(A::Strided.StridedView{T,M}, P) where {T,M}
    Q = invperm_zero(P, size(A))  # also checks size of dropped dimensions
    N = length(P)
    if M == N && P == ntuple(identity, N)  # trivial case
        data
    elseif unique_or_zero(P)
        sz = map(d -> d==0 ? 1 : size(A,d), P)
        st = map(d -> d==0 ? 0 : stride(A,d), P)
        Strided.StridedView(A.parent, sz, st, A.offset, A.op)
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

# function _densecopy_permuted!(dst::Array{T}, A::StridedArray, ::Val{P}) where {T,P}
#     @info "strided densecopy"
#     sz = map(d -> d==0 ? 1 : size(A,d), P)
#     st = map(d -> d==0 ? 0 : stride(A,d), P)
#     _A = StridedView(A, sz, st) # , 0, identity)
#     LinearAlgebra.axpby!(one(T), _A, zero(T), dst)
# end

#         if isbitstype(eltype(A)) && isbitstype(eltype(C))
#             @unsafe_strided A C _add!(α, A, β, C, (indCinA...,))
#         else
#             _add!(α, StridedView(A), β, StridedView(C), (indCinA...,))
#         end

# _add!(α, A::AbstractStridedView{<:Any,N},
#         β, C::AbstractStridedView{<:Any,N}, indCinA::IndexTuple{N}) where N =
#     LinearAlgebra.axpby!(α, permutedims(A, indCinA), β, C)
