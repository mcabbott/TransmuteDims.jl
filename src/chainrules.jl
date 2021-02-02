using ChainRulesCore, LinearAlgebra

function ChainRulesCore.rrule(::Type{<:TransmutedDimsArray{T,N,P,Q,AT}}, A::AbstractArray) where {T,N,P,Q,AT}
    B = TransmutedDimsArray{T,N,P,Q,AT}(A)
    ∇TransmutedDimsArray = if unique_or_zero(P)
        Δ -> (NO_FIELDS, transmute(Δ, Q), Zero())
    else
        Δ ->  (NO_FIELDS, _undiagonal(Δ, P, Q), Zero())
    end
    B, ∇TransmutedDimsArray
end

function _undiagonal(Δ::AbstractArray, P::NTuple{N,Int}, Q::NTuple{M,Int}) where {M,N}
    if P == (1,1) && Q == (2,) && ndims(Δ) == 2
        return Δ[LinearAlgebra.diagind(Δ)]
    end
    axe = map(Q) do q
        q == 0 ? Base.OneTo(1) : axes(Δ,q)
    end
    for d in 1:N
        if P[d] == 0
            size(Δ,d) == 1 || throw(DimensionMismatch(
                "expected size(Δ,$d) == 1, got size(Δ) = $(size(Δ)). P = $P, Q = $Q"))
        else
            axes(Δ,d) == axe[P[d]] || throw(DimensionMismatch(
                "expected axes(Δ,$d) == $(axe[P[d]]), got $(axes(Δ,d)). P = $P, Q = $Q"))
        end
    end
    out = fill!(similar(Δ, axe), 0)
    @inbounds for I in CartesianIndices(axe)
        J = CartesianIndex(map(p -> I[p], P))
        out[I] = Δ[J]
    end
    out
end

unique_or_zero(P::Tuple{}) = true
function unique_or_zero(P::Tuple)
    first(P) == 0 && return true
    first(P) in Base.tail(P) && return false
    return unique_or_zero(Base.tail(P))
end

