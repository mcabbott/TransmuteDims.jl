
using ChainRulesCore, LinearAlgebra

function ChainRulesCore.rrule(::Type{<:TransmutedDimsArray{T,N,P,Q,AT}}, A::AbstractArray) where {T,N,P,Q,AT}
    B = TransmutedDimsArray{T,N,P,Q,AT}(A)
    function transmute_back(Δ::AbstractArray)
        if unique_or_zero(Val(P))
            (NO_FIELDS, transmute(Δ, Q), ZeroTangent())
        else
            (NO_FIELDS, _undiagonal(Δ, P, Q), ZeroTangent())
        end
    end
    function transmute_back(Δ::Tangent)
        (NO_FIELDS, Δ.parent, ZeroTangent())
    end
    B, transmute_back
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
