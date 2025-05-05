module TD_ChainRulesCore

using ChainRulesCore, TransmuteDims, LinearAlgebra
using ChainRulesCore: ZeroTangent, NoTangent
using TransmuteDims: TransmutedDimsArray, unique_or_zero, _undiagonal, getinvperm

function ChainRulesCore.rrule(::Type{<:TransmutedDimsArray{T,N,P,Q,AT}}, A::AbstractArray) where {T,N,P,Q,AT}
    B = TransmutedDimsArray{T,N,P,Q,AT}(A)
    function transmute_back(Δ::AbstractArray)
        if unique_or_zero(Val(P))
            (ZeroTangent(), transmute(Δ, Q), ZeroTangent())
        else
            (ZeroTangent(), _undiagonal(Δ, P, Q), ZeroTangent())
        end
    end
    function transmute_back(Δ::Tangent)
        (ZeroTangent(), Δ.parent, ZeroTangent())
    end
    B, transmute_back
end

end  # module
