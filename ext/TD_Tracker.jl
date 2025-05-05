module TD_Tracker

using TransmuteDims, Tracker

using Tracker: @grad, TrackedArray, track, data
using TransmuteDims: TransmutedDimsArray, unique_or_zero, _undiagonal, getinvperm

TransmutedDimsArray{T,N,P,Q,AT}(A::TrackedArray) where {T,N,P,Q,AT<:AbstractArray} = track(TransmutedDimsArray, A, P)

@grad function TransmutedDimsArray(A::AbstractArray, P)
    B = TransmutedDimsArray(data(A), P)
    Q = getinvperm(B)
    function transmute_back(Δ::AbstractArray)
        if unique_or_zero(P)
            (transmute(data(Δ), Q), nothing)
        else
            (_undiagonal(data(Δ), P, Q), nothing)
        end
    end
    B, transmute_back
end

end  # module
