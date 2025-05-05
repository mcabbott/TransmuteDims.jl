module TD_OffsetArrays

using TransmuteDims, OffsetArrays

TransmuteDims.may_reshape(::Type{<:OffsetArray{T,N,AT}}) where {T,N,AT} = TransmuteDims.may_reshape(AT)

end
