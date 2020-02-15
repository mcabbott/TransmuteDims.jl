#========== dropdims ==========#

# Especially when dropping a trivial dimension, we don't want to produce
# reshape(TransmutedDimsArray(::Array ...

Base.dropdims(A::TransmutedDimsArray; dims) = _dropdims(A, dims...)

_dropdims(A) = A
function _dropdims(A::TransmutedDimsArray{T,N,P}, d::Int, dims...) where {T,N,P}
    if P[d]==0
        perm = ntuple(n -> n<d ? P[n] : P[n+1], N-1)
        newdims = map(n -> n<d ? n : n-1, dims)
        _dropdims(Transmute{perm}(A.parent), newdims...)
        # _dropdims(transmute(A.parent, perm), newdims...)
    else
        perm = ntuple(N-1) do n
            Pn = n<d ? P[n] : P[n+1]
            Pn<d ? Pn : Pn-1
        end
        newdims = map(n -> n<d ? n : n-1, dims)
        _dropdims(Transmute{perm}(dropdims(A.parent; dims=P[d])), newdims...)
        # _dropdims(transmute(dropdims(A.parent; dims=P[d]), perm), newdims...)
    end
end

#=

@btime        dropdims($(Transmute{(2,0,1)}(rand(3,3))), dims=2)
@code_warntype dropdims(Transmute{(2,0,1)}(rand(3,3)), dims=2)

@btime        dropdims($(Transmute{(3,2,1)}(rand(3,1,3))), dims=2)
@code_warntype dropdims(Transmute{(3,2,1)}(rand(3,1,3)), dims=2)

=#

#========== transpose, etc ==========#

Base.transpose(A::TransmutedDimsArray{<:Number, 1}) = Transmute{(2,1)}(A)
Base.transpose(A::TransmutedDimsArray{<:Number, 2}) = Transmute{(2,1)}(A)
Base.adjoint(A::TransmutedDimsArray{<:Real, 1}) = Transmute{(2,1)}(A)
Base.adjoint(A::TransmutedDimsArray{<:Real, 2}) = Transmute{(2,1)}(A)

Base.PermutedDimsArray(A::TransmutedDimsArray, P) = Transmute{P}(A)

#=
# piracy!

for N=3:15
    @eval Base.transpose(x::AbstractArray{<:Number,$N}; dims) = _transpose(x, Tuple(dims))
    # @eval Base.transpose(x::AbstractArray{<:Number,$N}; dims) = _transpose(x, Val(Tuple(dims)))
end

@btime transpose($(rand(2,2,2,2)), dims=(2,3)) # 3.686 μs with Val, 259.155 ns without!

@btime TransmuteDims._transpose($(rand(2,2,2,2)), (2,3))      # 262.377 ns
@btime TransmuteDims._transpose($(rand(2,2,2,2)), Val((2,3))) #   5.617 ns


# transpose has no 2-arg methods:
map(m -> m.sig.parameters |> length, methods(transpose))

=#

"""
    _transpose(A, (d, d′))
    _transpose(A, Val((d, d′)))

Constructs a lazy `TransmutedDimsArray` in which dimensions `d` and `d′` are exchanged.
These may be larger than `ndims(A)`, although if both are then it's a waste.
"""
function _transpose(x::AbstractArray{T,M}, dims::Tuple{Int,Int}) where {T,M}
    a, b = dims
    N = max(a, b, M)
    perm = ntuple(N) do d
        if d==a
            b<=M && return b
        elseif d==b
            a<=M && return a
        elseif d<=M
            return d
        end
        0
    end
    iperm = invperm_zero(perm, M)
    TransmutedDimsArray{T,N,perm,iperm,typeof(x),false}(x)
end

@generated function _transpose(data::AbstractArray{T,M}, ::Val{dims}=Val((1,2))) where {T,M,dims}
    a, b = dims
    N = max(a, b, M)
    perm = ntuple(N) do d
        if d==a
            b<=M && return b
        elseif d==b
            a<=M && return a
        elseif d<=M
            return d
        end
        0
    end
    :( Transmute{$perm}(data) )
end

# function transpose_a_b(a, b, x)
#     N = max(a, b, ndims(x))
#     perm = ntuple(d -> d==a ? b : d==b ? a : d, N)
#     perm, N
# end

#========== Reductions ==========#

Base.sum(A::TransmutedDimsArray) = sum(A.parent)
