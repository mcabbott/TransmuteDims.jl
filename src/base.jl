#========== Other Base functions ==========#

Base.dropdims(A::TransmutedDimsArray; dims) = _dropdims(A, dims...)

_dropdims(A) = A
function _dropdims(A::TransmutedDimsArray{T,N,P}, d::Int, dims...) where {T,N,P}
    if P[d]==0
        perm = ntuple(n -> n<d ? P[n] : P[n+1], N-1)
        newdims = map(n -> n<d ? n : n-1, dims)
        _dropdims(Transmute{perm}(A.parent), newdims...)
    else
        perm = ntuple(N-1) do n
            Pn = n<d ? P[n] : P[n+1]
            Pn<d ? Pn : Pn-1
        end
        newdims = map(n -> n<d ? n : n-1, dims)
        _dropdims(Transmute{perm}(dropdims(A.parent; dims=P[d])), newdims...)
    end
end

#=

@btime        dropdims($(Transmute{(2,0,1)}(rand(3,3))), dims=2)
@code_warntype dropdims(Transmute{(2,0,1)}(rand(3,3)), dims=2)

@btime        dropdims($(Transmute{(3,2,1)}(rand(3,1,3))), dims=2)
@code_warntype dropdims(Transmute{(3,2,1)}(rand(3,1,3)), dims=2)

=#

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
function _transpose(x::AbstractArray{<:Number}, dims::Tuple{Int,Int})
    a, b = dims
    N = max(a, b, ndims(x))
    perm = ntuple(d -> d==a ? b : d==b ? a : d<ndims(x) ? d : 0, N) # no, this is only for < ndims
    TransmutedDimsArray{eltype(x),N,perm,perm,typeof(x),false}(x)

    # P, N = transpose_a_b(a, b, x)
    # TransmutedDimsArray{eltype(x),N,P,P,typeof(x),false}(x)
end

@generated function _transpose(data::AbstractArray, ::Val{dims}=Val((1,2))) where {dims}
    a, b = dims
    perm = ntuple(d -> d==a ? b : d==b ? a : d, max(a, b, ndims(data)))
    :( Transmute{$perm}(data) )
end

# function transpose_a_b(a, b, x)
#     N = max(a, b, ndims(x))
#     perm = ntuple(d -> d==a ? b : d==b ? a : d, N)
#     perm, N
# end

#========== Reductions ==========#

Base.sum(A::TransmutedDimsArray) = sum(A.parent)
