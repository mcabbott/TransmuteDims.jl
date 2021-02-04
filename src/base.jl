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

#========== Reductions ==========#

# Different strategy to that in https://github.com/JuliaLang/julia/pull/39513
# this may return another transmuted array. But it infers badly.

function Base.mapreduce(f, op, A::TransmutedDimsArray{T,N,P,Q,AT}; dims=(:), init=Base._InitialValue()) where {T,N,P,Q,AT}
    if dims === Colon() && f === identity && op === Base.add_sum
        # any complete sum, including equivalent of sum(::Diagonal)
        mapreduce(f, op, parent(A); dims=(:), init=init)
    elseif unique_or_zero(P)
        # any dense transmutation
        if dims === Colon()
            mapreduce(f, op, parent(A); dims=(:), init=init)
        # elseif ! any(iszero, P)
        #     new_dims = map(d -> P[d], Tuple(dims))
        #     B = mapreduce(f, op, parent(A); dims=new_dims, init=init)
        #     new_P = map(p -> p in new_dims ? 0 : p, P)
        #     transmute(B, new_P)
        else
            new_dims = filter(!=(0), map(d -> P[d], Tuple(dims)))
            B = mapreduce(f, op, parent(A); dims=new_dims, init=init)
            # this new_P step is optional, but sometimes allows simplifications.
            new_P = map(p -> p in new_dims ? 0 : p, P)
            # new_P = map(p -> Q[p] in dims ? 0 : p, P)
            # @show P Q dims new_dims new_P
            transmute(B, new_P)
            # transmute(B, P)
        end
    else
        # cases like sum(::Diagonal; dims), or prod(::Diagonal), use default path
        Base._mapreduce_dim(f, op, init, A, dims)

        # # Don't sum over any duplicated dimension? Not quite right, e.g. dims=(1,2)
        # tmp = map(dims) do d
        #     p = P[d]
        #     count(isequal(p), P) > 1 ? 0 : d
        # end
        # new_dims = filter(!=(0), Tuple(tmp))

        # # Collapse to trivial summed repeated? Not quite right, transmute(1:10, (1,1,1))
        # new_P = ntuple(d -> P[d] in P[d+1:end] ? 0 : P[d], length(P))
        # B = mapreduce(f, op, parent(A); dims=new_dims, init=init)

        # And still that would only be for sum, not prod
        # transmute(B, new_P)
    end
end

