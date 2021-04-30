using TransmuteDims
using Test, LinearAlgebra, Random

@testset "utils" begin

    using TransmuteDims: sanitise_zero, genperm_zero, invperm_zero, is_off_diag

    @test sanitise_zero((1,2,3), Val(2)) === (1,2,0)
    @test sanitise_zero([1,2,nothing,4], Val(3)) === (1,2,0,0)

    @test genperm_zero((:a,:b,:c,:d), (3,2,4,1)) == Base.PermutedDimsArrays.genperm((:a,:b,:c,:d), (3,2,4,1))
    @test genperm_zero((:a,:b,:c,:d), (3,0,4,1)) == (:c, 1, :d, :a)
    @test genperm_zero((:a,:b,:c,:d), (3,0,4,1), :z) == (:c, :z, :d, :a)

    @test invperm_zero((5,1,3,4,2),5,:A)[1] == invperm((5,1,3,4,2))
    @test invperm_zero((1,2,3),4,:A)[1] == (1,2,3,0)
    @test invperm_zero((1,3),4,:A)[2][2] isa Expr  # two assertions, size(A,2)==1 && size(A,4)==1

    using TransmuteDims: increasing_or_zero, unique_or_zero, may_reshape

    @test increasing_or_zero((1,2,0,3)) == true
    @test increasing_or_zero((1,2,0,2)) == false

    @test unique_or_zero((1,3,0,2)) == true
    @test unique_or_zero((0,1,3)) == true
    @test unique_or_zero((0,1,0,3,0)) == true
    @test unique_or_zero((0,1,1)) == false
    @test unique_or_zero((0,1,2,0,2)) == false

    @test may_reshape(typeof(rand(3))) == true
    @test may_reshape(typeof(rand(3)')) == false
    @test may_reshape(typeof(reshape(1:4,2,2))) == true

end
@testset "eager" begin

    m = rand(1:99, 3,4)
    @test transmutedims(m, (1,0,2)) == reshape(m, 3,1,4)
    @test transmutedims(m, (2,99,1)) == reshape(transpose(m), 4,1,3)
    @test transmutedims(m, (2,nothing,1)) == reshape(permutedims(m), 4,1,3)

    @test transmutedims(m, (3,2,1)) == reshape(m', 1,4,3)
    @test transmutedims(transmutedims(m, (3,2,1)), (3,2)) == m

    o314 = rand(Int, 3,1,4);
    @test transmutedims!(o314, m, (1,0,2)) == reshape(m, 3,1,4)
    o413 = rand(Int, 4,1,3);
    @test transmutedims!(o413, m, (2,99,1)) == reshape(permutedims(m), 4,1,3)
    @test o413 == reshape(permutedims(m), 4,1,3)

    # default perm
    @test transmutedims(m) == m'
    @test transmutedims(m[:,1]) == m[:,1]'

    # errors
    @test_throws ArgumentError transmutedims(m, (1,))    # too few
    @test_throws ArgumentError transmutedims(m, (1,0,0)) # 2 doesn't appear

    @test_throws ArgumentError transmutedims!(o314, m, (1,))
    @test_throws Exception transmutedims!(o314, m, (1,0,0))
    @test_throws Exception transmutedims!(o314, m, (1,2,2))

    @test_throws Exception transmutedims!(o314, m, (2,0,1)) # wrong size

end
@testset "lazy" begin

    m = rand(1:99, 3,4)
    @test transmute(m, (1,0,2)) == reshape(m, 3,1,4)
    @test transmute(m, (2,99,1)) == reshape(transpose(m), 4,1,3)
    @test transmute(m, (2,nothing,1)) == reshape(transpose(m), 4,1,3)
    @test transmute(m, Val((1,0,2))) == reshape(m, 3,1,4)
    @test transmute(m, Val((2,99,1))) == reshape(transpose(m), 4,1,3)
    @test transmute(m, Val((2,nothing,1))) == reshape(transpose(m), 4,1,3)
    @test TransmutedDimsArray(m, (1,0,2)) == reshape(m, 3,1,4)
    @test TransmutedDimsArray(m, (2,99,1)) == reshape(transpose(m), 4,1,3)
    @test TransmutedDimsArray(m, (2,nothing,1)) == reshape(transpose(m), 4,1,3)

    v = m[:,1]
    @test transmute(v, (1,1)) isa Diagonal
    @test transmute(v, Val((1,1))) isa Diagonal

    g = TransmutedDimsArray(m, (1,0,2)) # could be an Array
    t = transmute(m, (2,0,1))

    # setindex!
    g[3,1,4] = 222
    @test m[3,4] == 222
    t[2,1,3] = 333
    @test m[3,2] == 333

    # linear indexing
    @test Base.IndexStyle(t) == IndexCartesian()
    g[4] = 444
    @test m[4] == 444
    t[5] = 555
    @test transpose(m)[5] == 555

    # linear indexing, for more constructors
    y = zeros(1,2,3);
    @test IndexStyle(transmute(y, (1,0,2,0,3))) == IndexLinear() # it's an Array
    @test IndexStyle(transmute(y, (3,0,2,0,1))) == IndexCartesian()
    @test IndexStyle(TransmutedDimsArray(y, (3,0,2,0,1))) == IndexCartesian()

    @test transmute(ones(3) .+ im, (1,))[1] == 1 + im # was an ambiguity error

    # reshape
    @test vec(TransmutedDimsArray(m, (1,0,2))) isa Vector
    @test vec(TransmutedDimsArray(m, (2,0,1))) isa Base.ReshapedArray
    @test transmute(reshape(1:6, 2,3), (1,0,2)) isa Base.ReshapedArray

    # errors
    @test_throws ArgumentError transmute(m, (2,))
    @test_throws ArgumentError transmute(m, (2,0,3,0,2))
    @test_throws ArgumentError transmute(m, Val((2,)))
    @test_throws ArgumentError transmute(m, Val((2,0,3,0,2)))
    @test_throws Exception TransmutedDimsArray(m, (2,))
    @test_throws Exception TransmutedDimsArray(m, (2,0,3,0,2))

end
@testset "unwrapping" begin

    m = rand(1:99, 3,4)
    v = m[:,1]

    # unwrapping: LinearAlgebra
    @test transmute(m', (2,0,1)) == transmute(m, (1,0,2)) # both are reshapes
    @test transmute(m', (2,0,1)) isa Array
    @test transmute(m', Val((2,0,1))) isa Array
    @test transmute(m, (1,0,2)) isa Array
    @test transmute(m, Val((1,0,2))) isa Array
    @test transmute(v |> transpose, (2,3,1)) == transmute(v, (1,0,0))
    @test transmute(v |> transpose, (2,3,1)) isa Array
    @test transmute(v |> transpose, Val((2,3,1))) isa Array

    @test transmute((1:3)',(2,2)) === Diagonal(1:3)
    @test_throws ArgumentError transmute((1:3)',(1,1)) # 1 is trivial dim. Confusing message though!
    @test transmute((1:3)',(1,2)) == (1:3)' # but not ===, unwrapped before noticing.

    @test transmute(Diagonal(1:10), (3,1)) === TransmutedDimsArray(1:10, (0,1))
    @test transmute(Diagonal(rand(10)), (3,1)) isa Matrix
    @test transmute(Diagonal(rand(10)), Val((3,1))) isa Matrix
    @test transmute(TransmutedDimsArray(1:10, (1,1)), (3,1)) === TransmutedDimsArray(1:10, (0,1))
    @test transmute(TransmutedDimsArray(1:10, (1,1)), Val((3,1))) === TransmutedDimsArray(1:10, (0,1))

    # unwrapping: permutations
    @test transmute(PermutedDimsArray(m,(2,1)), (2,nothing,1)) == transmute(m, (1,0,2))
    @test transmute(PermutedDimsArray(m,(2,1)), (2,nothing,1)) isa Array
    @test transmute(PermutedDimsArray(m,(2,1)), Val((2,nothing,1))) isa Array
    @test transmute(TransmutedDimsArray(m,(2,1)), (2,nothing,1)) isa Array
    @test transmute(TransmutedDimsArray(m,(2,1)), Val((2,nothing,1))) isa Array

    @test transmute(PermutedDimsArray(m,(2,1)), (1,3,2)) === transmute(m, (2,3,1))

    x = rand(1:99, 5,4,3,2);
    x1 = permutedims(x, (4,1,2,3))
    x2 = PermutedDimsArray(x, (4,1,2,3))
    @test x1 == x2
    @test TransmutedDimsArray(x1, (2,4,1,3)) == TransmutedDimsArray(x2, (2,4,1,3))
    @test TransmutedDimsArray(x1, (2,0,1,4,3)) == TransmutedDimsArray(x2, (2,0,1,4,3))

    x3 = TransmutedDimsArray(x, (4,1,2,3));
    @test TransmutedDimsArray(x1, (2,4,1,3)) == TransmutedDimsArray(x3, (2,4,1,3))
    @test TransmutedDimsArray(x1, (2,0,1,4,3)) == TransmutedDimsArray(x3, (2,0,1,4,3))

    # unwrapping caused by transpose etc
    @test transpose(TransmutedDimsArray(m, (2,1))) isa Matrix
    @test adjoint(TransmutedDimsArray(v, (0,1))) isa Matrix

end
@testset "view" begin

    m = rand(1:99, 3,4)
    g = TransmutedDimsArray(m, (1,0,2)) # could be an Array
    t = transmute(m, (2,0,1))

    @test view(g, :,1,2) isa SubArray{Int64, 1, <:Matrix}
    @test view(g, :,1,2) == g[:,1,2]

    @test view(t, :,1,2) isa SubArray{Int64, 1, <:Matrix}
    @test view(t, :,1,2) == t[:,1,2]
    @test view(t, 2,1,:) == t[2,1,:]

    @test first(eachcol(TransmutedDimsArray(m,(2,1)))) isa SubArray{Int64, 1, <:Matrix}

    # not unwrapped
    @test view(t, :,1,:) == m'
    @test view(t, 2,:,3) == t[2,:,3]

end
@testset "reductions" begin

    m = rand(1:99, 3,4)

    # reductions
    @test sum(transmute(m,(2,1,3))) == sum(m)
    @test sum(transmute(m,(1,1,2))) == sum(m)
    @test prod(transmute(m,(1,1,2))) == 0
    @test sum(x->x+1, transmute(m,(1,1,2))) == sum(m) + 3*3*4

    @test sum(TransmutedDimsArray(m,(1,2)); dims=1) == sum(m, dims=1)
    @test sum(TransmutedDimsArray(m,(2,1)); dims=1) == sum(m', dims=1)
    @test sum(TransmutedDimsArray(m,(2,1)), dims=2) == sum(m', dims=2)
    @test sum(x->x+1, TransmutedDimsArray(m,(2,1)), dims=2) == sum(x->x+1, m', dims=2)

    @test sum(transmute(m,(1,1,2)), dims=1) == sum(collect(transmute(m,(1,1,2))), dims=1)
    @test sum(transmute(m,(1,1,1,2)), dims=1) == sum(collect(transmute(m,(1,1,1,2))), dims=1)
    @test sum(transmute(m,(1,1,2)), dims=(1,2)) == sum(collect(transmute(m,(1,1,2))), dims=(1,2))
    @test prod(transmute(m,(1,1,2)), dims=(1,2)) == prod(collect(transmute(m,(1,1,2))), dims=(1,2))

    g = TransmutedDimsArray(m, (1,0,2)) # could be an Array
    t = transmute(m, (2,0,1))

    @test sum(g, dims=2) == g
    @test sum(t, dims=2) == t
    @test sum(x->x+1, t, dims=2) == sum(x->x+1, collect(t), dims=2)

    x = rand(1:99, 5,4,3,2);
    x1 = permutedims(x, (4,1,2,3))
    x3 = TransmutedDimsArray(x, (4,1,2,3))
    @test sum(x1, dims=3) == sum(x3, dims=3)
    @test sum(x1, dims=(2,4)) == sum(x3, dims=(2,4))
    @test sum(x->x+1, x1, dims=(2,4)) == sum(x->x+1, x3, dims=(2,4))

end
@testset "dropdims" begin

    m = rand(1:99, 3,4)
    g = TransmutedDimsArray(m, (1,0,2)) # could be an Array
    r = collect(g)
    t = transmute(m, (2,0,1))

    # dropdims
    @test dropdims(g, dims=2) == m
    @test dropdims(g, dims=2) isa Matrix # now unwraps!
    @test dropdims(t, dims=2) == m'
    @test dropdims(t, dims=2) isa Transpose

    h = reshape(1:9, 3,1,3)
    f = transmute(h, (0,3,2,1))

    @test size(f) == (1,3,1,3)
    @test dropdims(f, dims=1) == permutedims(h, (3,2,1))
    @test dropdims(f, dims=3).parent == reshape(h,3,3)
    @test dropdims(f, dims=(1,3)) == reshape(h,3,3)'
    @test dropdims(f, dims=(3,1)) == reshape(h,3,3)'

    # without dropdims
    @test transmute(g, (1,3)) === m # unwraps
    @test transmute(r, (1,3)) == m # reshapes
    @test transmute(r, (1,3)) isa Matrix
    @test transmute(t, (3,1)) === m # unwraps
    @test transmute(t, (1,3)) === transpose(m)

    @test transmute(g, Val((1,3))) === m
    @test transmute(r, Val((1,3))) == m
    @test transmute(r, Val((1,3))) isa Matrix
    @test transmute(t, Val((3,1))) === m
    @test transmute(t, Val((1,3))) === transpose(m)

    @test transmutedims(g, (1,3)) === m
    @test transmutedims(r, (1,3)) == m
    @test transmutedims(r, (1,3)) isa Matrix
    @test transmutedims(t, (3,1)) === m
    @test transmutedims(t, (1,3)) == transpose(m)

    # errors
    @test_throws ArgumentError transmute(g, (1,2))
    @test_throws ArgumentError transmute(g, Val((1,2)))
    @test_throws ArgumentError transmutedims(g, (1,2))

end
@testset "diagonal" begin

    for d in [
        transmute(ones(Int,3), (1,1)),
        transmute(ones(Int,3), Val((1,1))),
        TransmutedDimsArray(ones(Int,3), (1,1)),
        ]
        @test d == [1 0 0; 0 1 0; 0 0 1]
        @test d[2] == 0
        @test (d[3,3] = 33) == 33
        @test d[3,3] == 33
        @test (d[2,1] = 0) == 0
        @test_throws ArgumentError d[1,2] = 99
        @test IndexStyle(d) == IndexCartesian()

        @test sum(d .+ 10) == 90 + 2 + 33
    end

    r = rand(3)
    q = TransmutedDimsArray(r, (1,0,1,1))
    @test q[2,1,2,2] == r[2]
    @test q[2,1,2,3] == 0
    @test_throws ArgumentError q[1,1,1,3] = 99

    # eager
    @test q == transmutedims(r, (1,0,1,1))

    # dropdims
    @test transmute(rand(3,1), (1,1)) isa Diagonal
    @test transmute(rand(1,3), (2,2)) isa Diagonal
    @test_throws Exception transmute(rand(3,2), (1,1))
    @test transmute(rand(3,1), Val((1,1))) isa TransmutedDimsArray{Float64, 2, (1,1), (2,0)}

end
@testset "zero dims" begin

    @test transmute(fill(3), (1,2)) isa Matrix
    @test transmute(fill(3), ()) isa Array{Int,0}
    @test transmute([3], ()) isa Array{Int,0}

    @test transmute(fill(3), Val((1,2))) isa Matrix
    @test transmute(fill(3), Val(())) isa Array{Int,0}
    @test transmute([3], Val(())) isa Array{Int,0}

    @test TransmutedDimsArray(fill(3), (1,2)) isa AbstractMatrix
    @test TransmutedDimsArray(fill(3), ()) isa AbstractArray{Int,0}
    @test TransmutedDimsArray([3], ()) isa AbstractArray{Int,0}

    @test transmutedims(fill(3), (1,2)) isa Matrix
    @test transmutedims(fill(3), ()) isa Array{Int,0}
    @test transmutedims([3], ()) isa Array{Int,0}

end
@testset "tuples" begin

    @test transmute((1,2,3), (2,1)) == [1 2 3]
    @test transmute((1,2,3), (1,1)) == Diagonal(1:3)

    @test transmute((1,2,3), (1,)) isa AbstractVector  # always wraps
    @test transmute((1,2,3), (1,nothing)) isa AbstractMatrix
    @test transmute(transmute((1,2,3), (2,1)), (2,)) isa TupleVector  # unwraps once

    # eager
    @test transmutedims((1,2,3), (2,1)) == [1 2 3]
    @test transmutedims((1,2,3)) == [1 2 3]
    @test transmutedims((1,2,3), (1,1)) isa Matrix

    # promotion
    @test eltype(transmute((1,2,3.0),(2,1)) .+ [1 2 3]) == eltype((1,2,3.0) .+ [1,2,3]) # Real
    @test eltype(transmute((1,2,3+im),(2,1)) .+ [1 2 3]) == eltype((1,2,3+im) .+ [1,2,3]) # Number
    @test transmutedims((1,2,3.0)) isa Matrix{Real}

end
@testset "from Base" begin

    # keeps the num of dim
    p = randperm(5)
    q = randperm(5)
    a = rand(p...)
    b = transmutedims(a,q)
    @test isequal(size(b), tuple(p[q]...))

    # hand made case
    y = zeros(1,2,3)
    for i = 1:6
        y[i]=i
    end

    z = zeros(3,1,2)
    for i = 1:3
        z[i] = i*2-1
        z[i+3] = i*2
    end

    # permutes correctly
    @test isequal(z,transmutedims(y,[3,1,2]))
    @test isequal(z,transmutedims(y,(3,1,2)))

    # of a subarray
    a = rand(5,5)
    s = view(a,2:3,2:3)
    p = transmutedims(s, [2,1])
    @test p[1,1]==a[2,2] && p[1,2]==a[3,2]
    @test p[2,1]==a[2,3] && p[2,2]==a[3,3]

    # of a non-strided subarray
    a = reshape(1:60, 3, 4, 5)
    s = view(a,:,[1,2,4],[1,5])
    c = convert(Array, s)
    for p in ([1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1])
        @test transmutedims(s, p) == transmutedims(c, p)
        @test TransmutedDimsArray(s, p) == transmutedims(c, p)
    end
    @test_throws ArgumentError transmutedims(a, (1,1,1))
    @test_throws ArgumentError transmutedims(s, (1,1,1))
    @test_throws ArgumentError TransmutedDimsArray(a, (1,1,1))
    @test_throws ArgumentError TransmutedDimsArray(s, (1,1,1))
    cp = TransmutedDimsArray(c, (3,2,1))
    @test pointer(cp) == pointer(c)
    @test_throws ArgumentError pointer(cp, 2)
    @test strides(cp) == (9,3,1)
    ap = TransmutedDimsArray(Array(a), (2,1,3))
    @test strides(ap) == (3,1,12)

    for A in [rand(1,2,3,4),rand(2,2,2,2),rand(5,6,5,6),rand(1,1,1,1)]
        perm = randperm(4)
        @test isequal(A,transmutedims(transmutedims(A,perm),invperm(perm)))
        @test isequal(A,transmutedims(transmutedims(A,invperm(perm)),perm))
    end

    # m = [1 2; 3 4]
    # @test transmutedims(m) == [1 3; 2 4]

    # v = [1,2,3]
    # @test transmutedims(v) == [1 2 3]
end

using TransmuteDims: _float_transmutedims  # exists to test permutedims! path
@testset "sans Strided" begin

    m = rand(1:99, 3,4)
    @test _float_transmutedims(m, (1,0,2)) == reshape(m, 3,1,4)
    @test _float_transmutedims(m, (2,99,1)) == reshape(transpose(m), 4,1,3)
    @test _float_transmutedims(m, (2,nothing,1)) == reshape(permutedims(m), 4,1,3)

    g = transmutedims(m, (1,0,2)) # an Array

    @test _float_transmutedims(g, (3,1)) == m'
    @test _float_transmutedims(g, (3,0,1)) == transmute(m, (2,0,1))

end

using BenchmarkTools
@testset "allocations" begin

    y = ones(1,2,3);

    # wrap
    @test 97 > @ballocated transmute($y, (3,2,0,1)) # zero on 1.7, maybe 1.5?
    @test 17 > @ballocated transmute($y, Val((3,2,0,1)))  # nonzero only on 1.3
    # @code_warntype (y -> transmute(y, (3,2,0,1)))(y)

    # reshape
    @test 129 > @ballocated transmute($y, (1,2,0,3))
    @test 129 > @ballocated transmute($y, Val((1,2,0,3)))
    # @code_warntype (y -> transmute(y, (1,2,0,3)))(y)

end

using OffsetArrays, Random
@testset "offset" begin

    o34 = ones(11:13, 21:24)
    rand!(o34)

    @test transmute(o34, (2,1)) == permutedims(o34)
    @test axes(transmute(o34, (2,0,1))) == (21:24, 1:1, 11:13)

    @test transmute(o34, (2,0,1))[21, 1, 13] == o34[13, 21]

    # dropdims
    @test axes(transmute(sum(o34, dims = 1), (2,))) == (21:24,)
    @test transmute(sum(o34, dims = 1), (2,)) isa OffsetArray # issue 23
    o34 .= 1
    @test all(==(3), transmute(sum(o34, dims = 1), (2,))) # issue 24

end

using Strided
@testset "Strided" begin

    m = rand(1:99, 3,4)
    @test @strided(transmute(m, (0,2,1)) .+ transmute(m, (1,2,0))) isa StridedView

end

using StaticArrays
@testset "StaticArrays" begin

    m = SMatrix{3,4}(rand(1:99, 3,4))
    v = SVector{4}(rand(4))

    permutedims(v) isa Base.ReshapedArray
    @test transmute(v, (2,1)) isa TransmutedDimsArray  # no worse than Base

    permutedims(m) isa MMatrix
    @test transmute(m, (2,1)) isa SMatrix
    @test transmutedims(m, (2,1)) isa MMatrix  # this is permutedims?

    transmute(m, (2,3,1)) isa TransmutedDimsArray
end

if VERSION < v"1.5"
    @warn "skipping tests of GPUArrays"
else
    using GPUArrays, Adapt
    GPUArrays.allowscalar(false)

    jl_file = normpath(joinpath(pathof(GPUArrays), "..", "..", "test", "jlarray.jl"))
    include(jl_file)
    using .JLArrays # a fake GPU array, for testing

    @testset "GPUArrays" begin

        m = rand(4,4)
        jm = JLArray(m)
        @test_throws Exception jm[1]  # scalar getindex is disallowed

        tjm = TransmutedDimsArray(jm, (2,1))  # simple wrapper
        j2 = jm .* log.(tjm) ./ 2
        @test j2 isa JLArray
        @test collect(j2) ≈ m .* log.(m') ./ 2

        jmd = transmute(jm, (1,1,2))  # Diagonal-like
        j3 = jmd .+ 2 .* jmd .+ 1
        @test j3 isa JLArray
        @test collect(j3) ≈ 3 .* transmute(m, (1,1,2)) .+ 1

        # printing, without scalar indexing
        @test sprint(show, jm) isa String
        io = IOBuffer()
        Base.print_array(io, jm)
        @test contains(String(take!(io)), string(m[1]))

        @test jm isa DenseArray  # but reshaping it does not preserve this
        TransmuteDims.may_reshape(::Type{<:JLArray}) = false

        # eager
        @test transmutedims(jm, (0,1,2)) isa JLArray
        @test transmutedims(jm, (2,1)) isa JLArray
        @test_broken transmutedims(jm, (2,0,1)) isa JLArray # permutedims!(dest::Base.ReshapedArray{Float64, 2, JLArray{Float64, 3}, Tuple{}}, src::JLArray{Float64, 2}, perm::Tuple{Int64, Int64})

    end
end

using Tracker
@testset "Tracker: $func" for func in [transmute, transmutedims]
    # subset of the Zygote tests

    @test size(Tracker.gradient(x -> sum(sin, func(x, (2,1))), rand(2,3))[1]) == (2,3)
    @test size(Tracker.gradient(x -> sum(sin, func(x, (2,3,1))), rand(2,3))[1]) == (2,3)
    @test size(Tracker.gradient(x -> sum(sin, func(x, (2,1,1))), rand(2,3))[1]) == (2,3)

    v, m, t = rand(1:99, 3), rand(1:99, 3,3), rand(1:99, 3,3,3)

    fwd, back = Tracker.forward(x -> func(x, (2,1)), v)
    @test fwd == v'
    @test back(ones(1,3))[1] == ones(3)
    @test back(v')[1] == v  # awkward reshape(adjoint)

end

print("... loading Zygote, 🐌 ... ")
using Zygote
println("done")
@testset "Zygote: $func" for func in [transmute, transmutedims]
    NEW = VERSION >= v"1.6-"

    # sizes, and no errors!
    @test size(Zygote.gradient(x -> sum(sin, func(x, (2,1))), rand(2,3))[1]) == (2,3)
    @test size(Zygote.gradient(x -> sum(sin, func(x, (2,3,1))), rand(2,3))[1]) == (2,3)
    @test size(Zygote.gradient(x -> sum(sin, func(x, (2,1,1))), rand(2,3))[1]) == (2,3)
    NEW && func==transmute && @test size(Zygote.gradient(x -> sum(sin, func(x, (2,2,3,1,1))), rand(2,3))[1]) == (2,3)

    @test size(Zygote.gradient(x -> sum(sin, func(x, (1,))), rand(3,1))[1]) == (3,1)
    @test size(Zygote.gradient(x -> sum(sin, func(x, (2,1))), rand(3,1))[1]) == (3,1)
    NEW || @test size(Zygote.gradient(x -> sum(sin, func(x, (1,1))), rand(3,1))[1]) == (3,1)
    NEW && func==transmute && @test size(Zygote.gradient(x -> sum(sin, func(x, (1,3,1,3))), rand(3,1))[1]) == (3,1)

    NEW || @test size(Zygote.gradient(x -> sum(sin, func(x, (2,3,1))), rand(2,3,4))[1]) == (2,3,4)
    NEW || @test size(Zygote.gradient(x -> sum(sin, func(x, (2,4,3,1))), rand(2,3,4))[1]) == (2,3,4)

    # values
    v, m, t = rand(1:99, 3), rand(1:99, 3,3), rand(1:99, 3,3,3)

    fwd, back = Zygote.pullback(x -> func(x, (2,1)), v)
    @test fwd == v'
    @test back(ones(1,3))[1] == ones(3)
    @test back(v')[1] == v  # awkward reshape(adjoint)

    # extracting diagonals
    fwd, back = pullback(x -> func(x, (1,1)), v)
    @test fwd == Diagonal(v)
    @test back(m)[1] == diag(m)
    @test_skip back(m[:,:,:,:])[1] == diag(m)  # trivial extra dimensions, different path

    fwd, back = pullback(x -> func(x, (1,1,2)), m)
    @test fwd[:,:,1] == Diagonal(m[:,1])
    @test back(t)[1] == [t[i,i,k] for i in 1:3, k in 1:3]

end

if VERSION < v"1.6-"
    @warn "skipping doctests, on Julia $VERSION"
else
    println("... starting Documenter 🗣")
    using Documenter
    @testset "doctests" begin

        doctest(TransmuteDims, manual=false)

    end
end
