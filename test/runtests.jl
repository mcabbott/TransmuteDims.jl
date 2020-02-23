using TransmuteDims, Random, Test, OffsetArrays
using TransmuteDims: TransmutedDimsArray, _transpose

@testset "eager" begin

    m = rand(1:99, 3,4)
    @test transmutedims(m, (1,0,2)) == reshape(m, 3,1,4)
    @test transmutedims(m, (2,99,1)) == reshape(transpose(m), 4,1,3)
    @test transmutedims(m, (2,nothing,1)) == reshape(permutedims(m), 4,1,3)

    o314 = rand(3,1,4);
    @test transmutedims!(o314, m, (1,0,2)) == reshape(m, 3,1,4)
    o413 = rand(4,1,3);
    @test transmutedims!(o413, m, (2,99,1)) == reshape(permutedims(m), 4,1,3)
    @test o413 == reshape(permutedims(m), 4,1,3)

    @test_throws ArgumentError transmutedims(m, (1,))    # too few
    @test_throws ArgumentError transmutedims(m, (1,0,0)) # 2 doesn't appear
    # @test_throws Exception transmutedims(m, (1,2,1)) # 1 is repeated

    @test_throws ArgumentError transmutedims!(o314, m, (1,))
    @test_throws Exception transmutedims!(o314, m, (1,0,0))
    @test_throws DimensionMismatch transmutedims!(o314, m, (1,2,2))

    @test_throws Exception transmutedims!(o314, m, (2,0,1)) # wrong size

end
@testset "lazy" begin

    m = rand(1:99, 3,4)
    @test Transmute{(1,0,2)}(m) == reshape(m, 3,1,4)
    @test Transmute{(1,999,2)}(m) == reshape(m, 3,1,4)
    @test Transmute{(1,nothing,2)}(m) == reshape(m, 3,1,4)
    @test transmute(m, (1,0,2)) == reshape(m, 3,1,4)
    @test transmute(m, (2,99,1)) == reshape(transpose(m), 4,1,3)
    @test transmute(m, (2,nothing,1)) == reshape(transpose(m), 4,1,3)
    @test TransmutedDimsArray(m, (1,0,2)) == reshape(m, 3,1,4)
    @test TransmutedDimsArray(m, (2,99,1)) == reshape(transpose(m), 4,1,3)
    @test TransmutedDimsArray(m, (2,nothing,1)) == reshape(transpose(m), 4,1,3)

    g = Transmute{(1,0,2)}(m)
    t = Transmute{(2,0,1)}(m)

    # setindex!
    g[3,1,4] = 222
    @test m[3,4] == 222
    t[2,1,3] = 333
    @test m[3,2] == 333

    # linear indexing
    @test Base.IndexStyle(g) == IndexLinear()
    @test Base.IndexStyle(t) == IndexCartesian()
    g[4] = 444
    @test m[4] == 444
    t[5] = 555
    @test transpose(m)[5] == 555

    # dropdims
    @test dropdims(g, dims=2) == m
    @test dropdims(g, dims=2) isa TransmutedDimsArray{Int,2,(1,2)}
    @test dropdims(t, dims=2) == m'
    @test dropdims(t, dims=2) isa TransmutedDimsArray{Int,2,(2,1)}

    h = reshape(1:9, 3,1,3)
    f = Transmute{(0,3,2,1)}(h)
    @test size(f) == (1,3,1,3)
    @test dropdims(f, dims=1) == permutedims(h, (3,2,1))
    @test dropdims(f, dims=3).parent == reshape(h,3,3)
    @test dropdims(f, dims=(1,3)) == reshape(h,3,3)'
    @test dropdims(f, dims=(3,1)) == reshape(h,3,3)'

    # errors
    @test_throws ArgumentError Transmute{(2,)}(m)
    # @test_throws ArgumentError Transmute{(2,0,1,0,2)}(m)
    @test_throws ArgumentError transmute(m, (2,))
    # @test_throws ArgumentError transmute(m, (2,0,1,0,2))
    @test_throws Exception TransmutedDimsArray(m, (2,))
    # @test_throws Exception TransmutedDimsArray(m, (2,0,1,0,2))

    # unwrapping
    @test Transmute{(2,0,1)}(m') === Transmute{(1,0,2)}(m)
    v = m[:,1]
    @test Transmute{(2,3,1)}(v |> transpose) === Transmute{(1,0,0)}(v)

    @test Transmute{(2,nothing,1)}(PermutedDimsArray(m,(2,1))) === Transmute{(1,0,2)}(m)
    @test Transmute{(2,nothing,1)}(TransmutedDimsArray(m,(2,1))) === Transmute{(1,0,2)}(m)

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
    @test transpose(Transmute{(2,1)}(m)) isa TransmutedDimsArray{Int,2,(1,2)}
    @test adjoint(Transmute{(0,1)}(v)) isa TransmutedDimsArray{Int,2,(1,0)}

    # linear indexing, for more constructors
    y = zeros(1,2,3);
    @test IndexStyle(transmute(y, (1,0,2,0,3))) == IndexLinear()
    @test IndexStyle(Transmute{(1,0,2,0,3)}(y)) == IndexLinear()
    @test IndexStyle(TransmutedDimsArray(y, (1,0,2,0,3))) == IndexLinear()
    @test IndexStyle(transmute(y, (3,0,2,0,1))) == IndexCartesian()
    @test IndexStyle(Transmute{(3,0,2,0,1)}(y)) == IndexCartesian()
    @test IndexStyle(TransmutedDimsArray(y, (3,0,2,0,1))) == IndexCartesian()

    @test transmute(ones(3) .+ im, (1,))[1] == 1 + im # was an ambiguity error

end
@testset "offset" begin

    o34 = ones(11:13, 21:24)
    rand!(o34)

    @test transmutedims(o34, (2,1)) == permutedims(o34)
    @test axes(transmutedims(o34, (2,0,1))) == (21:24, 1:1, 11:13)

    @test Transmute{(2,0,1)}(o34)[21, 1, 13] == o34[13, 21]

end
@testset "allocations" begin

    y = ones(1,2,3);

    @allocated Transmute{(3,2,0,1)}(y)
    @test 17 > @allocated Transmute{(3,2,0,1)}(y)

    @allocated transmute(y, (3,2,0,1))
    @test 129 > @allocated transmute(y, (3,2,0,1))

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
@testset "transpose" begin

    @test size(_transpose(ones(1,2,3,4), (2,4))) == (1,4,3,2)
    @test size(_transpose(ones(1,2,3,4), Val((2,4)))) == (1,4,3,2)

    @test size(_transpose(ones(1,2,3), (2,5))) == (1,1,3,1,2)
    @test size(_transpose(ones(1,2,3), Val((2,5)))) == (1,1,3,1,2)

end
@testset "diagonal" begin

    for d in [
        TransmutedDimsArray(ones(Int,3), (1,1)),
        transmute(ones(Int,3), (1,1)),
        Transmute{(1,1)}(ones(Int,3)),
        ]
        @test d == [1 0 0; 0 1 0; 0 0 1]
        @test d[2] == 0
        @test (d[3,3] = 33) == 33
        @test d[3,3] == 33
        @test (d[2,1] = 0) == 0
        @test_throws ArgumentError d[1,2] = 99
        @test IndexStyle(d) == IndexCartesian()

        @test sum(d .+ 10) == 90 + 1 + 33
    end

    r = rand(3)
    q = TransmutedDimsArray(r, (1,0,1,1))
    @test q[2,1,2,2] == r[2]
    @test q[2,1,2,3] == 0
    @test_throws ArgumentError q[1,1,1,3] = 99

    # eager
    @test q == transmutedims(r, (1,0,1,1))

end
