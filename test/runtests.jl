using TransmuteDims, Random, Test

@testset "new features" begin

    m = rand(1:99, 3,4)
    @test transmutedims(m, (1,0,2)) == reshape(m, 3,1,4)
    @test transmutedims(m, (2,99,1)) == reshape(transpose(m), 4,1,3)
    @test Transmute{(1,0,2)}(m) == reshape(m, 3,1,4)
    @test Transmute{(1,999,2)}(m) == reshape(m, 3,1,4)
    @test Transmute{(1,nothing,2)}(m) == reshape(m, 3,1,4)

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

    @test_throws ArgumentError TransmutedDimsArray(m, (2,0,1,0,2))

end
@testset "permutedims from Base" begin
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
