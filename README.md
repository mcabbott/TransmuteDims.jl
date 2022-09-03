# TransmuteDims.jl

[![Build Status](https://github.com/mcabbott/TransmuteDims.jl/workflows/CI/badge.svg)](https://github.com/mcabbott/TransmuteDims.jl/actions)
[![Docstrings](https://img.shields.io/badge/docs-juliahub-blue?labelColor=333)](https://juliahub.com/docs/TransmuteDims/)

This package provides generalisations of Julia's `permutedims` function and `PermutedDimsArray` wrapper, which allow things other than permutations. These can replace `dropdims` and many uses of `reshape`.

The first generalisation is that you may introduce trivial dimensions. This can be thought of as re-positioning the implicit trivial dimensions beyond `ndims(A)`, such as the 4th and 5th dimensions here:

```julia
A = ones(10,20,30);
ntuple(d -> size(A,d), 5)          # (10, 20, 30, 1, 1)

permutedims(A, (2,3,1)) |> size    # (20, 30, 10)

using TransmuteDims
transmute(A, (4,2,3,5,1)) |> size  # (1, 20, 30, 1, 10)
```

Here `(4,2,3,5,1)` is a valid permutation of `1:5`, but the positions of `4,5` don't matter, so in fact this is normalised to `(0,2,3,0,1)`. Zeros indicate trivial output dimensions.

Second, input dimensions below `ndims(A)` may also be omitted, provided they are of size 1:

```julia
A2 = sum(A, dims=2); size(A2)      # (10, 1, 30)
transmute(A2, (3,1)) |> size       # (30, 10)

try transmute(A, (3,1)) catch err; err end  # ArgumentError, "... not allowed when size(A, 2) = 20"
```

Third, you may also repeat numbers, to place an input dimension "diagonally" into several output dimensions:

```julia
using LinearAlgebra
transmute(1:10, (1,1)) == Diagonal(1:10)  # true

transmute(A, (2,2,0,3,1)) |> size  # (20, 20, 1, 30, 10)
```

The function `transmute` is always lazy, but also tries to minimise the number of wrappers. Ideally to none at all, by un-wrapping and reshaping:

```julia
transmute(A, (4,2,3,5,1)) isa TransmutedDimsArray{Float64, 5, (0,2,3,0,1), (5,2,3), <:Array}

transmute(A, (1,0,2,3)) isa Array{Float64, 4}

transmute(PermutedDimsArray(A, (2,3,1)),(3,1,0,2)) isa Array{Float64, 4}

transmute(Diagonal(1:10), (3,1)) isa TransmutedDimsArray{Int64, 2, (0,1), (2,), <:UnitRange}
transmute(Diagonal(rand(10)), (3,1)) isa Matrix
```

Calling the constructor directly `TransmutedDimsArray(A, (3,2,0,1))` simply applies the wrapper. 
There is also a method `transmute(A, Val((3,2,0,1)))` which works out any un-wrapping at compile-time:

```julia
using BenchmarkTools
@btime transmute($A, (2,3,1));           #   6.996 ns (1 allocation: 16 bytes)
@btime PermutedDimsArray($A, (2,3,1));   # 386.738 ns (4 allocations: 176 bytes)
@btime transmute($A, Val((2,3,1)));      #   1.430 ns (0 allocations: 0 bytes)

@btime transmute($A, (1,2,0,3));         #  45.642 ns (2 allocations: 128 bytes)
@btime reshape($A, (10,20,1,30));        #  34.479 ns (1 allocation: 80 bytes)
```

Finally, there is also an eager variant, which tries always to return a `DenseArray`. 
This will similarly un-wrap `Transpose` etc, and prefers to reshape if possible, copying data only when necessary. 
It uses [Strided.jl](https://github.com/Jutho/Strided.jl) to speed this up, when possible, so should be faster than Base's `permutedims`:

```julia
transmutedims(A, (3,2,0,1)) isa Array{Float64, 4}
transmutedims(1:3, (2,1)) isa Matrix

@btime transmutedims($(rand(40,50,60)), (3,2,1));  #  57.365 μs (61 allocations: 944.62 KiB)
@btime permutedims($(rand(40,50,60)), (3,2,1));    # 172.643 μs (2 allocations: 937.58 KiB)

@strided(transmute(A, (3,2,0,1))) isa StridedView{Float64, 4}
@strided(transmutedims(A, (3,2,0,1))) isa StridedView{Float64, 4}
```

The `StridedView` type is general enough to allow the insertion/removal of trivial dimensions, in addition to permutations, so these functions preserve it.

The lower-case functions also treat tuples as if they were vectors:

```julia
transmute((1,2,3), (1,)) isa AbstractVector
transmutedims((1,2,3), (nothing,1)) isa Matrix
```

### About

This was written largely for [TensorCast.jl](https://github.com/mcabbott/TensorCast.jl).
The immediate issue there was that a `reshape(transpose(::GPUArray))` may fail to trigger GPU broadcasting. 
This package replaces that with at most one wrapper, ideally none.
Calling `transmute` also allowed `@cast` to express what it needs more cleanly.
