# TransmuteDims.jl

[![Build Status](https://github.com/mcabbott/TransmuteDims.jl/workflows/CI/badge.svg)](https://github.com/mcabbott/TransmuteDims.jl/actions)

This package provides a generalisation of Julia's `PermutedDimsArray`, which allows things other than permutations.

First, arrays may be thought of as having trivial dimensions beyond `ndims(A)`, which we can re-position like this:

```julia
A = ones(10,20,30);
ntuple(d -> size(A,d), 5)         # (10, 20, 30, 1, 1)

size(permutedims(A, (2,3,1)))     # (20, 30, 10)

using TransmuteDims
size(transmute(A, (4,2,3,5,1)))   # (1, 20, 30, 1, 10)
```

Here `(4,2,3,5,1)` is a valid permutation of `1:5`, but the positions of `4,5` don't matter, so in fact this is normalised to `(0,2,3,0,1)`. Zeros indicate trivial output dimensions.

Second, you may also repeat numbers, to place an input dimension "diagonally" into several output dimensions:

```julia
using LinearAlgebra
transmute(1:10, (1,1)) == Diagonal(1:10) # true

size(transmute(A, (2,2,0,3,1)))   # (20, 20, 1, 30, 10)
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
There is also a variant `transmute(A, Val((3,2,0,1)))` which works out any un-wrapping at compile-time:

```julia
using BenchmarkTools
@btime transmute($A, (2,3,1));           # 376.985 ns (3 allocations: 80 bytes)
@btime PermutedDimsArray($A, (2,3,1));   # 386.738 ns (4 allocations: 176 bytes)
@btime transmute($A, Val((2,3,1)));      #   1.430 ns (0 allocations: 0 bytes)

@btime transmute($A, (1,2,0,3));         #  56.164 ns (2 allocations: 128 bytes)
@btime reshape($A, (10,20,1,30));        #  34.479 ns (1 allocation: 80 bytes)
```

There was going to be an eager variant `transmutedims(A, (3,2,0,1))` but for now that's absent.
