# TransmuteDims.jl

[![Build Status](https://travis-ci.org/mcabbott/TransmuteDims.jl.svg?branch=master)](https://travis-ci.org/mcabbott/TransmuteDims.jl)

In some senses Julia's arrays have trivial dimensions beyond `ndims(A)`:

```julia
A = ones(10,20,30);
Tuple(size(A,d) for d=1:5)  # (10, 20, 30, 1, 1)

B = ones(1,20,1,40,50);
size(A .+ B .+ 1)           # (10, 20, 30, 40, 50)
```

The function `transmutedims` extends `permutedims` to understand these trivial dimensions:

```julia
size(permutedims(A, (2,3,1)))        # (20, 30, 10)

using TransmuteDims
size(transmutedims(A, (4,2,3,5,1)))  # (1, 20, 30, 1, 10)
```

Here `(4,2,3,5,1)` is a valid permutation of `1:5`, but the positions of `4,5` don't matter, 
so `(0,2,3,0,1)` is treated identically. Things like `(nothing,2,3,nothing,1)` are also allowed.
It is also allowed to repeat numbers, which places that dimension diagonally into 
several output dimensions:

```julia
using LinearAlgebra
transmutedims(1:10, (1,1)) == diagm(1:10) # true

size(transmutedims(A, (2,2,0,3,1)))     # (20, 20, 1, 30, 10)
```

The lazy `TransmutedDimsArray` similarly extends `PermutedDimsArray`. 
The most efficient way to construct these is to call `Transmute{perm}(A)`, 
if `perm` is known at compile-time:

```julia
using BenchmarkTools
@btime PermutedDimsArray($A, (2,3,1));  # 843.706 ns

@btime Transmute{(0,2,3,0,1)}($A);      #   6.452 ns 
```

It is smart enough to unwrap `Transpose` amd `PermutedDimsArray` etc, 
by altering the permutation to leave just one wrapper:

```julia
C = PermutedDimsArray(A, (2,3,1));

summary(Transmute{(3,2,0,1)}(C))     # 10×30×1×20 TransmutedDimsArray(::Array{Float64,3}, (1, 3, 0, 2))

IndexStyle(C)                        # IndexCartesian()
IndexStyle(Transmute{(3,1,2,0)}(C))  # IndexLinear()
```

The original motivation for this
in [NamedPlus.jl](https://github.com/mcabbott/NamedPlus.jl) was to align arrays for broadcasting
according to their axis names, not positions.

<!--
```julia
using TransmuteDims: _transpose

size(_transpose(A))         # (20, 10, 30)
_transpose(A) == transmute(A, (2,1,3))

size(_transpose(A, (2,5)))  # (10, 1, 30, 1, 20)
_transpose(A, (2,5)) == transmute(A, (1,0,3,0,2))
```
-->
