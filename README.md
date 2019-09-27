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

The lazy `TransmutedDimsArray` similarly extends `PermutedDimsArray`,
except that where possible it supports linear indexing,
and it is smart enough to unwrap `transpose(A)` by altering the permutation.
Calling `Transmute{perm}(A)` will avoid creation overhead:

```julia
using BenchmarkTools
@btime size(PermutedDimsArray($A, (2,3,1)))  # 644.509 ns

@btime size(Transmute{(0,2,3,0,1)}($A))      #   1.327 ns
```

Perhaps there are other uses, but the motivation for this 
in [NamedPlus.jl](https://github.com/mcabbott/NamedPlus.jl) is to align arrays for broadcasting
according to their axis names, not positions. 
