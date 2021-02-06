
julia> using TransmuteDims, BenchmarkTools

julia> VERSION
v"1.7.0-DEV.406"

julia> V1 = randn(1000);

julia> M1 = randn(1000,1000);
julia> M2 = randn(1000,1000);
julia> M3 = similar(M1);

julia> T1 = randn(100,100,100);
julia> T2 = randn(100,100,100);
julia> T3 = similar(T1);

julia> G2 = randn(100,1,100);

#========== Constructors ==========#
# Aim to be faster than PermutedDimsArray.

julia> @btime PermutedDimsArray($T1, (3,2,1));
  381.424 ns (4 allocations: 176 bytes)

julia> @btime TransmutedDimsArray($T1, (3,2,1));
  338.963 ns (3 allocations: 80 bytes)

julia> @btime transmute($T1, (3,2,1));  # fast at last!
  1.430 ns (0 allocations: 0 bytes)

julia> @btime transmute($T1, Val((3,2,1)));
  1.430 ns (0 allocations: 0 bytes)

# ... transpose

julia> @btime transpose($M1);
  1.430 ns (0 allocations: 0 bytes)

julia> @btime transmute($M1, (2,1));
  1.430 ns (0 allocations: 0 bytes)

julia> @btime transmute($M1, Val((2,1)));
  1.430 ns (0 allocations: 0 bytes)

# ... reshape

julia> @btime reshape($M1, (1000, 1, 1000));
  31.764 ns (1 allocation: 80 bytes)

julia> @btime transmute($M1, (1,0,2));
  40.796 ns (2 allocations: 112 bytes)

julia> @btime transmute($M1, Val((1,0,2)));
  40.613 ns (2 allocations: 112 bytes)

# Best-case constant prop:

julia> const cT1 = randn(100,100,100);

julia> @btime (() -> PermutedDimsArray(cT1,(3,2,1)))();
  41.292 ns (1 allocation: 96 bytes)

julia> @btime (() -> TransmutedDimsArray(cT1,(3,2,1)))();
  1.428 ns (0 allocations: 0 bytes)

julia> @btime (() -> transmute(cT1,(3,2,1)))();
  1.428 ns (0 allocations: 0 bytes)

# ... and a harder case:

julia> @btime (x -> PermutedDimsArray(x, (3,2,1)))($T1);
  385.429 ns (4 allocations: 176 bytes)

julia> @btime (x -> TransmutedDimsArray(x, (3,2,1)))($T1);
  338.977 ns (3 allocations: 80 bytes)

julia> @btime (x -> transmute(x, (3,2,1)))($T1);
  1.430 ns (0 allocations: 0 bytes)

#========== Broadcasting ==========#
# Here PermutedDimsArray is great, aim to be no slower.
# Times at right are from earlier version, Julia 1.2?

julia> @btime $M3 .= $M1 .* $M2;                              # 766.213 μs
  771.604 μs (0 allocations: 0 bytes)

julia> @btime $M3 .= $M1 .* $(PermutedDimsArray(M2,(1,2)));   # 769.648 μs
  772.055 μs (0 allocations: 0 bytes)

julia> @btime $M3 .= $M1 .* $(TransmutedDimsArray(M2,(1,2))); # 769.523 μs with IndexCartesian()
  771.612 μs (0 allocations: 0 bytes)

# transposed

julia> @btime $M3 .= $M1 .* transpose($M2);                 # 2.179 ms
  1.890 ms (0 allocations: 0 bytes)

julia> @btime $M3 .= $M1 .* PermutedDimsArray($M2,(2,1));   # 2.186 ms
  1.897 ms (6 allocations: 240 bytes)

julia> @btime $M3 .= $M1 .* $(PermutedDimsArray(M2,(2,1))); # 2.179 ms
  1.889 ms (0 allocations: 0 bytes)

julia> @btime $M3 .= $M1 .* $(TransmutedDimsArray(M2,(2,1))); # 2.179 ms
  1.877 ms (0 allocations: 0 bytes)

#========== Reductions ==========#
# These should be faster than PermutedDimsArray, passed through.

julia> @btime sum($M1);                              #   290.741 μs
  294.625 μs (0 allocations: 0 bytes)

julia> @btime sum($(PermutedDimsArray(M2,(1,2))));   # 1.399 ms
  1.493 ms (0 allocations: 0 bytes)

julia> @btime sum($(TransmutedDimsArray(M2,(1,2)))); #   293.250 μs with method!
  301.680 μs (0 allocations: 0 bytes)

# transposed

julia> @btime sum($(transpose(M2)));                 # 2.341 ms
  2.044 ms (0 allocations: 0 bytes)

julia> @btime sum($(PermutedDimsArray(M2,(2,1))));   # 2.450 ms
  2.619 ms (0 allocations: 0 bytes)

julia> @btime sum($(TransmutedDimsArray(M2,(2,1))));
  301.856 μs (0 allocations: 0 bytes)

# partial

julia> @btime sum!($V1, $M1);                              # 291.655 μs
  316.322 μs (0 allocations: 0 bytes)

julia> @btime sum!($V1, $(PermutedDimsArray(M2,(1,2))));   # 292.307 μs
  316.824 μs (0 allocations: 0 bytes)

julia> @btime sum!($V1, $(TransmutedDimsArray(M2,(1,2)))); # 290.363 μs
  316.575 μs (0 allocations: 0 bytes)

# partial, transposed

julia> @btime sum!($V1, $(transpose(M1)));
  1.659 ms (0 allocations: 0 bytes)

julia> @btime sum!($V1, $(PermutedDimsArray(M2,(2,1))));
  1.540 ms (0 allocations: 0 bytes)

julia> @btime sum!($V1, $(TransmutedDimsArray(M2,(2,1))));
  294.152 μs (3 allocations: 112 bytes)  # why the allocations?

# tiny

julia> M0 = rand(5,5);

julia> @btime sum($M0, dims=2);
  238.289 ns (5 allocations: 208 bytes)

julia> @btime sum($(transpose(M0)), dims=1);
  73.177 ns (1 allocation: 128 bytes)

julia> @btime sum($(TransmutedDimsArray(M0,(2,1))), dims=1);
  73.045 ns (1 allocation: 128 bytes)

#========== Dropdims ==========#
# Reshape is faster than Base's dropdims.

julia> @btime dropdims($G2, dims=2);
  337.256 ns (9 allocations: 224 bytes)

julia> @btime transmute($G2, (1,3));  # reshape
  39.945 ns (2 allocations: 96 bytes)

julia> @btime transmute($G2, (3,1));  # wrap
  4.195 ns (0 allocations: 0 bytes)


#========== Collected ==========#
#

julia> @btime permutedims!($M3, $M2, (2,1));
  1.509 ms (0 allocations: 0 bytes)

julia> @btime copyto!($M3, $(transpose(M2)));
  2.467 ms (0 allocations: 0 bytes)

julia> @btime copyto!($M3, $(PermutedDimsArray(M2, (2,1)))); # same with M3 .=
  3.361 ms (0 allocations: 0 bytes)

julia> @btime copyto!($M3, $(TransmutedDimsArray(M2, (2,1))));
  1.569 ms (0 allocations: 0 bytes)

# three dimensions

julia> @btime permutedims!($T3, $T1, (3,2,1));
  2.283 ms (0 allocations: 0 bytes)

julia> @btime copyto!($T3, $(PermutedDimsArray(T1, (3,2,1))));
  4.528 ms (0 allocations: 0 bytes)

julia> @btime copyto!($T3, $(TransmutedDimsArray(T1, (3,2,1))));
  2.292 ms (0 allocations: 0 bytes)

# fast library

julia> using Strided

julia> @btime @strided permutedims!($M3, $M2, (2,1));
  605.804 μs (66 allocations: 6.81 KiB)

julia> @btime @strided permutedims!($T3, $T1, (3,2,1));
  520.262 μs (65 allocations: 7.38 KiB)

