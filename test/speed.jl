
using TransmuteDims, BenchmarkTools

M1 = randn(1000,1000);
M2 = randn(1000,1000);
M3 = similar(M1);
V1 = randn(1000);

# Constructors

@btime transpose($M1);                 #   5.880 ns
@btime PermutedDimsArray($M1,(2,1));   # 340.115 ns
@btime TransmutedDimsArray($M1,(2,1)); # 605.966 ns
@btime Transmute{(2,1)}($M1);          #   5.894 ns

const cM1 = randn(1000,1000);
@btime (() -> PermutedDimsArray(cM1,(2,1)))();   # 41.970 ns
@btime (() -> TransmutedDimsArray(cM1,(2,1)))(); # 5.598 ns -- constant prop.

# Broadcasting

@btime $M3 .= $M1 .* $M2;                              # 766.213 μs
@btime $M3 .= $M1 .* $(PermutedDimsArray(M2,(1,2)));   # 769.648 μs
@btime $M3 .= $M1 .* $(TransmutedDimsArray(M2,(1,2))); # 769.523 μs

@btime $M3 .= $M1 .* transpose($M2);                 # 2.179 ms
@btime $M3 .= $M1 .* PermutedDimsArray($M2,(2,1));   # 2.186 ms
@btime $M3 .= $M1 .* $(PermutedDimsArray(M2,(2,1))); # 2.179 ms

# Reductions

@btime sum($M1)                              #   290.741 μs
@btime sum($(PermutedDimsArray(M2,(1,2))))   # 1.399 ms
@btime sum($(TransmutedDimsArray(M2,(1,2)))) # 1.399 ms -- could be faster!
@btime sum($(transpose(M2)))                 # 2.341 ms
@btime sum($(PermutedDimsArray(M2,(2,1))))   # 2.450 ms

@btime sum!($V1, $M1);                              # 291.655 μs
@btime sum!($V1, $(PermutedDimsArray(M2,(1,2))));   # 292.307 μs
@btime sum!($V1, $(TransmutedDimsArray(M2,(1,2)))); # 290.363 μs

# 3-tensors

T1 = randn(100,100,100);
T2 = randn(100,100,100);
T3 = similar(T1);

@btime $T3 .= $T1 .* $T2;                              # 766.960 μs
@btime $T3 .= $T1 .* $(PermutedDimsArray(T2,(1,2,3))); # 758.425 μs

@btime $T3 .= $T1 .* $(PermutedDimsArray(T2,(3,2,1))); # 2.912 ms

# Gap insertion

G0 = randn(100,100);
G1 = reshape(G0, 100,1,100);
G2 = Transmute{(1,0,2)}(G0);
G1 == G2

@btime $T3 .= $T1 .* $G1;  # 555.309 μs
@btime $T3 .= $T1 .* $G2;  # 555.691 μs

G3 = reshape(transpose(G0), 100,1,100);
G4 = Transmute{(2,0,1)}(G0);
G5 = Transmute{(1,0,2)}(transpose(G0));
G3 == G4 == G5

@btime $T3 .= $T1 .* $G3;  # 1.900 ms
@btime $T3 .= $T1 .* $G4;  #   696.364 μs
@btime $T3 .= $T1 .* $G5;  #   689.818 μs -- no penalty!

# Copies

@btime permutedims($M1);                      # 2.658 ms
@btime permutedims($M1, (2,1));               # 2.665 ms
@btime collect(transpose($M1));               # 3.674 ms
@btime collect(PermutedDimsArray($M1,(2,1))); # 3.700 ms

@btime permutedims($T1, (3,2,1));               # 3.421 ms
@btime collect(PermutedDimsArray($T1,(3,2,1))); # 4.607 ms

# In-place

@btime permutedims!($M3, $M1, (2,1));            # 1.682 ms
@btime copy!($M3, PermutedDimsArray($M1,(2,1))); # 2.784 ms

@btime permutedims!($T3, $T1, (3,2,1));            # 2.312 ms
@btime copy!($T3, PermutedDimsArray($T1,(3,2,1))); # 3.675 ms
