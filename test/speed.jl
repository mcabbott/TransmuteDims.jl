
using TransmuteDims, BenchmarkTools # All times Julia 1.2 or 1.3, Macbook Escape

M1 = randn(1000,1000);
M2 = randn(1000,1000);
M3 = similar(M1);
V1 = randn(1000);

# Constructors

@btime transpose($M1);                 #     5.947 ns (1 allocation: 16 bytes)
@btime PermutedDimsArray($M1,(2,1));   #   490.928 ns (4 allocations: 176 bytes)

@btime Transmute{(2,1)}($M1);          #     6.459 ns (1 allocation: 16 bytes)
@btime transmute($M1, (2,1));          #   563.924 ns (4 allocations: 112 bytes)
@btime TransmutedDimsArray($M1,(2,1)); #   767.825 ns (8 allocations: 592 bytes)

const cM1 = randn(1000,1000);                    # best-case constant propagation
@btime (() -> PermutedDimsArray(cM1,(2,1)))();   #    43.158 ns (2 allocations: 112 bytes)
@btime (() -> transmute(cM1,(2,1)))();           #     5.694 ns (1 allocation: 16 bytes)
@btime (() -> TransmutedDimsArray(cM1,(2,1)))(); #   761.885 ns (8 allocations: 592 bytes)
                                                 # harder case?
@btime (x -> PermutedDimsArray(x, (2,1)))($M1);  #   411.575 ns (4 allocations: 176 bytes)
@btime (x -> transmute(x, (2,1)))($M1);          #   565.357 ns (4 allocations: 112 bytes)


const newaxis = [CartesianIndex()] # another way of making gaps

@btime Transmute{(2,0,1)}($M1);             #  5.636 ns
@btime view(transpose($M1), :, newaxis, :); # 14.715 ns

# Broadcasting

@btime $M3 .= $M1 .* $M2;                              # 766.213 μs
@btime $M3 .= $M1 .* $(PermutedDimsArray(M2,(1,2)));   # 769.648 μs
@btime $M3 .= $M1 .* $(TransmutedDimsArray(M2,(1,2))); # 769.523 μs with IndexCartesian()
@btime $M3 .= $M1 .* $(TransmutedDimsArray(M2,(1,2))); # 857.322 μs with IndexLinear()

@btime $M3 .= $M1 .* transpose($M2);                 # 2.179 ms
@btime $M3 .= $M1 .* PermutedDimsArray($M2,(2,1));   # 2.186 ms
@btime $M3 .= $M1 .* $(PermutedDimsArray(M2,(2,1))); # 2.179 ms
@btime $M3 .= $M1 .* $(Transmute{(2,1)}(M2));        # 2.001 ms

@btime $M3 .= $M1 .* transpose($(PermutedDimsArray(M2,(2,1)))); # 767.724 μs
@btime $M3 .= $M1 .* $(PermutedDimsArray(transpose(M2),(2,1))); # 769.943 μs


# Reductions

@btime sum($M1)                              #   290.741 μs
@btime sum($(PermutedDimsArray(M2,(1,2))))   # 1.399 ms
@btime sum($(TransmutedDimsArray(M2,(1,2)))) # 1.399 ms with IndexCartesian()
@btime sum($(TransmutedDimsArray(M2,(1,2)))) # 1.097 ms with IndexLinear()
@btime sum($(TransmutedDimsArray(M2,(1,2)))) #   293.250 μs with method!

@btime sum($(transpose(M2)))                 # 2.341 ms
@btime sum($(PermutedDimsArray(M2,(2,1))))   # 2.450 ms

@btime sum!($V1, $M1);                              # 291.655 μs
@btime sum!($V1, $(PermutedDimsArray(M2,(1,2))));   # 292.307 μs
@btime sum!($V1, $(TransmutedDimsArray(M2,(1,2)))); # 290.363 μs

# 3-tensors

T1 = randn(100,100,100);
T2 = randn(100,100,100);
T3 = similar(T1);

@btime $T3 .= $T1 .* $T2;                                # 766.960 μs
@btime $T3 .= $T1 .* $(PermutedDimsArray(T2,(1,2,3)));   # 758.425 μs
@btime $T3 .= $T1 .* $(TransmutedDimsArray(T2,(1,2,3))); # 757.789 μs with IndexCartesian()
@btime $T3 .= $T1 .* $(TransmutedDimsArray(T2,(1,2,3))); # 919.709 μs with IndexLinear()

@btime $T3 .= $T1 .* $(PermutedDimsArray(T2,(3,2,1)));   # 2.912 ms

# repeat those when passing unused kw... along?
# @btime $T3 .= $T1 .* $(TransmutedDimsArray(T2,(1,2,3))); # 920.303 μs
# @btime $T3 .= $T1 .* $(TransmutedDimsArray(T2,(3,2,1))); # 3.265 ms

# Gap insertion

G0 = randn(100,100);
G1 = reshape(G0, 100,1,100);
G2 = Transmute{(1,0,2)}(G0);
G6 = @view G0[:, newaxis, :];
G1 == G2 == G6

@btime $T3 .= $T1 .* $G1;  # 555.309 μs
@btime $T3 .= $T1 .* $G2;  # 555.691 μs
@btime $T3 .= $T1 .* $G6;  # 552.592 μs

G3 = reshape(transpose(G0), 100,1,100);
G4 = Transmute{(2,0,1)}(G0);
G5 = Transmute{(1,0,2)}(transpose(G0));
G7 = @view transpose(G0)[:, newaxis, :];
G3 == G4 == G5 == G7

@btime $T3 .= $T1 .* $G3;  # 1.900 ms
@btime $T3 .= $T1 .* $G4;  #   696.364 μs
@btime $T3 .= $T1 .* $G5;  #   687.152 μs
@btime $T3 .= $T1 .* $G7;  #   694.963 μs

summary(G5) # Transpose now unwrapped
G5 === G4

# Dropdims

@btime dropdims($G1, dims=2); # 527.047 ns (9 allocations: 224 bytes)
@btime dropdims($G4, dims=2); # 662.510 ns (4 allocations: 80 bytes)

# Copies

@btime permutedims($M1);                      # 1.851 ms +
@btime permutedims($M1, (2,1));               # 1.769 ms +
@btime collect(transpose($M1));               # 3.122 ms +
@btime collect(PermutedDimsArray($M1,(2,1))); # 3.130 ms +
@btime transmutedims($M1, (2,1));             # 1.773 ms +

@btime permutedims($T1, (3,2,1));               # 3.421 ms
@btime collect(PermutedDimsArray($T1,(3,2,1))); # 4.607 ms

M4 = randn(2,2);
@btime permutedims($M4, (2,1));               #  48.147 ns (1 allocation: 112 bytes)
@btime transmutedims($M4, (2,1));             # 216.911 ns (3 allocations: 176 bytes)

@btime reshape($M4, (2,1,2));                 #  28.114 ns (1 allocation: 80 bytes)
@btime transmutedims($M4, (1,0,2));           # 254.127 ns (5 allocations: 288 bytes)

# In-place

@btime permutedims!($M3, $M1, (2,1));            # 1.701 ms +
@btime copy!($M3, PermutedDimsArray($M1,(2,1))); # 2.686 ms +
@btime transmutedims!($M3, $M1, (2,1));          # 1.702 ms +

@btime permutedims!($T3, $T1, (3,2,1));              # 2.312 ms
@btime copyto!($T3, PermutedDimsArray($T1,(3,2,1))); # 3.684 ms
@btime copyto!($T3, Transmute{(3,2,1)}($T1));        # 3.674 ms

# Inference

@code_warntype (x -> PermutedDimsArray(x, (2, 1)))(M1)
# Body::PermutedDimsArray{Float64,2,_A,_B,Array{Float64,2}} where _B where _A
@code_warntype (x -> TransmutedDimsArray(x, (2, 1)))(M1)
# Body::TransmutedDimsArray{Float64,2,_A,_B,Array{Float64,2},_C} where _C where _B where _A
@code_warntype (x -> Transmute{(2,1)}(x))(M1)
# Body::TransmutedDimsArray{Float64,2,(2, 1),(2, 1),Array{Float64,2},false}
@code_warntype (x -> transmute(x, (2,1)))(M1)
# Body::TransmutedDimsArray{Float64,2,_A,_B,Array{Float64,2},_C} where _C where _B where _A

@code_warntype (() -> Val(invperm((2,1))))()
# Body::Val{(2, 1)}

@code_warntype (x -> view(x,:,newaxis,:))(M1)
# Body::SubArray{Float64,3,Array{Float64,2},...


# Transpose

using TransmuteDims: _transpose

@btime _transpose($M1, (2,1));           # 504.098 ns (3 allocations: 80 bytes)
@btime _transpose($M1, Val((2,1)));      #   5.922 ns (1 allocation: 16 bytes)

@btime (() -> _transpose(cM1, (2,1)))(); #   5.730 ns (1 allocation: 16 bytes)
@btime (() -> _transpose($M1, (2,1)))(); # 502.979 ns (2 allocations: 48 bytes)
@btime (m -> _transpose(m, (2,1)))($M1); # 396.413 ns (2 allocations: 48 bytes)

