
using CuArrays, TransmuteDims, Test
CuArrays.allowscalar(false)

m = rand(4,4)
tm = Transmute{(2,1)}(m)

cm = cu(m)
ctm = cu(tm)
tcm = Transmute{(2,1)}(cm)
@test typeof(tcm) == typeof(ctm)

c2 = cm .* log.(ctm) ./ 2
@test collect(c2) ≈ m .* log.(m') ./ 2

# A fake GPU array for testing:
jm = JLArray(m)
tjm = Transmute{(2,1)}(jm)

j2 = jm .* log.(tjm) ./ 2 # errors
