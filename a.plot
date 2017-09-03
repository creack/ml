set term png
set output "a.png"
ds = "'1 1' '2 2' '3 3'"

set xrange [-5:5]

h(th0,th1,x) = th0 + th1*x

J(th0,th1,x) = x

s = (do for [i in ds] {
   J(word(i,1),word(i,2),x)
})c

