set terminal png

set xrange [-1.00:1.00]
set yrange [-1.00:1.00]
h(i,j,x) = i + j*x
J(i,j,x,y) = h(i,j,x)-y

do for[j=-1.00:1.00] {
   splot J(0,j,x,y)
}
