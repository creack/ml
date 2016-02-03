#set terminal dumb
set terminal png
set output "a.png"
set style data lines
set key outside

$ds << EOF
1 1
2 2
3 3
EOF

$d2 << EOF
-1 5.25
0 2.5
1 0
2 2.5
3 5.25
EOF

plot "$ds","$d2"
