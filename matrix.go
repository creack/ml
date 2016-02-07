package ml

import (
	"errors"
	"fmt"
	"math"
	"strings"
)

// Common erros.
var (
	ErrBadDim              = errors.New("bad dimenstion for matrix")
	ErrInconsistentData    = errors.New("matrix has different y dimension per x")
	ErrUninitialized       = errors.New("matrix not initialized")
	ErrIdentityInvalidSize = errors.New("current dimension of matrix does not have an identity")
	ErrNegativeIndex       = errors.New("negative index are not supported")
	ErrOutOfBound          = errors.New("index out of bound")
	ErrNotAVector          = errors.New("the current vector has invalid dimension for a vector")
	ErrSingularMatrix      = errors.New("the matrix is singuler")
)

// MRow is the row type for matrix.
type MRow []float64

// Scale returns the result of the scalar multiplication of the given scalar
// and the current matrix row.
// NOTE: Does not change current row state.
func (mr MRow) Scale(n float64) MRow {
	return Matrix{mr}.Scale(n)[0]
}

// Add adds the given matrix to the current one and return the result.
// NOTE: Does not change current matrix state.
func (mr MRow) Add(mr2 MRow) MRow {
	return Matrix{mr}.Add(Matrix{mr2})[0]
}

// ToVector returns the current row as a vector.
// Note: Not a copy, changes to the vector affect the row.
func (mr MRow) ToVector() Vector {
	v := NewVector(len(mr))
	for i, elem := range mr {
		v[i][0] = elem
	}
	return v
}

// Matrix .
type Matrix []MRow

// NewMatrix instantiates a new matrix of (n,m) dimension.
func NewMatrix(m, n int) Matrix {
	ret := make(Matrix, m)
	for i := 0; i < m; i++ {
		ret[i] = make([]float64, n)
	}
	return ret
}

// ToVector asserts the current matrix as a vector
// and returns it.
func (ma Matrix) ToVector() Vector {
	return ToVector(ma)
}

// Dim returns the dimension of the matrix.
func (ma Matrix) Dim() (int, int) {
	if len(ma) == 0 {
		return 0, 0
	}
	if len(ma[0]) == 0 {
		return len(ma), 1
	}
	return len(ma), len(ma[0])
}

// Add adds the given matrix to the current one and return the result.
// NOTE: Does not change current matrix state.
func (ma Matrix) Add(ma2 Matrix) Matrix {
	if !ma.DimMatch(ma2) {
		panic(ErrBadDim)
	}
	ret := NewMatrix(ma.Dim())
	for i, line := range ma {
		if len(ma[i]) == 0 {
			continue
		}
		for j := range line {
			ret[i][j] = ma[i][j] + ma2[i][j]
		}
	}
	return ret
}

// Sub substracts the given matrix to the current one and return the result.
// NOTE: Does not change current matrix state.
func (ma Matrix) Sub(ma2 Matrix) Matrix {
	if !ma.DimMatch(ma2) {
		panic(ErrBadDim)
	}
	ret := NewMatrix(ma.Dim())
	for i, line := range ma {
		if len(ma[i]) == 0 {
			continue
		}
		for j := range line {
			ret[i][j] = ma[i][j] - ma2[i][j]
		}
	}
	return ret
}

// Mul returns the result of the current matrix multiplied by the given one.
// NOTE: Does not change current matrix state.
func (ma Matrix) Mul(ma2 Matrix) Matrix {
	_, n1 := ma.Dim()
	m2, _ := ma2.Dim()
	if n1 != m2 {
		panic(ErrBadDim)
	}
	ret := NewMatrix(ma.Dim())
	for i := range ma {
		if len(ma[i]) == 0 {
			continue
		}
		for j := range ma2[0] {
			sum := 0.
			for k := range ma[0] {
				sum += ma[i][k] * ma2[k][j]
			}
			ret[i][j] = sum
		}
	}
	return ret
}

// Scale returns the result of the scalar multiplication of the given scalar
// and the current matrix.
// NOTE: Does not change current matrix state.
func (ma Matrix) Scale(n float64) Matrix {
	ret := NewMatrix(ma.Dim())
	for i, line := range ma {
		if len(ma[i]) == 0 {
			continue
		}
		for j := range line {
			ret[i][j] = ma[i][j] * n
		}
	}
	return ret
}

// MulV multiplies the current matrix with the given vector.
// Returns a Vector.
// NOTE: Does not change current matrix state.
func (ma Matrix) MulV(v Vector) Vector {
	return Vector(ma.Mul(Matrix(v)))
}

// Transpose returns a transposed copy of the current matrix.
// NOTE: Does not change current matrix state.
func (ma Matrix) Transpose() Matrix {
	x, y := ma.Dim()
	ret := NewMatrix(y, x)
	for i, line := range ma {
		if len(ma[i]) == 0 {
			continue
		}
		for j := range line {
			ret[j][i] = ma[i][j]
		}
	}
	return ret
}

// Inverse returns the inverted copy of the current matrix.
// NOTE: Does not change current matrix state.
func (ma Matrix) Inverse() Matrix {
	m, n := ma.Dim()
	if m != n {
		panic(ErrBadDim)
	}
	// Step 1: Double the width of the matrix.
	ret := ma.Extend(0, n) // Add 0 rows and n cols.

	// Step 2: Set the right half of the matrix as the identity matrix.
	ret = ret.SetSubMatrix(ret.SubMatrix(0, n, m, n).Identity(), 0, n) // sub matrix starts at (0,n) and has a (m,n) size.

	for i := 0; i < len(ret); i++ {
		if len(ma[i]) == 0 {
			continue
		}
		j := i
		for k := i; k < len(ret); k++ {
			if math.Abs(ret[k][j]) > math.Abs(ret[j][i]) {
				j = k
			}
		}
		if j != i {
			// Swap rows.
			tmp := ret[i]
			ret[i] = ret[j]
			ret[j] = tmp
		}
		if ret[i][i] == 0 {
			panic(ErrSingularMatrix)
		}
		// Inverse the i'th row.
		ret[i] = ret[i].Scale(1 / ret[i][i])
		for k := 0; k < n; k++ {
			if k == i {
				continue
			}
			ret[k] = ret[k].Add(ret[i].Scale(-ret[k][i]))
		}
	}
	return ret.SubMatrix(0, n, m, n)
}

// Identity returns the identify matrix for the current one.
// NOTE: Does not change current matrix state.
func (ma Matrix) Identity() Matrix {
	m, n := ma.Dim()
	if m != n {
		panic(ErrIdentityInvalidSize)
	}
	ret := NewMatrix(m, n) // Default to 0 for all fields.
	for i := 0; i < m; i++ {
		ret[i][i] = 1
	}
	return ret
}

// Equal compares the given matrix to the current one.
func (ma Matrix) Equal(ma2 Matrix) bool {
	// If dim mismatch, mot equal.
	if !ma.DimMatch(ma2) {
		return false
	}
	// Check each element of both matrix.
	m, n := ma.Dim()
	for i := 0; i < m; i++ {
		if len(ma[i]) == 0 {
			if len(ma2[i]) != 0 {
				return false
			}
			continue
		}
		for j := 0; j < n; j++ {
			if ma[i][j] != ma2[i][j] {
				return false
			}
		}
	}
	return true
}

// Validate checks if the current matrix is valid.
// NOTE: When instantiating a matrix outside ml.NewMatrix, Validate should be called.
func (ma Matrix) Validate() error {
	// Empty matrix is valid, but nil one is not.
	if ma == nil {
		return ErrUninitialized
	}
	if len(ma) == 0 {
		return nil
	}
	n := len(ma[0])
	for _, line := range ma {
		if len(line) != n {
			return ErrInconsistentData
		}
	}
	return nil
}

// DimMatch checks if the given matrice has the same dimension as the current one.
func (ma Matrix) DimMatch(ma2 Matrix) bool {
	m, n := ma.Dim()
	m1, n1 := ma2.Dim()
	return m == m1 && n == n1
}

// Extend returns a copy of the current matrix with m more rows and n more cols.
// The extended rows/cols are set to 0.
// Extend(0, 0) creates an identical copy of the matrix.
// NOTE: Does not change state of current matrix.
func (ma Matrix) Extend(m1, n1 int) Matrix {
	if ma == nil {
		return NewMatrix(m1, n1)
	}
	m, n := ma.Dim()
	ret := NewMatrix(m+m1, n+n1)
	for i, line := range ma {
		if len(ma[i]) == 0 {
			continue
		}
		for j := range line {
			ret[i][j] = ma[i][j]
		}
	}
	return ret
}

// Copy returns a copy of the current matrix.
func (ma Matrix) Copy() Matrix {
	return ma.Extend(0, 0)
}

// SubMatrix return a sub matrix part of the current matrix.
// Starts at (m,n) index (0 indexed) and of dimension (m1,n1)
// NOTE: Changes to the sub matrix will change the parent one.
func (ma Matrix) SubMatrix(m, n, m1, n1 int) Matrix {
	if m < 0 || n < 0 || // Negative start index
		m > len(ma) || // ma[m+i], m needs not be smaller than len(ma).
		n+n1 > len(ma[0]) { // [n:n+n1], needs to be within slice length.
		panic(ErrOutOfBound)
	}

	ret := make(Matrix, m1)
	for i := 0; i < len(ret); i++ {
		ret[i] = ma[m+i][n : n+n1]
	}
	return ret
}

// SetSubMatrix updates the current matrix with the given submatrix starting at m,n index.
// NOTE: Changes the state of the current matrix.
// NOTE: Overflowing submatrix produce an error.
func (ma Matrix) SetSubMatrix(ma2 Matrix, m, n int) Matrix {
	m1, n1 := ma.Dim()
	m2, n2 := ma2.Dim()
	if m < 0 || n < 0 || m+m2 > m1 || n+n2 > n1 {
		panic(ErrOutOfBound)
	}
	for i, line := range ma2 {
		for j := range line {
			ma[i+m][j+n] = ma2[i][j]
		}
	}
	return ma
}

// // Row returns the ith row of the current matrix.
// // NOTE: Changes to the row will change the parent matrix.
// func (ma Matrix) Row(i int) MRow {
// 	if i < 0 || i >= len(ma) {
// 		panic(ErrOutOfBound)
// 	}
// 	return ma[i]
// }

// // SetRow sets the given row as ith row of the current matrix.
// // NOTE: Changes the state of the current matrix.
// func (ma Matrix) SetRow(row MRow, i int) Matrix {
// 	if i < 0 || i >= len(ma) {
// 		panic(ErrOutOfBound)
// 	}
// 	if _, n := ma.Dim(); n != len(row) {
// 		panic(ErrBadDim)
// 	}
// 	ma[i] = row
// 	return ma
// }

// String pretty prints the matrix.
func (ma Matrix) String() string {
	if ma == nil {
		return "<nil>"
	}
	if len(ma) == 0 {
		return "||"
	}
	m, n := ma.Dim()
	ret := fmt.Sprintf("(%d,%d)\n", m, n)
	for _, line := range ma {
		ret += fmt.Sprintf("%4v\n", line)
	}
	return strings.TrimSpace(ret)
}

// Vector is a matrix with 1 column.
type Vector Matrix

// NewVector instantiate a new vector of dimensin n.
func NewVector(n int) Vector {
	return Vector(NewMatrix(n, 1))
}

// ToVector converts the matrix type to vector
// and validates the resulting vector.
// panic if the given matrix is not of (1,n) dimension.
func ToVector(m Matrix) Vector {
	v := Vector(m)
	if err := v.Validate(); err != nil {
		panic(err)
	}
	return v
}

// Dim returns the size of the vector. dim (1,n)
func (v Vector) Dim() (int, int) {
	return Matrix(v).Dim()
}

// Validate checks if the current matrix is valid.
// NOTE: When instantiating a matrix outside ml.NewVector, Validate should be called.
func (v Vector) Validate() error {
	if err := Matrix(v).Validate(); err != nil {
		return err
	}
	if len(v) == 0 || len(v[0]) != 1 {
		return ErrNotAVector
	}
	return nil
}

// Transpose return a transposed copy of the vector as a matrix (n,1).
// NOTE: Does not change state of current vector.
func (v Vector) Transpose() Matrix {
	return Matrix(v).Transpose()
}

// Sum computes the sum of all the vector elements.
func (v Vector) Sum() float64 {
	sum := 0.0
	for _, elem := range v[0] {
		sum += elem
	}
	return sum
}

// SubV returns the result of v - v2 as a copy.
// NOTE: Does not change cureent vector state.
func (v Vector) SubV(v2 Vector) Vector {
	return Vector(Matrix(v).Sub(Matrix(v2)))
}

// Scale .
func (v Vector) Scale(n float64) Vector {
	return Vector(Matrix(v).Scale(n))
}

func (v Vector) String() string {
	return Matrix(v).String()
}
