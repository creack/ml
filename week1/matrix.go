package ml

import (
	"errors"
	"fmt"
	"strings"
)

// Common erros.
var (
	ErrBadDim           = errors.New("dimension of the two matrix differs")
	ErrInconsistentData = errors.New("matrix has different y dimension per x")
	ErrUninitialized    = errors.New("matrix not initialized")
	ErrNotAVector       = errors.New("the current vector has multi dimension")
)

// Matrix .
type Matrix [][]float64

// NewMatrix instantiates a new matrix of (n,m) dimension.
func NewMatrix(n, m int) Matrix {
	ret := make(Matrix, n)
	for i := 0; i < n; i++ {
		ret[i] = make([]float64, m)
	}
	return ret
}

// Dim returns the dimension of the matrix.
func (m Matrix) Dim() (int, int) {
	if len(m) == 0 {
		return 0, 0
	}
	return len(m), len(m[0])
}

// Add adds Kthe given matrix to the current one and return the result.
// Does not change current matrix state.
func (m Matrix) Add(m2 Matrix) Matrix {
	if !m.DimMatch(m2) {
		panic(ErrBadDim)
	}
	ret := NewMatrix(m.Dim())
	for i, line := range m {
		for j := range line {
			ret[i][j] = m[i][j] + m2[i][j]
		}
	}
	return ret
}

// Sub substracts the given matrix to the current one and return the result.
// Does not change current matrix state.
func (m Matrix) Sub(m2 Matrix) Matrix {
	if !m.DimMatch(m2) {
		panic(ErrBadDim)
	}
	ret := NewMatrix(m.Dim())
	for i, line := range m {
		for j := range line {
			ret[i][j] = m[i][j] - m2[i][j]
		}
	}
	return ret
}

// Equal compares the given matrix to the current one.
func (m Matrix) Equal(m2 Matrix) bool {
	// If dim mismatch, mot equal.
	if !m.DimMatch(m2) {
		return false
	}
	// Check each element of both matrix.
	x, y := m.Dim()
	for i := 0; i < x; i++ {
		for j := 0; j < y; j++ {
			if m[i][j] != m2[i][j] {
				return false
			}
		}
	}
	return true
}

// Validate checks if the current matrix is valid.
// NOTE: When instantiating a matrix outside ml.NewMatrix, Validate should be called.
func (m Matrix) Validate() error {
	// Empty matrix is valid, but nil one is not.
	if m == nil {
		return ErrUninitialized
	}
	if len(m) == 0 {
		return nil
	}
	y := len(m[0])
	for _, line := range m {
		if len(line) != y {
			return ErrInconsistentData
		}
	}
	return nil
}

// DimMatch checks if the given matrice has the same dimension as the current one.
func (m Matrix) DimMatch(m2 Matrix) bool {
	x, y := m.Dim()
	x1, y1 := m2.Dim()
	return x == x1 && y == y1
}

// String pretty prints the matrix.
func (m Matrix) String() string {
	if m == nil {
		return "<nil>"
	}
	if len(m) == 0 {
		return "||"
	}
	ret := ""
	for _, line := range m {
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

// Validate checks if the current matrix is valid.
// NOTE: When instantiating a matrix outside ml.NewVector, Validate should be called.
func (v Vector) Validate() error {
	if err := Matrix(v).Validate(); err != nil {
		return err
	}
	if len(v) > 0 && len(v[0]) != 1 {
		return ErrNotAVector
	}
	return nil
}
