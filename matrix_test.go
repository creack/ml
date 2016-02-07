package ml_test

import (
	"testing"

	"github.com/creack/ml"
)

func TestMatrixDimMatch(t *testing.T) {
	m1 := ml.NewMatrix(2, 4)
	m2 := ml.NewMatrix(2, 4)

	if !m1.DimMatch(m2) {
		x1, y1 := m1.Dim()
		x2, y2 := m2.Dim()
		t.Fatalf("Identical dimension didn't match: (%d,%d) vs (%d,%d).", x1, y1, x2, y2)
	}

	m2 = ml.NewMatrix(2, 8)
	if m1.DimMatch(m2) {
		t.Fatal("Different dim matrix matched.")
	}
}

func TestMatrixAddFailure(t *testing.T) {
	defer func() {
		if err := recover(); err == nil {
			t.Fatal("no panic received when adding mismtch dim matrix")
		}
	}()
	m1 := ml.NewMatrix(2, 4)
	m2 := ml.NewMatrix(2, 2)

	m1.Add(m2)
}

func TestMatrixAddSuccess(t *testing.T) {
	ms := []ml.Matrix{
		{
			{1, 2, 1},
			{-1, 22, 3},
		},
		{
			{1, 2, 1},
			{-1, 22, 3},
		},
		{
			{12, 22, 21},
			{-11, 232, 23},
		},
		{
			{14, 26, 23},
			{-13, 276, 29},
		},
	}
	if ret := ms[0].Add(ms[1]).Add(ms[2]); !ret.Equal(ms[3]) {
		t.Fatalf("m0 + m1 + m2 != m3\n%s\n+\n%s\n+\n%s\n!=\n%s\n--->\n%s\n", ms[0], ms[1], ms[2], ms[3], ret)
	}
}

func TestMatrixSubFailure(t *testing.T) {
	defer func() {
		if err := recover(); err == nil {
			t.Fatal("no panic received when substracting mismtch dim matrix")
		}
	}()
	m1 := ml.NewMatrix(2, 4)
	m2 := ml.NewMatrix(2, 2)

	m1.Sub(m2)
}

func TestMatrixSubSuccess(t *testing.T) {
	ms := []ml.Matrix{
		{
			{1, 2, 1},
			{-1, 22, 3},
		},
		{
			{1, 2, 1},
			{-1, 22, 3},
		},
		{
			{12, 22, 21},
			{-11, 232, 23},
		},
		{
			{11, 20, 20},
			{-10, 210, 20},
		},
	}
	if ret := ms[0].Sub(ms[1]).Add(ms[2]).Sub(ms[0]); !ret.Equal(ms[3]) || !ret.Equal(ms[2].Sub(ms[1])) {
		t.Fatalf("m0 + m1 + m2 != m3\n%s\n+\n%s\n+\n%s\n!=\n%s\n--->\n%s\n", ms[0], ms[1], ms[2], ms[3], ret)
	}
}

func TestDim(t *testing.T) {
	if m, n := (ml.Matrix{}).Dim(); m != 0 || n != 0 {
		t.Fatalf("Empty matrix should have dimension (0,0). Got: (%d,%d)", m, n)
	}
	if m, n := (ml.Matrix{{}}).Dim(); m != 1 || n != 1 {
		t.Fatalf("Empty 1x1 matrix should have dimension (1,1). Got: (%d,%d)", m, n)
	}
	if m, n := (ml.Matrix{{1}}).Dim(); m != 1 || n != 1 {
		t.Fatalf("1x1 matrix should have dimension (1,1). Got: (%d,%d)", m, n)
	}
	if m, n := (ml.Matrix{{1}, {1}}).Dim(); m != 2 || n != 1 {
		t.Fatalf("2x1 matrix should have dimension (2,1). Got: (%d,%d)", m, n)
	}
	if m, n := (ml.Matrix{{1, 1}, {1, 1}}).Dim(); m != 2 || n != 2 {
		t.Fatalf("2x2 matrix should have dimension (2,2). Got: (%d,%d)", m, n)
	}
	if m, n := (ml.Matrix{{1, 1, 1}, {1, 1, 1}}).Dim(); m != 2 || n != 3 {
		t.Fatalf("2x3 matrix should have dimension (2,3). Got: (%d,%d)", m, n)
	}
}

func TestEqual(t *testing.T) {
	msIn := []ml.Matrix{
		{},
		{{}},
		{{1}},
		{
			{1, 2, 1},
		},
		{
			{-1},
			{22},
			{3},
		},
		{
			{12, 22},
			{-11, 232},
		},
		{
			{11, 20, 20},
			{-10, 210, 20},
		},
	}
	msOut := []ml.Matrix{
		{},
		{{}},
		{{1}},
		{
			{1, 2, 1},
		},
		{
			{-1},
			{22},
			{3},
		},
		{
			{12, 22},
			{-11, 232},
		},
		{
			{11, 20, 20},
			{-10, 210, 20},
		},
	}
	for i := range msIn {
		if !msIn[i].Equal(msOut[i]) {
			t.Fatalf("Unexpected mismatch\n%s\n!=\n%s\n", msIn[i], msOut[i])
		}
	}
}

func TestTranspose(t *testing.T) {
	msIn := []ml.Matrix{
		{},
		{{}}, // Default value of type: 0. Equivalent to {{0}}.
		{{1}},
		{
			{1, 2, 1},
		},
		{
			{-1},
			{22},
			{3},
		},
		{
			{12, 22},
			{-11, 232},
		},
		{
			{11, 20, 20},
			{-10, 210, 20},
		},
	}
	msOut := []ml.Matrix{
		{},
		{{0}}, // {{}} transposes to {{0}} as 0 is the default value of the type.
		{{1}},
		{
			{1},
			{2},
			{1},
		},
		{
			{-1, 22, 3},
		},
		{
			{12, -11},
			{22, 232},
		},
		{
			{11, -10},
			{20, 210},
			{20, 20},
		},
	}
	for i := range msIn {
		if ret := msIn[i].Transpose(); !ret.Equal(msOut[i]) {
			t.Fatalf("Unexpected value for transpose\n%sT\n!=\n%s\n--->\n%s\n", msIn[i], msOut[i], ret)
		}
	}
}

func TestSubMatrix(t *testing.T) {
	ms := []ml.Matrix{
		{
			{1, 2, 42, 21},
			{12, 52, 32, 21},
			{3, 22, 22, 1},
			{4, 23, 12, 1},
		},
		{{1}},
		{
			{1, 2},
			{12, 52},
		},
		{
			{2, 42},
			{52, 32},
		},
		{
			{12, 52},
			{3, 22},
			{4, 23},
		},
		{
			{52, 32, 21},
			{22, 22, 1},
			{23, 12, 1},
		},
	}
	for i, elem := range []struct {
		m, n, m1, n1 int
		parent       ml.Matrix
		expect       ml.Matrix
	}{
		{0, 0, 1, 1, ms[0], ms[1]},
		{0, 0, 2, 2, ms[0], ms[2]},
		{0, 1, 2, 2, ms[0], ms[3]},
		{1, 0, 3, 2, ms[0], ms[4]},
		{1, 1, 3, 3, ms[0], ms[5]},
	} {
		if subM := elem.parent.SubMatrix(elem.m, elem.n, elem.m1, elem.n1); !subM.Equal(elem.expect) {
			t.Fatalf("[%d] Unexpected submatrix starting at (%d,%d) of size (%d,%d)\nParent:\n%s\nGot:\n%s\nExpect:\n%s\n", i, elem.m, elem.n, elem.m1, elem.n1, elem.parent, subM, elem.expect)
		}
	}
}

func TestExtend(t *testing.T) {
	ms := []ml.Matrix{
		{
			{1, 3, 3},
			{1, 4, 3},
			{1, 3, 4},
		},
		{
			{1, 3, 3, 0, 0, 0},
			{1, 4, 3, 0, 0, 0},
			{1, 3, 4, 0, 0, 0},
		},
		{
			{1, 3, 3, 1, 0, 0},
			{1, 4, 3, 0, 1, 0},
			{1, 3, 4, 0, 0, 1},
		},
	}
	ext := ms[0].Extend(0, 3)
	if !ext.Equal(ms[1]) {
		t.Fatalf("Unexpected extend result\nm1:\n%s\ngot:\n%s\nexpect:\n%s\n", ms[0], ext, ms[1])
	}
	right := ext.SubMatrix(0, 3, 3, 3)
	ext = ext.SetSubMatrix(right.Identity(), 0, 3)
	if !ext.Equal(ms[2]) {
		t.Fatalf("Unexpected extend with right identity result\nm1:\n%s\ngot:\n%s\nexpect:\n%s\n", ms[0], ext, ms[2])
	}
}

func TestInverse(t *testing.T) {
	m1 := ml.Matrix{
		{1, 3, 3},
		{1, 4, 3},
		{1, 3, 4},
	}
	m2 := ml.Matrix{
		{7, -3, -3},
		{-1, 1, 0},
		{-1, 0, 1},
	}
	// Check that m1^-1 == m2.
	if inv := m1.Inverse(); !inv.Equal(m2) {
		t.Fatalf("m1 ^ -1 != m2\nm1:\n%s\nm1^1 got:\n%s\nm1^1 expect:\n%s\n", m1, inv, m2)

	}
	// Check that m1 * m2 == I (m1 * m1^-1)
	if !m2.Mul(m1).Equal(ml.NewMatrix(m1.Dim()).Identity()) {
		t.Fatalf("constants m2 * m1 with m2 = m1^-1 is not the Identity\n%s\n*\n%s\n--->\n%s\n", m2, m1, m2.Mul(m1))
	}
	if !m1.Inverse().Mul(m1).Equal(ml.NewMatrix(m1.Dim()).Identity()) {
		t.Fatalf("m1 ^ -1 * m1 is not the Identity\n%s\n*\n%s\n--->\n%s\n", m2, m1, m2.Mul(m1))
	}
}
