package ml_test

import (
	"fmt"
	"testing"

	"github.com/creack/ml"
)

// stringify is a small helper to turn float into
// a static 4 digit precision string.
func stringify(f float64) string {
	return fmt.Sprintf("%.6f", f)
}

// Test simplified version of the squared error.
// The simplification is setting Θ0 to 0.
func TestSimplifiedSquaredError(t *testing.T) {
	var testSimpleDataset = ml.Dataset{
		{X: 1, Y: 1},
		{X: 2, Y: 2},
		{X: 3, Y: 3},
	}

	lr := ml.LinearRegression{Θ0: -0.1, Θ1: 3}
	lr.GradientDescent(testSimpleDataset, 0.001, false)
	println(lr.String())

	plot, err := testSimpleDataset.PlotData()
	if err != nil {
		t.Errorf("Error plotting the graph: %s", err)
	} else {
		t.Logf("%s", plot)
	}

	simplifiedCostPlot, err := testSimpleDataset.PlotLineraRegressionSimplifiedCost(-0.5, 3, 0.5)
	if err != nil {
		t.Errorf("Error plotting the graph: %s", err)
	} else {
		t.Logf("%s", simplifiedCostPlot)
	}

	costPlot, err := testSimpleDataset.PlotLineraRegressionCost(10, -20, -0.5, -20, 10, 0.5)
	if err != nil {
		t.Errorf("Error plotting the graph: %s", err)
	} else {
		t.Logf("%s", costPlot)
	}

	for _, elem := range []struct {
		ml.Hypothesis
		expect float64
	}{
		{ml.LinearRegression{Θ0: 0, Θ1: 1}, 0},
		{ml.LinearRegression{Θ0: 0, Θ1: 0.5}, 0.583333},
		{ml.LinearRegression{Θ0: 0, Θ1: 0}, 2.333333},
		{ml.LinearRegression{Θ0: 0, Θ1: -0.5}, 5.25},
		{ml.LinearRegression{Θ0: 0, Θ1: 2.5}, 5.25},
	} {
		if expect, got := stringify(elem.expect), stringify(elem.Hypothesis.SquaredError(testSimpleDataset)); expect != got {
			t.Errorf("Unexpected simplified squared error for simple dataset\nExpect:\t%s\nGot:\t%s", expect, got)
		}
	}
}

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
