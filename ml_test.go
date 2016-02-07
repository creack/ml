package ml_test

import (
	"fmt"
	"math"
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
		X: ml.Matrix{
			{1},
			{2},
			{3},
		},
		Y: ml.Vector{
			{1},
			{2},
			{3},
		},
	}

	// lr := ml.LinearRegression{Θ0: -0.1, Θ1: 3}
	// lr.GradientDescent(testSimpleDataset, 0.001, false)
	// println(lr.String())

	// plot, err := testSimpleDataset.PlotData()
	// if err != nil {
	// 	t.Errorf("Error plotting the graph: %s", err)
	// } else {
	// 	t.Logf("%s", plot)
	// }

	// simplifiedCostPlot, err := testSimpleDataset.PlotLineraRegressionSimplifiedCost(-0.5, 3, 0.5)
	// if err != nil {
	// 	t.Errorf("Error plotting the graph: %s", err)
	// } else {
	// 	t.Logf("%s", simplifiedCostPlot)
	// }

	// costPlot, err := testSimpleDataset.PlotLineraRegressionCost(10, -20, -0.5, -20, 10, 0.5)
	// if err != nil {
	// 	t.Errorf("Error plotting the graph: %s", err)
	// } else {
	// 	t.Logf("%s", costPlot)
	// }

	for i, elem := range []struct {
		ml.Hypothesis
		expect float64
	}{
		{ml.LinearRegression{Θ: ml.Vector{0: {0}, 1: {1}}}, 0},
		{ml.LinearRegression{Θ: ml.Vector{0: {0}, 1: {0.5}}}, 0.583333},
		{ml.LinearRegression{Θ: ml.Vector{0: {0}, 1: {0}}}, 2.333333},
		{ml.LinearRegression{Θ: ml.Vector{0: {0}, 1: {-0.5}}}, 5.25},
		{ml.LinearRegression{Θ: ml.Vector{0: {0}, 1: {2.5}}}, 5.25},
	} {
		if expect, got := stringify(elem.expect), stringify(elem.Hypothesis.SquaredError(testSimpleDataset)); expect != got {
			t.Errorf("[%d] Unexpected simplified squared error for simple dataset\nExpect:\t%s\nGot:\t%s", i, expect, got)
		}
	}
}

func TestGradientDescent(t *testing.T) {
	var testSimpleDataset = ml.Dataset{
		X: ml.Matrix{
			{1},
			{2},
			{3},
		},
		Y: ml.Vector{
			{1},
			{2},
			{3},
		},
	}
	var parameters = ml.Vector{
		{-0.1},
		{3},
	}

	lr := &ml.LinearRegression{Θ: parameters}
	lr.GradientDescent(testSimpleDataset, 0.1, false)
	if expect, got := stringify(0.), stringify(math.Abs(lr.Θ[0][0])); expect != got {
		t.Fatalf("Unexpected Θ0 for gradient descent.\nExpect:\t%s\nGot:\t%s", expect, got)
	}
	if expect, got := stringify(1.), stringify(lr.Θ[1][0]); expect != got {
		t.Fatalf("Unexpected Θ0 for gradient descent.\nExpect:\t%s\nGot:\t%s", expect, got)
	}
}
