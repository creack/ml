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
