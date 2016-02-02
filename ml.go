package ml

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"strings"
)

// Hypothesis .
type Hypothesis interface {
	Fct(float64) float64                  // h.Fct(x) = y.
	SquaredError(dataset Dataset) float64 // Cost function.
}

// Dataset .
type Dataset []DatasetEntry

// PlotData returns the gnuplot generated ascii graph of the current dataset.
func (ds Dataset) PlotData() (string, error) {
	data := "'-' using 1:2\n"
	for _, entry := range ds {
		data += fmt.Sprintf("%f %f\n", entry.X, entry.Y)
	}

	cmd := exec.Command("gnuplot")
	cmd.Stdin = strings.NewReader("set terminal dumb\nset style data lines\nplot " + data)
	buf, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("Error plotting the graph: %s", err)
	}
	return fmt.Sprintf("\n    %s\n", bytes.TrimSpace(buf)), nil
}

// PlotLineraRegressionCost returns a gnuplot formatted data list.
func (ds Dataset) PlotLineraRegressionCost(x1, x2, xRate, y1, y2, yRate float64) (string, error) {
	data := "'-'\n"
	for i := x1; i < x2; i += xRate {
		for j := y1; j < y2; j += yRate {
			data += fmt.Sprintf("%f %f %f\n", i, j, LinearRegression{Θ0: i, Θ1: j}.SquaredError(ds))
		}
	}
	gnuplotData := "set terminal dumb\nset hidden3d\nset dgrid3d 50,50 qnorm 2\nsplot " + data + "\ne\n"
	ioutil.WriteFile("graph", []byte(gnuplotData), os.ModePerm)
	cmd := exec.Command("gnuplot")
	cmd.Stdin = strings.NewReader(gnuplotData)
	buf, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("Error plotting the graph: %s", err)
	}
	return fmt.Sprintf("\n    %s\n", bytes.TrimSpace(buf)), nil
}

// DatasetEntry .
type DatasetEntry struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
}

// LinearRegression is the linar regression algorithm.
//   - Hypothesis: h(x) = Θ0 + Θ1*x
type LinearRegression struct {
	Θ0 float64
	Θ1 float64
}

// Fct implements the hypothesis function.
func (b LinearRegression) Fct(x float64) float64 {
	return b.Θ0 + b.Θ1*x
}

// // Plot returns a gnuplot formatted data list.
// func (b LinearRegression) Plot(dataset Dataset) (string, error) {
// 	data := "'-' using 1:2\n"
// 	for _, entry := range dataset {
// 		data += fmt.Sprintf("%f %f\n", entry.X, entry.Y)
// 	}

// 	cmd := exec.Command("gnuplot")
// 	cmd.Stdin = strings.NewReader("set terminal dumb\nset palette rgbformulae 33,13,10\nplot " + data)
// 	buf, err := cmd.CombinedOutput()
// 	if err != nil {
// 		return "", fmt.Errorf("Error plotting the graph: %s", err)
// 	}
// 	return fmt.Sprintf("\n    %s\n", bytes.TrimSpace(buf)), nil
// }

// SquaredError process the squared error of the given hypothesis
// on the given dataset.
// The squared error equation is:
// $$\frac{1}{2m}\sum_{i=1}^{m} (h(x^{(i)})-y^{(i)})^2 $$
// - 1/(2m) * Sum from i=1 to m of square h(x(i))-y(i).
func (b LinearRegression) SquaredError(dataset Dataset) float64 {
	var sum float64

	m := len(dataset)
	for i := 1; i <= m; i++ {
		ret := b.Fct(dataset[i-1].X) // - dataset[i-1].Y
		sum += ret * ret
	}
	// 1/2m * sum.
	return (1 / (2 * float64(m))) * sum
}

// Minimize .
func (b LinearRegression) Minimize() {

}
