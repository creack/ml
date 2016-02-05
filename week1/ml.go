package ml

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"log"
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

// DatasetEntry .
type DatasetEntry struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
}

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

// PlotLineraRegressionSimplifiedCost renders the gnuplot graph of the simplified cost function (Θ0 = 0).
func (ds Dataset) PlotLineraRegressionSimplifiedCost(y1, y2, yRate float64) (string, error) {
	data := "'-'\n"
	for j := y1; (yRate > 0 && j < y2) || (yRate < 0 && j > y2); j += yRate {
		data += fmt.Sprintf("%f %f\n", j, LinearRegression{Θ0: 0, Θ1: j}.SquaredError(ds))
	}
	gnuplotData := "set terminal dumb\nset style data lines\nplot " + data + "\ne\n"
	_ = ioutil.WriteFile("simplified_cost.plot", []byte(gnuplotData), os.ModePerm)
	cmd := exec.Command("gnuplot")
	cmd.Stdin = strings.NewReader(gnuplotData)
	buf, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("Error plotting the graph: %s", err)
	}
	return fmt.Sprintf("\n    %s\n", bytes.TrimSpace(buf)), nil
}

// PlotLineraRegressionCost renders the gnuplot graph of the simplified cost function (Θ0 = 0).
func (ds Dataset) PlotLineraRegressionCost(x1, x2, xRate, y1, y2, yRate float64) (string, error) {
	data := "'-'\n"
	for i := x1; (xRate > 0 && i < x2) || (xRate < 0 && i > x2); i += xRate {
		for j := y1; (yRate > 0 && j < y2) || (yRate < 0 && j > y2); j += yRate {
			data += fmt.Sprintf("%f %f %f\n", i, j, LinearRegression{Θ0: i, Θ1: j}.SquaredError(ds))
		}
	}
	gnuplotData := "set terminal dumb\nset hidden3d\nset dgrid3d 50,50 qnorm 2\nset xrange [*:] reverse\nset grid ztics\nset grid ytics\nset grid xtics\nset zrange [0:]\nset style data lines\nsplot " + data + "\ne\n"
	_ = ioutil.WriteFile("graph", []byte(gnuplotData), os.ModePerm)
	cmd := exec.Command("gnuplot")
	cmd.Stdin = strings.NewReader(gnuplotData)
	buf, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("Error plotting the graph: %s", err)
	}
	return fmt.Sprintf("\n    %s\n", bytes.TrimSpace(buf)), nil
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

// Plot returns a gnuplot formatted data list.
func (b LinearRegression) Plot(dataset Dataset) (string, error) {
	data := "'-' title \"dataset\"\n"
	for _, entry := range dataset {
		data += fmt.Sprintf("%f %f\n", entry.X, entry.Y)
	}
	data += "e"
	data2 := fmt.Sprintf("h(x) = %f + %f * x\nplot h(x)", b.Θ0, b.Θ1)
	cmd := exec.Command("gnuplot")
	cmd.Stdin = strings.NewReader("set terminal png\nset xrange [0:4]\nset yrange [0:4]\nset multiplot layout 1,2\nset size 0.5,1\nplot " + data + "\nset size 0.5,1\nset style data lines\n" + data2 + "\n")
	buf, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("Error plotting the graph: %s", err)
	}
	return string(buf), nil
}

// SquaredError process the squared error of the given hypothesis
// on the given dataset.
// The squared error equation is:
// $$\frac{1}{2m}\sum_{i=1}^{m} (h(x^{(i)})-y^{(i)})^2 $$
// - 1/(2m) * Sum from i=1 to m of square h(x(i))-y(i).
func (b LinearRegression) SquaredError(dataset Dataset) float64 {
	var sum float64

	m := len(dataset)
	for i := 1; i <= m; i++ {
		ret := b.Fct(dataset[i-1].X) - dataset[i-1].Y
		sum += ret * ret
	}
	// 1/2m * sum.
	return (1 / (2 * float64(m))) * sum
}

// PartialDerivative .
func (b LinearRegression) PartialDerivative(dataset Dataset, j int) float64 {
	var sum float64

	m := len(dataset)
	for i := 1; i <= m; i++ {
		ret := b.Fct(dataset[i-1].X) - dataset[i-1].Y
		if j != 0 {
			ret *= dataset[i-1].X * float64(j)
		}
		sum += ret
	}
	// 1/2 * sum.
	return (1 / float64(m)) * sum
}

// GradientDescent .
func (b *LinearRegression) GradientDescent(dataset Dataset, alpha float64, plotData bool) <-chan string {
	ch := make(chan string)
	go func() {
		defer close(ch)
		for i := 0; i < 1e9; i++ {
			if int(b.SquaredError(dataset)*1e10) == 0 {
				println("----> converged in ", i, "steps")
				return
			}
			b.Θ0 = b.Θ0 - alpha*b.PartialDerivative(dataset, 0)
			b.Θ1 = b.Θ1 - alpha*b.PartialDerivative(dataset, 1)
			if plotData && i%100 == 0 {
				println(b.String())
				p, err := b.Plot(dataset)
				if err != nil {
					log.Printf("fail plot: %s\n", err)
					return
				}
				ch <- p
			}
		}
		log.Printf("didn't converge in 10^9 iterations")
	}()
	if !plotData {
		<-ch
	}
	return ch
}

func (b LinearRegression) String() string {
	return fmt.Sprintf("Θ0: %f, Θ1: %f\n", b.Θ0, b.Θ1)
}
