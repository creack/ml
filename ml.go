package ml

import (
	"fmt"
	"log"
)

// Hypothesis .
type Hypothesis interface {
	Fct(Vector) float64                   // h.Fct(x) = y.
	SquaredError(dataset Dataset) float64 // Cost function.
}

// Dataset .
type Dataset struct {
	X Matrix `json:"x"`
	Y Vector `json:"y"`
}

// // PlotData returns the gnuplot generated ascii graph of the current dataset.
// func (ds Dataset) PlotData() (string, error) {
// 	data := "'-' using 1:2\n"
// 	for _, entry := range ds {
// 		data += fmt.Sprintf("%f %f\n", entry.X, entry.Y)
// 	}

// 	cmd := exec.Command("gnuplot")
// 	cmd.Stdin = strings.NewReader("set terminal dumb\nset style data lines\nplot " + data)
// 	buf, err := cmd.CombinedOutput()
// 	if err != nil {
// 		return "", fmt.Errorf("Error plotting the graph: %s", err)
// 	}
// 	return fmt.Sprintf("\n    %s\n", bytes.TrimSpace(buf)), nil
// }

// // PlotLineraRegressionSimplifiedCost renders the gnuplot graph of the simplified cost function (Θ[0][0] = 0).
// func (ds Dataset) PlotLineraRegressionSimplifiedCost(y1, y2, yRate float64) (string, error) {
// 	data := "'-'\n"
// 	for j := y1; (yRate > 0 && j < y2) || (yRate < 0 && j > y2); j += yRate {
// 		data += fmt.Sprintf("%f %f\n", j, LinearRegression{Θ: Vector{{0}, {j}}}.SquaredError(ds))
// 	}
// 	gnuplotData := "set terminal dumb\nset style data lines\nplot " + data + "\ne\n"
// 	_ = ioutil.WriteFile("simplified_cost.plot", []byte(gnuplotData), os.ModePerm)
// 	cmd := exec.Command("gnuplot")
// 	cmd.Stdin = strings.NewReader(gnuplotData)
// 	buf, err := cmd.CombinedOutput()
// 	if err != nil {
// 		return "", fmt.Errorf("Error plotting the graph: %s", err)
// 	}
// 	return fmt.Sprintf("\n    %s\n", bytes.TrimSpace(buf)), nil
// }

// // PlotLineraRegressionCost renders the gnuplot graph of the simplified cost function (Θ[0][0] = 0).
// func (ds Dataset) PlotLineraRegressionCost(x1, x2, xRate, y1, y2, yRate float64) (string, error) {
// 	data := "'-'\n"
// 	for i := x1; (xRate > 0 && i < x2) || (xRate < 0 && i > x2); i += xRate {
// 		for j := y1; (yRate > 0 && j < y2) || (yRate < 0 && j > y2); j += yRate {
// 			data += fmt.Sprintf("%f %f %f\n", i, j, LinearRegression{Θ: Vector{{i}, {j}}}.SquaredError(ds))
// 		}
// 	}
// 	gnuplotData := "set terminal dumb\nset hidden3d\nset dgrid3d 50,50 qnorm 2\nset xrange [*:] reverse\nset grid ztics\nset grid ytics\nset grid xtics\nset zrange [0:]\nset style data lines\nsplot " + data + "\ne\n"
// 	_ = ioutil.WriteFile("graph", []byte(gnuplotData), os.ModePerm)
// 	cmd := exec.Command("gnuplot")
// 	cmd.Stdin = strings.NewReader(gnuplotData)
// 	buf, err := cmd.CombinedOutput()
// 	if err != nil {
// 		return "", fmt.Errorf("Error plotting the graph: %s", err)
// 	}
// 	return fmt.Sprintf("\n    %s\n", bytes.TrimSpace(buf)), nil
// }

// LinearRegression is the linar regression algorithm.
//   - Hypothesis: h(x) = Θ[0][0] + Θ[1][0]*x
type LinearRegression struct {
	Θ Vector
}

// Fct implements the hypothesis function.
func (b LinearRegression) Fct(x Vector) float64 {
	return b.Θ.Transpose().MulV(x).Sum()
}

// // Plot returns a gnuplot formatted data list.
// func (b LinearRegression) Plot(dataset Dataset) (string, error) {
// 	data := "'-' title \"dataset\"\n"
// 	for _, entry := range dataset {
// 		data += fmt.Sprintf("%f %f\n", entry.X, entry.Y)
// 	}
// 	data += "e"
// 	data2 := fmt.Sprintf("h(x) = %f + %f * x\nplot h(x)", b.Θ[0][0], b.Θ[1][0])
// 	cmd := exec.Command("gnuplot")
// 	cmd.Stdin = strings.NewReader("set terminal png\nset xrange [0:4]\nset yrange [0:4]\nset multiplot layout 1,2\nset size 0.5,1\nplot " + data + "\nset size 0.5,1\nset style data lines\n" + data2 + "\n")
// 	buf, err := cmd.CombinedOutput()
// 	if err != nil {
// 		return "", fmt.Errorf("Error plotting the graph: %s", err)
// 	}
// 	return string(buf), nil
// }

// SquaredError process the squared error of the given hypothesis
// on the given dataset.
// The squared error equation is:
// $$\frac{1}{2m}\sum_{i=1}^{m} (h(x^{(i)})-y^{(i)})^2 $$
// - 1/(2m) * Sum from i=1 to m of square h(x(i))-y(i).
// TODO: rename CostFunction() ?
func (b LinearRegression) SquaredError(dataset Dataset) float64 {

	// Add x(0) = 1 column to dataset.
	m, n := dataset.X.Dim()
	if n != len(b.Θ) {
		dataset.X = NewMatrix(m, n+1).SetSubMatrix(dataset.X, 0, 1)
		for i := 0; i < len(dataset.X); i++ {
			dataset.X[i][0] = 1
		}
	}
	// Process the sum of square error.
	var sum float64
	for i := 0; i < m; i++ {
		ret := b.Fct(dataset.X[i].ToVector())
		tmp := ret - dataset.Y[i][0]
		sum += tmp * tmp
	}
	// 1/2m * sum.
	return (1 / (2 * float64(m))) * sum
}

// PartialDerivative .
func (b LinearRegression) PartialDerivative(dataset Dataset, j int) float64 {

	// Add x(0) = 1 column to dataset.
	m, n := dataset.X.Dim()
	if n != len(b.Θ) {
		dataset.X = NewMatrix(m, n+1).SetSubMatrix(dataset.X, 0, 1)
		for i := 0; i < len(dataset.X); i++ {
			dataset.X[i][0] = 1
		}
	}

	var sum float64

	for i := 0; i < m; i++ {
		ret := b.Fct(dataset.X[i].ToVector())
		tmp := ret - dataset.Y[i][0]
		sum += tmp * dataset.X[i][j]
	}
	// 1/2 * sum.
	return (1 / float64(m)) * sum
}

// GradientDescent .
func (b *LinearRegression) GradientDescent(dataset Dataset, alpha float64, plotData bool) <-chan string {
	ch := make(chan string)
	go func() {
		defer close(ch)

		m, n := dataset.X.Dim()
		if n != len(b.Θ) {
			dataset.X = NewMatrix(m, n+1).SetSubMatrix(dataset.X, 0, 1)
			for i := 0; i < len(dataset.X); i++ {
				dataset.X[i][0] = 1
			}
		}

		for i := 0; i < 1e9; i++ {
			if int(b.SquaredError(dataset)*1e20) == 0 {
				println("----> converged in ", i, "steps")
				return
			}
			tmp := make([]float64, len(b.Θ))
			for j := 0; j < len(b.Θ); j++ {
				tmp[j] = b.Θ[j][0] - alpha*b.PartialDerivative(dataset, j)
			}
			for j, elem := range tmp {
				b.Θ[j][0] = elem
			}
			// if plotData && i%100 == 0 {
			// 	println(b.String())
			// 	p, err := b.Plot(dataset)
			// 	if err != nil {
			// 		log.Printf("fail plot: %s\n", err)
			// 		return
			// 	}
			// 	ch <- p
			// }
		}
		log.Printf("didn't converge in 10^9 iterations")
	}()
	if !plotData {
		<-ch
	}
	return ch
}

func (b LinearRegression) String() string {
	return fmt.Sprintf("Θ[0][0]: %f, Θ[1][0]: %f\n", b.Θ[0][0], b.Θ[1][0])
}
