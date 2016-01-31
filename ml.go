package ml

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

// Minimize .
func (b LinearRegression) Minimize() {

}
