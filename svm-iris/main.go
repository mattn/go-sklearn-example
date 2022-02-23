package main

import (
	"fmt"
	"math"
	"time"

	"github.com/pa-m/sklearn/datasets"
	"github.com/pa-m/sklearn/metrics"
	modelselection "github.com/pa-m/sklearn/model_selection"
	"github.com/pa-m/sklearn/preprocessing"
	"github.com/pa-m/sklearn/svm"
	"gonum.org/v1/gonum/mat"
)

type Euclidean struct{}

func NewEuclidean() *Euclidean {
	return &Euclidean{}
}

// InnerProduct computes a Eucledian inner product.
func (e *Euclidean) InnerProduct(vectorX *mat.Dense, vectorY *mat.Dense) float64 {
	subVector := mat.NewDense(1, 1, nil)
	subVector.Reset()
	subVector.MulElem(vectorX, vectorY)
	result := mat.Sum(subVector)

	return result
}

// Distance computes Euclidean distance (also known as L2 distance).
func (e *Euclidean) Distance(vectorX *mat.Dense, vectorY *mat.Dense) float64 {
	subVector := mat.NewDense(1, 1, nil)
	subVector.Reset()
	subVector.Sub(vectorX, vectorY)
	result := e.InnerProduct(subVector, subVector)

	return math.Sqrt(result)
}

// RBFKernel ...
type RBFKernel struct{ gamma float64 }

// Func for RBFKernel
func (kdata RBFKernel) Func(a, b []float64) float64 {
	/*
		L2 := 0.
		for i := range a {
			v := a[i] - b[i]
			L2 += v * v
		}
		return math.Exp(-kdata.gamma * L2)
	*/

	euclidean := NewEuclidean()
	vectorX := mat.NewDense(4, 1, a)
	vectorY := mat.NewDense(4, 1, b)
	distance := euclidean.Distance(vectorX, vectorY)

	result := math.Exp(-kdata.gamma * math.Pow(distance, 2))

	return result
}

func main() {
	ds := datasets.LoadIris()
	X1 := ds.X
	yscaler := preprocessing.NewMinMaxScaler([]float64{0.0, 2.0})
	Y1, _ := yscaler.FitTransform(ds.Y, nil)
	Xtrain, Xtest, Ytrain, Ytest := modelselection.TrainTestSplit(X1, Y1, 0.25, uint64(time.Now().UnixNano()))
	//fmt.Println(Xtrain.Dims())
	//fmt.Println(Ytrain.Dims())

	//clf := svm.NewSVC()
	clf := svm.NewSVR()
	clf.C = 0.1
	//clf.Epsilon = 0.00
	clf.Kernel = "rbf"
	//clf.Kernel = (&RBFKernel{gamma: 0.01}).Func
	clf.MaxIter = 8000
	clf.Epsilon = 0.05
	//clf.Tol = 0.1
	//clf.Gamma = 2
	//clf.Degree = 10
	clf.Fit(Xtrain, Ytrain)

	//BaseLibSVM: BaseLibSVM{C: 1., Epsilon: 0.1, Kernel: "rbf", Degree: 3., Gamma: 0., Coef0: 0., Shrinking: true, Tol: 1e-3, CacheSize: 200},

	_ = Xtest
	_ = Ytest
	result := clf.Predict(Xtest, nil)
	for i, v := range result.RawMatrix().Data {
		result.RawMatrix().Data[i] = float64(int64(v + 0.5))
	}
	fmt.Println(mat.Formatted(Ytest))
	fmt.Println(mat.Formatted(result))
	fmt.Printf("%.02f%%\n", metrics.AccuracyScore(result, Ytest, true, nil)*100)
}
