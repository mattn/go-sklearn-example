package main

import (
	"fmt"

	"github.com/pa-m/sklearn/metrics"
	"github.com/pa-m/sklearn/preprocessing"
	"github.com/pa-m/sklearn/svm"
	"gonum.org/v1/gonum/mat"
)

func main() {
	X := mat.NewDense(4, 2, []float64{
		0, 0,
		0, 1,
		1, 0,
		1, 1,
	})
	Y := mat.NewDense(4, 1, []float64{
		0,
		1,
		1,
		0,
	})
	yscaler := preprocessing.NewMinMaxScaler([]float64{-1, 1})
	X1 := X
	Y1, _ := yscaler.FitTransform(Y, nil)

	clf := svm.NewSVC()
	clf.Fit(X1, Y1)

	result := clf.Predict(X1, nil)
	fmt.Printf("%.02f%%\n", metrics.AccuracyScore(result, Y, true, nil)*100)
}
