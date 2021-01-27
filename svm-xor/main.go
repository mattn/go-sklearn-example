package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"

	"github.com/pa-m/sklearn/preprocessing"
	"github.com/pa-m/sklearn/svm"
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
	hit := 0
	rows := result.RawMatrix().Rows
	for i := 0; i < rows; i++ {
		fmt.Println(result.At(i, 0), Y.At(i, 0))
		if result.At(i, 0) == Y.At(i, 0) {
			hit++
		}
	}
	fmt.Printf("%.02f%%\n", float64(hit)/float64(rows)*100)
}
