package main

import (
	"fmt"

	"github.com/pa-m/sklearn/datasets"

	modelselection "github.com/pa-m/sklearn/model_selection"
	"github.com/pa-m/sklearn/neighbors"
)

func main() {
	ds := datasets.LoadIris()

	Xtrain, Xtest, Ytrain, Ytest := modelselection.TrainTestSplit(ds.X, ds.Y, 0.7, 4)
	nbrs := neighbors.NewKNeighborsClassifier(6, "uniform")
	nbrs.Fit(Xtrain, Ytrain)

	result := nbrs.Predict(Xtest, nil)
	hit := 0
	rows := result.RawMatrix().Rows
	for i := 0; i < rows; i++ {
		if result.At(i, 0) == Ytest.At(i, 0) {
			hit++
		}
	}
	fmt.Printf("%.02f%%\n", float64(hit)/float64(rows))
}
