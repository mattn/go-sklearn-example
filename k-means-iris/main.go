package main

import (
	"fmt"

	"github.com/pa-m/sklearn/datasets"
	"github.com/pa-m/sklearn/metrics"
	modelselection "github.com/pa-m/sklearn/model_selection"
	"github.com/pa-m/sklearn/neighbors"
)

func main() {
	ds := datasets.LoadIris()

	Xtrain, Xtest, Ytrain, Ytest := modelselection.TrainTestSplit(ds.X, ds.Y, 0.7, 4)
	nbrs := neighbors.NewKNeighborsClassifier(6, "uniform")
	nbrs.Fit(Xtrain, Ytrain)

	result := nbrs.Predict(Xtest, nil)
	fmt.Printf("%.02f%%\n", metrics.AccuracyScore(result, Ytest, true, nil)*100)
}
