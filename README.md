# Fuzzy Conviction Decision Tree Classifier
> This project implements a novel technique to improve a decision tree classifier performance using an approach from [Fuzzy Conviction Score for Discriminating Decision-Tree-Classified Feature Vectors w.r.t. Relative Distances from Decision Boundaries](https://www.researchgate.net/publication/347974343_Fuzzy_Conviction_Score_for_Discriminating_Decision-Tree-Classified_Feature_Vectors_wrt_Relative_Distances_from_Decision_Boundaries_with_Demonstration_on_Benchmark_Breast_Cancer_Data_Sets). 

## Usage

### Folder structure of the project

To make the project runable without path error, the structure of the folders must be similar to the listed below. 

```
.
├── datasets
│   └── dataset_link.txt
├── module
│   ├── decision_tree_util.py
│   └── model_pipline_util.py
├── results
│   ├── diagnostic
│   │   └── ......
│   └── original
│       └── ......
├── README.md
├── model_development_breast_cancer_wisconsin.ipynb
└── README.md
```

### Getting Started

Please follow these instructions to set the project up and run on your computer.

1. Download two datasets, `breast-cancer-wisconsin.data` and `wdbc.data` from [Breast Cancer Wisconsin Data Set](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/) and put them into `datasets`.
2. Now run all blocks in `model_development_breast_cancer_wisconsin.ipynb`
3. The results will be stored in `results` seperately with sub folders named `original` for Breast Cancer Wisconsin (Original) Data Set and `diagnostic` for Breast Cancer Wisconsin (Diagnostic) Data Set.

## Results

We used ROC as a measurement to compare the results from Traditional Decision Tree Classifier and Fuzzified Decision Tree Classifier and especially for Fuzzified Decision Tree Classifier, instead of using predicted probabilty alone
as a cut-off threshold to calculate true positive rate and false positive rate, we combined probabilty and conviction rate as a cut-off threshold.

The results show that using conviction rate along with probabilty can increase ROC.

### Breast Cancer Wisconsin (Original) Data Set

<p align="center">
    <img src="results/original/roc_decision_tree_d2_original.png" alt="Image"/>
</p>

### Breast Cancer Wisconsin (Diagnostic) Data Set

<p align="center">
    <img src="results/diagnostic/roc_decision_tree_d2_diagnostic.png" alt="Image"/>
</p>

## Authors

* **Poomjai Nacaskul**
* Kongkan Kalakan - [KongkanKalakan](https://github.com/KongkanKalakan)

## References

1. Fuzzy Conviction Score for Discriminating Decision-Tree-Classified Feature Vectors w.r.t. Relative Distances from Decision Boundaries(https://www.researchgate.net/publication/347974343_Fuzzy_Conviction_Score_for_Discriminating_Decision-Tree-Classified_Feature_Vectors_wrt_Relative_Distances_from_Decision_Boundaries_with_Demonstration_on_Benchmark_Breast_Cancer_Data_Sets)
2. Breast Cancer Wisconsin (Original) Data Set (https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(original))
2. Breast Cancer Wisconsin (Diagnostic) Data Set (https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))