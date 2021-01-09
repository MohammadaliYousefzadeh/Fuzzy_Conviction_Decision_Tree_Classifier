<style>
div.container {
  display: inline-block;
}

#left {
  clip-path: inset(66px 700px 30px 10px);
}
</style>

# Fuzzy Conviction Decision Tree Classifier
Contained herewith are codes used in the implementation of the paper [Fuzzy Conviction Score for Discriminating Decision-Tree-Classified Feature Vectors w.r.t. Relative Distances from Decision Boundaries](https://www.researchgate.net/publication/347974343_Fuzzy_Conviction_Score_for_Discriminating_Decision-Tree-Classified_Feature_Vectors_wrt_Relative_Distances_from_Decision_Boundaries_with_Demonstration_on_Benchmark_Breast_Cancer_Data_Sets).
> <b>Abstract -</b> We augment decision tree classification analysis with fuzzy membership functions that quantitatively qualify, at each binary decision boundary, the degree of "conviction" to which each data point (i.e. feature vector) is deemed to be on either side of said decision boundary, the further away from the decision threshold, relative to peers, the higher the fuzzy membership value (i.e. the closer to 1). Our fuzzy "conviction" score is analogous to the measure of "confidence" as per traditional statistical methods, whilst handily accommodates the nonlinear discriminant surface created by a decision tree. Although our method has been successfully deployed in confidential commercial setting, here we demonstrate the concept and computation on the benchmark "Breast Cancer Wisconsin (Original/Diagnostic)" Data Sets archived and made available publicly on the UCI (University of California, Irvine) Machine Learning Repository. In addition, we will as well demonstrate that without introducing any additional learning loops, our fuzzification of decision tree classifier improves the AUC (Area Under the ROC (Receiver Operating Characteristic) Curve) performance over that of the original decision tree classifier, provided the latter is decently capable of discriminating classes within the relevant data set to begin with.

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

### Decision Tree Classifier
Here we depict trained decision tree classifiers as directed graphs, where the size of each arc/node (depicted as arrow) is proportional to the relative number of data points being partitioned by the relevant decision boundary, and each vertex/edge contains decision boundary details. The "blue" nodes correspond to positive predictions, the darker the shade "blue", the higher in-sample positive proportion, hence the higher out-sample positive probability, according to decision tree classification. The "orange" nodes correspond to negative predictions, the darker the shade "orange", the higher in-sample negative proportion, hence the higher out-sample negative probability, according to decision tree classification.
<br/><br/>
<b>Breast Cancer Wisconsin (Original) Data Set</b> Figure 1 reveals 3 decision tree leafs associated with positive (malignant) prediction, the biggest majority as well as highest positive probability of which were identified as having "uniformity_of_cell_size" > 3.5 and "marginal_adhesion" > 1.5. The discriminatory power (figure 2) is very good, without-sample/test AUC of 0.98.
<br/>

<div>
	<p align="center">
		<figure class="container">
			<img src="Fuzzy_Conviction_Decision_Tree_Classifier/results/original/decision_tree_d2_original.png" />
			<figcaption>Figure 1: Decision Tree Classifier - trained on 70% Breast Cancer Wisconsin (Original) Data Set</figcaption>
		</figure>
	</p>
</div>

<div class="container">
	<p align="center">
		<figure>
			<img id="left" src="Fuzzy_Conviction_Decision_Tree_Classifier/results/original/roc_decision_tree_d2_original.png" />
			<figcaption>Figure 2: Decision Tree Classifier’s ROC - tested on 30% Breast Cancer Wisconsin (Original) Data Set</figcaption>
		</figure>
	</p>
</div>

Decision Tree Classifier with Fuzzy Conviction Score Overlay

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