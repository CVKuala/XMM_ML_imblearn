# XMM_Classification

# A little about the Data

XMM Newton is a telescope that observes different astronomical bodies given by the table in the data directory. Not all bodies however have been identified and the classes corresponding to these instances were declared by cross matching with other databases. The data that we receive has 7 classes from (0 - 6), however, the data is heavily imbalanced.

Our aim is design a classifier that can accurately predict the class of the astronomical body based soley on the column features. Since the dataser is unbalanced, we use various oversampling techniques to increase the instance count of the deficient classes.

![image](https://user-images.githubusercontent.com/90802245/193864974-345da6d1-8a61-435b-ace0-58bf6e7a9cfe.png)

This is the class distribution for the astronomical bodies given by the table. The deficient classes need to have their instance count increased if we are to develop an effective classifier. We experiment with the following oversampling techniques - 

1) SMOTE (Synthetic Minority Oversampling Technique)
2) ADAYSN (Adaptive Synthetic)
3) SMOTEENN

# SMOTE

SMOTE is an algorithm that creates synthetic data points based on the original data points present in our data. SMOTE is basically an oversampling technique with the main advantage being that you are not generating duplicates, but rather creating new datapoints that are modifications on the existing data points we have. 

A detailed explanation of SMOTE can be found here, however, to explain briefly, SMOTE is an algorithm that takes a few samples from each numerically small class and then for each sample it calculates K nearest neighbors. After this the algorothm calculates the vector from the original point to this chosen point and the vector is multiplied by a number between 0 to 1. The addition of this modified vector to the original point results in a new point of the same class.

# ADAYSN

This is based on the idea of adaptively generating minority data samples according to their distributions using K nearest neighbor. The algorithm adaptively updates the distribution and it uses Euclidean distance for the KNN Algorithm. The key difference between ADASYN and SMOTE is that the former uses a density distribution, as a criterion to automatically decide the number of synthetic samples that must be generated for each minority sample by adaptively changing the weights of the different minority samples to compensate for the skewed distributions. The latter generates the same number of synthetic samples for each original minority sample.

# SMOTEENN

This algorithm is a hybrid one which first perfroms oversampling of the deficient classes, and then perfroms undersampling to set better contours for the classification of the data. It first perfroms SMOTE as described above and then performs ENN (Edited Nearest Neighbour).

ENN works as follows, first, for each data point, the k nearest neighbors are calculated. Then the majority class of the K neighbors are computed and if this majority class is different form the class of the data point, then all the points (including the original data point) are deleted. This algorithm helps set better contours for classification, however there is a problem as well.

In the experiments performed that you will see in the .ipynb notebooks, you will observe that SMOTEENN actually leads to worse results as compared to the Baseline method. This is because SMOTEENN actually produces a lot of sythetic examples through SMOTE and then gets rid of many actual data samples through ENN. Getting rid of such samples actually negatively affects the dataset as crucial data points that would've been used are now eliminated. This is shown in our experiments where SMOTE and Baseline method (where we use no oversampling) actually outperforms SMOTEENN.


# A little about the Classifiers used

We use the following classifiers - 

1) Random Forest Classifier
2) XGBoost Classifier
3) Bayesian Neural Network (5 - Layers)

You can see the Results in the .ipynb files, however, to give a brief summary, Bayesian Neural Networks and XGBoost Classifiers worked the best with SMOTE applied to the data. Surprisingly, from a Baseline perspective, Bayesian Neural Networks actually performed better than Random Forest Classifiers! and the difference in accuracy becomes even more pronounced once we use SMOTE.
SMOTEENN, on the other hand reduces the accuracy, this may be beacause of reasons stated above.


# How to Perform the Experiments

First lets clone the repo
```
git clone https://github.com/bose1998/XMM_Classification.git
```
After this, we need to install all dependencies, however, dont forget to first change the current directory to this one in your local environment.
```
pip install requirements.txt
```
After this, we need to run main.py that contains the code that defines a 5 layer Bayesian NN and trains it, it also correspondingly trains an RF_classifier and a XGB_Classifier. In the arguments, the layer structure of the Bayesian NN is defined with default value [100, 200, 100]. This corresponds to the number of neurons in each layer, you may change it if you like. Additionally, the oversampling_threshold variable defines the minimum number of samples for each class after oversampling. The classes that have a number of instances less tha  this number get oversampled. You may also change this value if you like.
```
python main.py --input_file="path to file" --oversampling="choose between Baseline, SMOTE, SMOTEENN" --iterations=2000
```

Alternatively, you can also check out the .ipynb files present above to check out the experiment results.

# References

https://towardsdatascience.com/smote-fdce2f605729
https://www.datasciencecentral.com/handling-imbalanced-data-sets-in-supervised-learning-using-family/
https://towardsdatascience.com/imbalanced-classification-in-python-smote-enn-method-db5db06b8d50
https://github.com/Harry24k/bayesian-neural-network-pytorch.git
