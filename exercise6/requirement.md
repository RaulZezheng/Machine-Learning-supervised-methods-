1. Consider the code template loading the breast cancer Wisconsin data from sklearn.
Investigate various feature transformation techniques: centering, standardisation, unit range and normalisation with l2-norm. 
With each feature transformation, consider feature selection with variable ranking, using squared Pearson correlation as the criterion. 
Selecting top-k variables with k ranging from 1 to 30, train the sklearn’s SVC with default settings, and plot the test accuracies w.r.t the number of features selected and used (k).

Note: while feature transformation techniques can be (and often are) applied consecutively, in this question only one feature transformation technique is applied at a time. 
Also recall good machine learning practices about testing: no information about the test set should be leaked to the training stage.
The image of this procedure?

<img width="140" alt="image" src="https://github.com/Raulllllll/Machine-Learning-supervised-methods-/assets/48178795/55f55054-2871-47ce-90a5-94719b3a494c">
line1-centering
line2-standardization
line3-normalization
line4-no feature transformation
line5-unit range

2. Investigate the effect of adding the noisy features (as given in the data file) to the original data with support vector machine using RBF kernel, and Lasso imported in the template.
For all amounts [0, 30] of added noisy features, perform nested cross-validation to select the optimal parameters from the lists of parameters given in template (for all other parameters use the defaults).
For nested crossvalidation, use n = 5 outer loops, and k = 2 inner loops (see "Nested cross-validation" from lecture 4 to recall 1the procedure); do not shuffle the data here but consider them in order as illustrated there.
Use accuracy as the metric. If ties occur during the parameter selection, select the parameters with lowest index.
What is the lowest amount of noisy features added to the original data, when the mean Lasso accuracy on test sets is better than SVM?

-8

3. With this data, train various multi-class classifiers from sklearn:
• Linear SVM models: one-vs-one, one-vs-all, and Crammer-Singer multiclass model. Note: SVC internally trains multi-class one-vs-one style, irrespective of the decision function shape parameter. LinearSVC implements OVA and native multiclass versions.
• Additional inherently multi-class models: decision tree, random forest, multi-layer perceptron, logistic regression.
For all methods, perform 5-fold cross-validation over the parameters given in the code template.
Do not shuffle the data but consider the samples in order as illustrated there. Use accuracy as the metric.
If ties occur during the CV, select the parameters with lowest index. For other parameters, use defaults except for those that are given in the template.
Which image of the decision boundaries corresponds to which multi-class approach after the final training stage?
<img width="77" alt="image" src="https://github.com/Raulllllll/Machine-Learning-supervised-methods-/assets/48178795/d1d32579-6c48-4ac9-8832-aeddedb785ad">
SVC-CS
<img width="71" alt="image" src="https://github.com/Raulllllll/Machine-Learning-supervised-methods-/assets/48178795/eeef8900-2173-4252-ae98-db409dfaf9e3">
LR
<img width="74" alt="image" src="https://github.com/Raulllllll/Machine-Learning-supervised-methods-/assets/48178795/52a1ac97-0e66-48a6-bdfe-433be5e08597">
Random Forest
<img width="71" alt="image" src="https://github.com/Raulllllll/Machine-Learning-supervised-methods-/assets/48178795/65ece1bf-0ff5-4b7d-90de-5b12ac10b3d2">
SVC-OVR
<img width="65" alt="image" src="https://github.com/Raulllllll/Machine-Learning-supervised-methods-/assets/48178795/a907cc62-e818-4fc1-b725-abcec97650bc">
SVC-OVO
<img width="68" alt="image" src="https://github.com/Raulllllll/Machine-Learning-supervised-methods-/assets/48178795/069b8f37-03b5-42be-8472-ed8653c9a1a8">
Decision Tree
<img width="65" alt="image" src="https://github.com/Raulllllll/Machine-Learning-supervised-methods-/assets/48178795/f9eaff93-fc00-400b-9dd0-0db3adfd305e">
MLP

What is the order of SVM accuracies on test set? (A > B denotes here that accuracy obtained with A is higher than accuracy with B)
- OVO > MC > OVA









