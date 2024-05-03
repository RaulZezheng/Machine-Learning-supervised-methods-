1. Consider the code template loading the breast cancer Wisconsin data from sklearn.
Investigate various feature transformation techniques: centering, standardisation, unit range and normalisation with l2-norm. 
With each feature transformation, consider feature selection with variable ranking, using squared Pearson correlation as the criterion. 
Selecting top-k variables with k ranging from 1 to 30, train the sklearn’s SVC with default settings, and plot the test accuracies w.r.t the number of features selected and used (k).

Note: while feature transformation techniques can be (and often are) applied consecutively, in this question only one feature transformation technique is applied at a time. 
Also recall good machine learning practices about testing: no information about the test set should be leaked to the training stage.
The image of this procedure?
<img width="140" alt="image" src="https://github.com/Raulllllll/Machine-Learning-supervised-methods-/assets/48178795/55f55054-2871-47ce-90a5-94719b3a494c">


