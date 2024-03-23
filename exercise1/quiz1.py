import numpy as np
from collections import Counter


"""
The data in this exercise is a subset of the much used Breast cancer Wisconsin dataset:
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html 

Documentation states the class distribution as: 212 - Malignant, 357 - Benign
"""

X = np.load("quiz1_X.npy")
y = np.load("quiz1_y.npy")

n = len(y)
print("There are %d samples in total in the dataset" % n)
print("The shape of X:", X.shape)

print("Unique labels in y:", np.unique(y))
print("Counts of labels in y:", Counter(y))

# shuffle the data and randomly divide it to training and testing
# give the random generator a seed to get reproducable results
np.random.seed(0)
order = np.random.permutation(n)
# --------------------------------------------------------------------------------------------------------------------
# Note! Random number generator has changed between numpy versions X and Y! In this course we use up-to-date versions.
# Check that the generated order should be the same as what is given in the materials:
# order = np.load("quiz1_sample_order.npy")
# --------------------------------------------------------------------------------------------------------------------
tr_samples = order[:int(0.5*n)]
tst_samples = order[int(0.5*n):]
print("The data is divided into %d training and %d test samples" % (len(tr_samples), len(tst_samples)))
Xtr = X[tr_samples, :]
Xtst = X[tst_samples, :]
ytr = y[tr_samples]
ytst = y[tst_samples]






# this is a helper function for transforming continuous labels to binary ones
# works with both 0&1 and -1&1 labels
def get_classification_labels_from_regression_predictions(unique_labels, y_pred):
    assert len(unique_labels) == 2  # this function is meant only for binary classification

    meanval = np.mean(unique_labels)

    transformed_predictions = np.zeros(len(y_pred))
    transformed_predictions[y_pred < meanval] = np.min(unique_labels)
    transformed_predictions[y_pred >= meanval] = np.max(unique_labels)

    return transformed_predictions

#-------------------------------------------------------------------------------------------------------
#linear regression
from sklearn import linear_model
from sklearn.metrics import accuracy_score
regr = linear_model.LinearRegression(fit_intercept=False)
model1 = regr.fit(Xtr,ytr)
ypred = model1.predict(Xtst)
# The coefficients
print("Coefficients: \n", regr.coef_)
# The intercept
print("Intercept: \n", regr.intercept_)
ypred_trans = get_classification_labels_from_regression_predictions([0,1], ypred)
# Accuracy Score
print("Linear Regression Accuracy Score: %.4f" % accuracy_score(ypred_trans,ytst))
model1_TP = model1_FP = model1_TN = model1_FN = 0
i = 0
while i<len(ypred_trans):
    if ypred_trans[i] == 1 and ytst[i] == 1 :
        model1_TP += 1
    elif ypred_trans[i] == 0 and ytst[i] == 1 :
        model1_FN += 1
    elif ypred_trans[i] == 0 and ytst[i] == 0 :
        model1_TN += 1
    else:
        model1_FP += 1
    i+=1
print("FN, TN, FP, TP for Linear Regression are ",model1_FN, model1_TN, model1_FP, model1_TP)
model1_Recall=model1_TP/(model1_TP+model1_FN)
print("Recall for Linear Regression = ",model1_Recall)
#---------------------------------------------------------------------------------------------------------

#linear SVM
# For exercise 1, use the sklearn's LinearSVM with these settings:
from sklearn.svm import LinearSVC
svm = LinearSVC(dual=False)
model2 = svm.fit(Xtr,ytr)
ypred2 = model2.predict(Xtst)
# Accuracy Score
print("Linear SVM Accuracy Score: %.4f" % accuracy_score(ypred2,ytst))

model2_TP = model2_FP = model2_TN = model2_FN = 0
i = 0
while i<len(ypred2):
    if ypred2[i] == 1 and ytst[i] == 1 :
        model2_TP += 1
    elif ypred2[i] == 0 and ytst[i] == 1 :
        model2_FN += 1
    elif ypred2[i] == 0 and ytst[i] == 0 :
        model2_TN += 1
    else:
        model2_FP += 1
    i+=1
print("FN, TN, FP, TP for Linear SVM are ",model2_FN, model2_TN, model2_FP, model2_TP)
model2_Recall=model2_TP/(model2_TP+model2_FN)
print("Recall for Linear SVM = ",model2_Recall)
# to train the svm, call svm.fit(...) and to predict, call svm.predict(...) with suitable arguments in place of ...
