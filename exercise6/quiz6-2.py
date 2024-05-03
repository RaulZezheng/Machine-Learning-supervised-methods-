import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_moons
from sklearn.model_selection import KFold


n_tot = 100
"""
X, y = make_moons(n_tot, shuffle=True, noise=0.2, random_state=42)  # last year was 112
y[y==0] = -1

# standardisation to create better dataset for the exercise
xmean = np.mean(X, axis=0)
xvar = np.var(X, axis=0)
X = (X - xmean[np.newaxis, :]) / np.sqrt(xvar)[np.newaxis, :]

np.save("quiz6data/ex3noisy_X.npy", X)
np.save("quiz6data/ex3noisy_y.npy", y)
"""
X = np.load("quiz6data/ex3noisy_X.npy")
y = np.load("quiz6data/ex3noisy_y.npy")

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, marker='x')
plt.tight_layout()
# plt.savefig("../quiz6/fig/twomoons.png")
# plt.show()

n = 5  # number of outer loops in nested cross-validation
k = 2  # number of inner loops in nested cross-validation

# order lasso params from largest to smallest so that if ties occur CV will select the one with more sparsity
params_lasso = [1, 1e-1, 1e-2, 1e-3, 1e-4]
params_rbf = [1e-3, 1e-2, 1e-1, 1, 10, 100]
params_svm = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]  # smaller C: more regularisation
len_lasso = len(params_lasso)
len_rbf = len(params_rbf)
len_svm = len(params_svm)

n_tot_noisy_feats = 30

"""
np.random.seed(42)
X_noise = 0.5*np.random.randn(X.shape[0], n_tot_noisy_feats)
np.save("quiz6data/ex3noisy_X_noise.npy", X_noise)
"""
X_noise = np.load("quiz6data/ex3noisy_X_noise.npy")


"""
# to perform nested cv, you should consider a structure something like this (pseudocode):

for number_of_noisy_features in all_amounts_of_noisy_features_to_consider:
    # build the data with this amount of noise

    for outer loops:
        # get the training and test data split 
        
        for inner loops:
            # get the training and validation split from training data in outer loop  
            # train and test on those  
            
        # get the best parameters based on the inner loop results 
        # train and test on data split in the outer loop 
        
    # get average performance of the model averaged over the test sets
        
"""
accuracies_noise = np.zeros(n_tot_noisy_feats)
for m in range(1,n_tot_noisy_feats+1):
    X_dataset = np.concatenate((X,X_noise[ : ,0:m]),axis=1)
    y_dataset = y
    nfold_outer = 5
    nfold_inner = 2
    np.random.seed(12345)
    cselection_outer = KFold(n_splits=nfold_outer, random_state=None, shuffle=False)
    accuracies_outerfold = np.zeros(nfold_outer)
    ifold_out = 0

    for index_train, index_test in cselection_outer.split(X):
        X_train = X_dataset[index_train]
        y_train = y_dataset[index_train]
        X_test = X_dataset[index_test]
        y_test = y_dataset[index_test]
        mtrain = X_train.shape[0]
        mtest = X_test.shape[0]
        print('Training size:', mtrain)
        print('Test size:', mtest)
        accuracies = np.zeros(len_lasso)


        for i in range(len_lasso):
            model = Lasso(alpha=params_lasso[i])
            cselection_inner = KFold(n_splits=nfold_inner,random_state=None, shuffle=False)

            ifold_in = 0
            accuracies_infold = np.zeros(nfold_inner)
            for index_in_train,index_in_test in cselection_inner.split(X_train):
                X_in_train = X_train[index_in_train]
                y_in_train = y_train[index_in_train]
                X_in_test = X_train[index_in_test]
                y_in_test = y_train[index_in_test]
                model.fit(X_in_train,y_in_train)
                y_in_pred = model.predict(X_in_test)
                accuracies_infold[ifold_in] = accuracy_score(np.sign(y_in_pred),y_in_test)
                ifold_in +=1
            accuracies[i] = np.mean(accuracies_infold)

        best_accuracy_index = accuracies.tolist().index(np.max(accuracies))
        best_lasso = params_lasso[best_accuracy_index]
        model = Lasso(alpha=best_lasso)
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        accuracies_outerfold[ifold_out] = accuracy_score(np.sign(y_pred),y_test)
        ifold_out += 1

    accuracies_noise[m-1] = np.mean(accuracies_outerfold)
print(accuracies_noise)


