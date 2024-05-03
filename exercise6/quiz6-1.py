import numpy as np
import matplotlib.pyplot as plt
import heapq
import mininet

from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import r_regression

"""
More info about the attributes in the dataset:
https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset
"""
mininet.topo
# the data is the breast cancer wisconsin dataset
X_original = np.load("quiz6data/ex2_X.npy")
y_original = np.load("quiz6data/ex2_y.npy")
tr = np.load("quiz6data/ex2_tr.npy")
tst = np.load("quiz6data/ex2_tst.npy")


def ranking(X_train, y_train, top):
    row, column = X_train.shape
    x_bar = np.zeros(column)
    s = np.zeros(column)
    y_bar = sum(y_train) / row

    for j in range(column):
        x_bar[j] = sum(X_train[ : ,j]) / row   #calculate x_bar

    sum_numerator = np.zeros(column)
    sum_denominator1 = np.zeros(column)
    sum_denominator2 = np.sqrt(sum(np.square(y_train - y_bar)))

    for j in range(column):
        sum_numerator[j] = sum((X_train[ : ,j] - x_bar[j]) * (y_train - y_bar))
        sum_denominator1[j] = np.sqrt(sum(np.square(X_train[ : ,j] - x_bar[j])))
        s[j] = np.square(sum_numerator[j] / sum_denominator1[j] / sum_denominator2)
    # s = r_regression(X_train,y_train)
    # s = s * s
    array_min = heapq.nlargest(top,s.tolist())
    index_min = map(s.tolist().index,array_min)
    h = list(index_min)
    a = np.array(h)
    return a


def transformation(ii,X_train):
    row, column = X_train.shape
    if ii == 0:   #centering
        f_bar = np.zeros(column)
        for j in range(column):
            f_bar[j] = sum(X_train[ : ,j]) / row    #calculate f_bar
            X_train[ : ,j] = X_train[ : ,j] - f_bar[j]
        return X_train

    elif ii == 1:  #standardization
        f_bar = np.zeros(column)
        var_f = np.zeros(column)
        for j in range(column):
            f_bar[j] = sum(X_train[ : ,j]) / row
            var_f[j] = sum(np.square(X_train[ : ,j] - f_bar[j])) /row
            X_train[ : ,j] = (X_train[ : ,j] - f_bar[j]) / np.sqrt(var_f[j])
        return X_train

    elif ii == 2: #unit range
        f_max = np.zeros(column)
        f_min = np.zeros(column)
        for j in range(column):
            f_max[j] = np.max(X_train[ : ,j])
            f_min[j] = np.min(X_train[ : ,j])
            X_train[ : ,j] = (X_train[ : ,j] - f_min[j]) / (f_max[j] - f_min[j])
        return X_train

    elif ii == 3:  #l2 normal
        l2_normal = np.zeros(row)
        for j in range(row):
            l2_normal[j] = np.sqrt(sum(np.square(X_train[j, : ])))
            X_train[j, : ] = X_train[j, : ] / l2_normal[j]
        return X_train

    elif ii == 4:
        return X_train

# ---------------------------------------------------------------------
# colors and line styles used in plots
# (not mandatory to use these ones!)
# pretty colors :)
import matplotlib.cm
cmap = matplotlib.cm.get_cmap('viridis')
colors = []
for val in np.arange(0, 1, 0.23):
    colors.append(cmap(val))

linestyles = ["-", "--", ":", "--", "-"]
# ---------------------------------------------------------------------

# The following code was used to produce the plots
plt.figure(figsize=(5, 5))
ax1 = plt.subplot(111)
for ii in range(5):  # a loop to go through all five feature transformation cases

    X = np.copy(X_original)
    y = np.copy(y_original)

    """
    Here you need to insert the solution to the exercise, that produces the accuracies to be plotted
    """
    X = transformation(ii,X)

    accuracies = []
    for i in range(1,30):
        tr_index = ranking(X[tr],y[tr],i)
        X_train = X[tr, : ][ : ,tr_index]
        y_train = y[tr]
        X_test = X[tst, : ][ : , tr_index]
        y_test = y[tst]
        model = SVC()
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        #z = y_original[tst]
        accuracies.append(accuracy_score(y_test,y_pred))
        #print(accuracies)

    ax1.plot(np.arange(1, X.shape[1]), accuracies,
             label="line " + str(ii + 1), c=colors[ii], linestyle=linestyles[ii], linewidth=2)

ax1.legend(loc='lower right', fontsize=16)
ax1.set_xlabel("#feats", fontsize=18)
ax1.set_ylabel("accuracy", fontsize=18)
ax1.tick_params(axis='both', which='major', labelsize=14)
plt.ylim(0.38,1)
plt.tight_layout()


plt.show()


