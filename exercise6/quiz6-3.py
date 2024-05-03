import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm

from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


"""
X, y = make_blobs(n_samples=200, centers=6, cluster_std=[1, 0.4, 0.8, 0.2, 1.4, 0.9], random_state=2)

np.random.seed(42)
order = np.random.permutation(len(y))
tr = order[:100]
tst = order[100:]

Xtr = X[tr, :]
Xtst = X[tst, :]
ytr = y[tr]
ytst = y[tst]

np.save("quiz6data/ex4_Xtr.npy", Xtr)
np.save("quiz6data/ex4_Xtst.npy", Xtst)
np.save("quiz6data/ex4_ytr.npy", ytr)
np.save("quiz6data/ex4_ytst.npy", ytst)
"""
Xtr = np.load("quiz6data/ex4_Xtr.npy")
Xtst = np.load("quiz6data/ex4_Xtst.npy")
ytr = np.load("quiz6data/ex4_ytr.npy")
ytst = np.load("quiz6data/ex4_ytst.npy")

# plt.figure()
# plt.scatter(Xtr[:, 0], Xtr[:, 1], c=ytr, marker='x')
# plt.scatter(Xtst[:, 0], Xtst[:, 1], c=ytst, marker='o')
# plt.tight_layout()


# plt.savefig("../quiz6/fig/multiclass.png")


"""
Use the models with the following settings: 
LinearSVC(max_iter=10000, multi_class="crammer_singer", dual="auto")
SVC(max_iter=10000, kernel="linear", decision_function_shape="ovo")
LinearSVC(max_iter=10000, multi_class="ovr", dual="auto")
DecisionTreeClassifier(random_state=0)
RandomForestClassifier(random_state=0)
MLPClassifier(max_iter=10000, random_state=0)
LogisticRegression(multi_class="multinomial", max_iter=10000)
"""

"""The parameters for the CV are (in the order as the models above):

{"C": [1e-3, 1e-2, 1e-1, 1, 10, 100]}
{"C": [1e-3, 1e-2, 1e-1, 1, 10, 100]}
{"C": [1e-3, 1e-2, 1e-1, 1, 10, 100]}
{"max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9]}  
{"n_estimators": [25, 50, 75, 100]}
{"alpha": [1e-3, 1e-2, 1e-1, 1, 10, 100]}
{"C": [1e-3, 1e-2, 1e-1, 1, 10, 100]}
"""



def plot_decision_boundaries(clf, ax):
    """

    :param clf: the trained sklearn learner object
    :param ax: figure axis
    :return:
    """
    X = Xtr
    y = ytr

    # Define the grid for the decision boundary
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Make predictions on the grid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)

    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

    # Set the limits and remove ticks
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())


def plot_decision_boundary_example():

    """
    Example on how to plot the decision boundary

    Note that the SVM trained here is not one that is asked about in the exercise!

    :return:
    """
    example_svm = SVC()  # svm with RBF kernel, default parameters
    example_svm.fit(Xtr, ytr)

    plt.figure(figsize=(4, 4.2))
    ax = plt.gca()
    # give the classifier and the ax object
    plot_decision_boundaries(example_svm, ax)


C = [1e-3, 1e-2, 1e-1, 1, 10, 100]
max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9]
n_estimators = [25, 50, 75, 100]
alpha = [1e-3, 1e-2, 1e-1, 1, 10, 100]
n_fold = 5
score = np.zeros(len(C))
cselection = KFold(n_splits=n_fold, random_state=None, shuffle=False)

# for i in range(len(C)):
#     accuracies = np.zeros(n_fold)
#     i_fold = 0
#     for index_train, index_test in cselection.split(Xtr):
#         X_train = Xtr[index_train]
#         y_train = ytr[index_train]
#         X_test = Xtst[index_test]
#         y_test = ytst[index_test]
#
#         model = LinearSVC(C=C[i],max_iter=10000, multi_class="ovr", dual="auto")
#         model.fit(X_train,y_train)
#         y_pred = model.predict(X_test)
#         accuracies[i_fold] = accuracy_score(y_pred,y_test)
#         i_fold += 1
#     score[i] = np.mean(accuracies)
# print(score)
# index = np.where(score == np.max(score))[0][0]
# print(index)

model = LinearSVC(C=C[4],max_iter=10000, multi_class="ovr", dual="auto")
model.fit(Xtr, ytr)
y_pred = model.predict(Xtst)
print(accuracy_score(y_pred,ytst))

plt.figure(figsize=(4, 4.2))
ax = plt.gca()
# give the classifier and the ax object
plot_decision_boundaries(model, ax)

# # run the example for plotting the decision boundary by uncommenting the following:
#plot_decision_boundary_example()

# remember to show everything that had been plotted in the end
plt.show()
