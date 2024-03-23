# ## ####################################################
# import numpy as np
# import sys
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import KFold
# from sklearn.metrics import roc_auc_score, f1_score
# from sklearn.linear_model import SGDClassifier
# ## ###################################################
# ## ###################################################
#
# iscenario = 0  ## =0 step size enumration Question 3,
#                  ## =1 iteration number enumeration, Question 4
#
#   # load the data
# X, y = load_breast_cancer(return_X_y=True)  ## X input, y output
#
# print(X.shape, y.shape)
# mdata,ndim = X.shape
#
#   ## to convert the {0,1} output into {-1,+1}
# y = 2*y - 1
#
# # max_array = []
#
# #   ## normalize
# # for i in range(30):
# #   max_array.append(np.max(np.abs(X[:,i])))
# #   for j in range(569):
# #     X[j][i] = X[j][i] / max_array[i]
#
#   ## hyperparameters of the learning problem
#
# if iscenario ==0: ## Question 3, step size enumeration
#     ## list of eta, the stepsize or learning rate is enumerated
#   neta = 10   ## number of different step size(0.1-1)
#   eta0 = 0.1  ## first setp size
#   leta = [ eta0*(i+1) for i in range(neta)]  ## list of step sizes
#
#     ## number of iteration
#   iteration =50
#
#
# # elif iscenario == 1: ## Question 4, iteration number enumeration
# #     ## list of iteration numbers
# #   niteration = 10  ## number of different iteration
# #   iteration0 = 10  ## first iteration number
# #   literation = [ iteration0*(i+1) for i in range(niteration)]
# #
# #     ## step size
# #   eta = 0.1
#
#
# nfold = 5         ## number of folds
#
# np.random.seed(12345)
#
#   ## split the data into 5-folds
# cselection = KFold(n_splits=nfold, random_state=None, shuffle=False)
#
#   ## normalization
#   ## scaling the rows by maximum absolute value, L infinite norm of columns
# X /= np.outer(np.ones(mdata),np.max(np.abs(X),0))
#   ## run the cross-validation
#
#
# ## ####################################################
# ## ###################################################
# # if __name__ == "__main__":
# #   if len(sys.argv)==1:
# #     iworkmode=0
# #   elif len(sys.argv)>=2:
# #     iworkmode=eval(sys.argv[1])
#
#
# class logreg_sgd_clf:
#   def __init__(self,iteration,eta):
#     self.iteration = iteration
#     self.eta = eta
#
#   def fit(self,X,y):
#     m,n = X.shape
#     w = np.zeros(30)
#     self.maxmargin = 0
#     for i in range(m):
#       margin = y[i] * np.dot(w,X[i])
#       if margin > self.maxmargin:
#         self.maxmargin = margin
#       phi_logistic = 1 / (1 + np.exp(margin))
#       delta_j = -phi_logistic * y[i] * X[i]
#       w = w - self.eta * delta_j
#     self.w = w
#
#   def predict(self,X):
#     xw = np.dot(X,self.w)
#     y_positive = 1 / (1 + np.exp(-xw))
#     y_negative = 1 / (1 + np.exp(xw))
#     y = 2 * (y_positive > y_negative) -1
#     return y
#
# i_fold = 0
# i = 0
# clf = logreg_sgd_clf(iteration,leta[i])
# score = np.zeros(nfold)
# score2 = np.zeros(nfold)
# for tra_index, val_index in cselection.split(X):
#   clf.fit(X[tra_index],y[tra_index])
#   y_pred = clf.predict(X[val_index])
#   score[i_fold] = roc_auc_score(y[val_index],y_pred)
#   score2[i_fold] = f1_score(y[val_index],y_pred)
#   print(leta[i],i_fold,score[i_fold],clf.maxmargin)
#   i_fold += 1
#
# print(np.mean(score),np.mean(score2))

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score  # evaluation metrics


class logreg_sgd_clf:
    def __init__(self, eta, n_iter_max=10):
        """
        :param eta:  learning rate/learning speed
        :param n_iter_max: max iteration
        """
        self.eta = eta
        self.n_iter_max = n_iter_max
        self.w = None  # learning weights
        self.margin_max = 0  # maximum margin in the training

    def fit(self, X, y):
        """
        Using stochastic gradient algorithm to solve the logistic regression problem
        :param X: 2d array of input examples in the rows
        :param y: 1d vector of +1, -1 labels
        :return:
        """
        m, n = X.shape

        # initialize the weights
        w = np.zeros(n)
        self.margin_max = 0

        # iteration on the full data
        for t in range(self.n_iter_max):
            # iteration on the example
            for i in range(m):
                xymargin = y[i] * np.dot(w, X[i])  # functional margin
                if xymargin > self.margin_max:
                    self.margin_max = xymargin

                # compute the stochastic gradient
                phi_logistic = 1 / (1 + np.exp(-(-xymargin)))
                delta_J = -phi_logistic * y[i] * X[i]
                w = w - self.eta * delta_J

            self.w = w

    def predict(self, X):
        """
        :param X: 2d array of input examples in the rows
        :return: predicted label y
        """

        xw = np.dot(X, self.w)

        # predict +1 prob
        y_positive_prob = 1 / (1 + np.exp(-xw))
        # predict -1 prob
        y_negative_prob = 1 / (1 + np.exp(xw))

        y = 2 * (y_positive_prob > y_negative_prob) - 1

        return y

if __name__ == "__main__":
    # load the data
    X, y = load_breast_cancer(return_X_y=True)  # X input, y output
    # to convert the {0,1} output into {-1,+1}
    y = 2 * y - 1

    # learning parameters
    n_iter_max = 100  # maximum iteration
    eta = 0.1  # learning speed

    n_fold = 5  # number of folds

    # to split the data into 5-folds we need
    cselection = KFold(n_splits=n_fold, random_state=None, shuffle=False)

    scaling = 2
    # 0 no scaling,
    # 1 scaling by row wise L2 norm,
    # 2 scaling the rows bt maximum absolute value, L infinite norm of columns

    if scaling == 2:
        X /= np.outer(np.ones(len(X)), np.max(np.abs(X), 0))

    # construct a learning object
    clf_logistic = logreg_sgd_clf(eta, n_iter_max)

    # initialize the learning results for all folds
    x_f1 = np.zeros(n_fold)
    x_precision = np.zeros(n_fold)
    x_recall = np.zeros(n_fold)
    x_margin = np.zeros(n_fold)
    x_roc = np.zeros((n_fold))

    i_fold = 0

    # cross-validation
    for index_train, index_test in cselection.split(X):
        X_train = X[index_train]
        y_train = y[index_train]
        X_test = X[index_test]
        y_test = y[index_test]
        m_train = X_train.shape[0]
        m_test = X_test.shape[0]

        print('Training size: {}'.format(m_train))
        print('Test size: {}'.format(m_test))

        clf_logistic.fit(X_train, y_train)  # training
        y_pred = clf_logistic.predict(X_test)  # predicting

        x_precision[i_fold] = precision_score(y_test, y_pred)
        x_recall[i_fold] = recall_score(y_test, y_pred)
        x_f1[i_fold] = f1_score(y_test, y_pred)
        x_margin[i_fold] = clf_logistic.margin_max
        x_roc[i_fold] = roc_auc_score(y_test, y_pred)

        print("Fold: {}, f1: {}, precision: {}, recall: {}, roc: {}".format(i_fold, x_f1[i_fold], x_precision[i_fold],
                                                                   x_recall[i_fold], x_roc[i_fold]))
        print("Maximum margin: {}".format(x_margin[i_fold]))

        i_fold += 1

    print("The average f1: {}".format(np.mean(x_f1)))
    print("The average maximum margin: {}".format(np.mean(x_margin)))
    print("The average roc: {}".format(np.mean(x_roc)))




 
