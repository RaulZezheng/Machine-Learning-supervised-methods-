"""
This code is based on the Sklearn example:
https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_alpha.html#sphx-glr-auto-examples-neural-networks-plot-mlp-alpha-py
"""
## #######################################################
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
## score
from sklearn.metrics import f1_score
## data set
from sklearn.datasets import make_circles, make_moons
## learners
from sklearn.neural_network import MLPClassifier
## #######################################################
class AdaBoost_cls:
  """
  Class realizes the AdaBoost training and prediction
  """

  ## ---------------------------  
  def __init__(self, llearners):
    """
    Input: llearners     list of the classifiers,
                         where the elements of the list constructed by the
                         sklearn.pipeline.make_pipeline method
    """

    self.llearners = llearners  ## list of the classifiers
    self.T = len(llearners)     ## number of learners
    self.alpha = None           ## the weights of the learners
    self.D = None               ## the weights of the data examples

  ## ---------------------------  
  def fit(self,X,y):
    """
    Input:  X       2d array of training inputs
            y       vector of training outputs
    Modifies:  self.D, self.alpha
    """

    nlearner = len(self.llearners)
    m,n = X.shape

    self.D = np.ones((self.T+1,m))/m
    self.alpha = np.zeros(nlearner)
    E = np.zeros(nlearner)
    Z = np.zeros(nlearner)

    ## enumerate the learners
    for t in range(self.T):
      ## implement the AdaBoost algorithm on the learners given in self.llearners
        self.llearners[t].fit(X,y)
        y_pre = self.llearners[t].predict(X)
        diff_num = 0
        for i in range(m):
          if y[i] != y_pre[i]:
            E[t] += self.D[t][i] * 1
        #     diff_num += 1
        # E[t] = diff_num / m
        self.alpha[t] = 0.5 * np.log((1-E[t])/E[t])
        Z[t] = 2 * np.power((E[t] * (1-E[t])),0.5)
        for i in range(m):
          self.D[t+1][i] = self.D[t][i] * np.exp(-self.alpha[t] * y[i] * y_pre[i]) / Z[t]
    print('Alphas:',self.alpha)
    print('E:', E)
    return
  ## ---------------------------
  def predict(self,Xtest):
    """
    Input: Xtest      2d array of test inputs
    Output ypredict   vector of predicted labels
    """

    ## compute the ensemble prediction
    ##  based on the weights computed in the training
    m, n = Xtest.shape
    ypred = np.zeros(m)
    for t in range(self.T):
      ypred += self.alpha[t] * self.llearners[t].predict(Xtest)

    ## The prediction is real valued, convert it into -1,+1!
    for i in range(m):
      if ypred[i] > 0:
        ypred[i] = 1
      elif ypred[i] < 0:
        ypred[i] =-1
    return(ypred)

## #######################################################
def main():

  rng =np.random.seed(1234)

  ## the data sets
  data_names = ["Circles", "Moons"]

  ## generate the data examples
  datasets = [make_circles(n_samples= 1000, noise=0.3, factor=0.5, random_state=0),
      make_moons(n_samples = 1000, noise=0.3, random_state=0)]
  ndataset = len(datasets)

  ## set the learners hyperparameters for the list of MLPClassifiers
  nlearner = 10   ## number of learners
  nnode = 10      ## number of nodes in the first learner of the list
  lnodes = [ nnode +i*nnode for i in range(nlearner)]  
  iteration = 1000 

  ## construct the list of learners 
  learner_names = []
  classifiers = []
  for nnode in lnodes:
    classifiers.append(
      make_pipeline(
        StandardScaler(),
        MLPClassifier(
          solver="adam",
          alpha=0.0001,
          random_state=0,
          max_iter=iteration,
          early_stopping=False,
          hidden_layer_sizes=[nnode]),   ## set the layers
      )
    )
    learner_names.append("mlp_node_"+str(nnode))
      
      
  # iterate over datasets
  f1 = []
  for ds_cnt, ds in enumerate(datasets):
    # preprocess datasets, split into training and test part
    X, y = ds

    ## change y values from (0,1) to (-1,+1)
    y = 2*y - 1

    ## select training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)


    # apply the AdaBoost training on the training set and
    # apply the prediction on the test set
    AdaBoost = AdaBoost_cls(classifiers)
    AdaBoost.fit(X_train,y_train)
    y_pred = AdaBoost.predict(X_test)
    
    # compute the F1 score for the two datasets
    f1.append(f1_score(y_test,y_pred))
    # find the learners for both datasets
    #    which have the highest alpha values in the AdaBoost algorithm
    #    the learners are identified with the number nodes!

  ## report the F1 scores for both datasets,
  ##   and the learners with the highest alpha
  print(f1)
    

  print('Bye')

  return

## ####################################################
## ###################################################

main()
  

