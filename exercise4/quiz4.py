## ####################################################
import sys
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import f1_score

## ##############################################################

class svm_primal_cls:
  """
  Class implementing the primal SVM algorithm:
  "Stochastic gradient descent algorithm for soft-margin SVM"
  """

  def __init__(self, C = 1000, eta = 0.1, xlambda = 0.01, nitermax = 10):

    self.C = C        ## Penalty coefficient 
    self.eta = eta    ## stepsize
    self.nitermax = nitermax  ## nunumber of iteration
    self.xlambda = xlambda    ## penalty parameter 1/C, see Slide 17,Lecture 6
    self.w = None             ## weight parameters
    
    return

  ## ---------------------------------------------------------
  def fit(self,X,y):
    """
    Task: to train the support Vector Machine
          by Stochastic gradient descent algorithm
    Input:  X      2d array of input examples in the rows
            y      1d(vector) array of +1,-1 labels
    Modifies: self.w  weight vector         
    """

    m,n = X.shape
    
    self.w = np.zeros(n)
    ## primal training algorithm
    for iter in range(self.nitermax):
      for i in range(m):          ## load the data examples
        kernel = y[i] * np.dot(self.w,X[i])
        if kernel < 1:
          delta_J = -y[i] * X[i] + self.xlambda * self.w
        else:
          delta_J = self.xlambda * self.w
        self.w = self.w - self.eta * delta_J

        ## primal training step
  ## ------------------------------------------
  def predict(self,Xtest):
    """
    Task: to predict the labels for the given examples based on the self.w
    Input:  Xtest    2d array of input examples in the rows
    Output: y    1d array of predicted labels
    """

    ## predictor function
    kernel = np.dot(Xtest,self.w)
    y = np.sign(kernel)
    return(y)

## ##############################################################

def main():

  # load the data
  X, y = load_breast_cancer(return_X_y=True)  ## X input, y output
  print(X.shape, y.shape)
  ## to convert the {0,1} output into {-1,+1}
  y = 2 * y -1

  mdata,ndim=X.shape   ## size of the data 

  ## normalize the input variables
  X /= np.outer(np.ones(mdata),np.max(np.abs(X),0))

  ## fix the random number generator  
  np.random.seed(12345) 
    
  ## number of iteration
  nitermax_primal = 200        
  ## fixed hyper-parameters for the primal algorithm
  eta = 0.1  ## stepsize
    
  ## list of penalty constats used in the cross validation
  lC = [100, 200, 500, 1000, 2000]
  nC = len(lC) 

  ## Nested cross-validation
  nfold_outer = 5   ## number of folds in the outer loop
  nfold_inner = 4   ## number of folds in the inner loop
  
  ## methods applied
  nmethod = 2 ## 0 svm_primal, 1 svc_linear

  ## split the data into 5-folds
  cselection_outer = KFold(n_splits=nfold_outer, random_state=None, \
    shuffle=False)

  ## run the cross-validation
  ## outer loop
  ifold = 0
  f1_score_outer_primal = np.zeros(nfold_outer)
  f1_score_outer_svc = np.zeros(nfold_outer)
  for index_train, index_test in cselection_outer.split(X):
    Xtrain = X[index_train]
    ytrain = y[index_train]
    Xtest = X[index_test]
    ytest = y[index_test]
    mtrain = Xtrain.shape[0]
    mtest = Xtest.shape[0]
    print('Training size:',mtrain)
    print('Test size:',mtest)
    f1_primal = np.zeros(nC)
    f1_svc = np.zeros(nC)


    ## process the hyper parameters
    for iC in range(nC):

      C = lC[iC]  
      
      ## Xtrain, ytrain contains the data to split in the validation

      ## Initialize the learners
      ## xlambda = 1/C, Slide 17, Lecture 6
      csvm_primal = svm_primal_cls(C = C, eta = eta, xlambda = 1/C ,
        nitermax = nitermax_primal) 

      ## sklearn scv method
      svc_lin = SVC(C = C, kernel = 'linear')

      ## split the training into folds
      cselection_inner = KFold(n_splits=nfold_inner, random_state=None,
        shuffle=False)

      ## inner loop
      x_f1 = np.zeros(nfold_inner)
      x_f1_svc = np.zeros(nfold_inner)
      ifold_in = 0
      for index_in_train, index_in_test in cselection_inner.split(Xtrain):

        ## Only the training data is used!!!
        Xtrain_in = Xtrain[index_in_train]
        ytrain_in = ytrain[index_in_train]
        Xtest_in = Xtrain[index_in_test]
        ytest_in = ytrain[index_in_test]
        mtrain_in = Xtrain_in.shape[0]
        mtest_in = Xtest_in.shape[0]
      
        ## stochastic gradient primal descent
        ## training
        csvm_primal.fit(Xtrain_in,ytrain_in)
        ## prediction
        y_pred = csvm_primal.predict(Xtest_in)
        ## compute the F1 score for the primal
        x_f1[ifold_in] = f1_score(ytest_in,y_pred)
        
        ## svc linear
        ## training
        svc_lin.fit(Xtrain_in,ytrain_in)
        ## prediction
        y_pred_svc = svc_lin.predict(Xtest_in)
        ## compute the F1 score for the svc
        x_f1_svc[ifold_in] = f1_score(ytest_in,y_pred_svc)


        ifold_in += 1

      ## end of inner loop

      ## compute the mean F1 score on the validation sets
      f1_primal[iC] = np.mean(x_f1)
      f1_svc[iC] = np.mean(x_f1_svc)
      #print("The average f1: {}, {}".format(f1_primal,f1_svc))
      ## for each learner for a given C value.     

    ## end of the C selection loop

    ## select the best C value with the highest mean F1 score for each method
    C_best_primal = f1_primal.tolist().index(np.max(f1_primal))
    C_best_svc = f1_svc.tolist().index(np.max(f1_svc))
    ## run the methods with those values: C_primal, C_svc
    C_primal = lC[C_best_primal]
    C_svc = lC[C_best_svc]
    ##  initialize the methods with the best C values
    csvm_primal = svm_primal_cls(C = C_primal, eta = eta,
      xlambda = 1/C_primal, nitermax = nitermax_primal)
    
    svc_lin = SVC(C = C_svc, kernel = 'linear')
        
    ## stochastic gradient primal descent
    ## training
    csvm_primal.fit(Xtrain,ytrain)
    ## prediction
    y_pre_primal = csvm_primal.predict(Xtest)
    ## compute the F1 score for the primal
    f1_score_outer_primal[ifold] = f1_score(ytest,y_pre_primal)
    ## svc linear
    ## training
    svc_lin.fit(Xtrain,ytrain)
    ## prediction
    y_pre_svc = svc_lin.predict(Xtest)
    ## compute the F1 score for the svc
    f1_score_outer_svc[ifold] = f1_score(ytest,y_pre_svc)
    ifold += 1

  ## finally compute the mean of the F1 scores on 5 test sets processed
  f1_score_svc = np.mean(f1_score_outer_svc)
  f1_score_primal = np.mean(f1_score_outer_primal)
  print("final f1 score for primal and svc: ",f1_score_primal,f1_score_svc)
  ## in the outer loop for the two methods.     

  ## compute the standard deviations of the F1 scores on 5 test sets processed
  sum_primal = 0
  sum_svc = 0
  for i in f1_score_outer_primal:
    sum_primal += np.square(i-f1_score_primal)
  for j in f1_score_outer_svc:
    sum_svc += np.square(j-f1_score_svc)
  std_dev_primal = np.sqrt(sum_primal/len(f1_score_outer_primal))
  std_dev_svc = np.sqrt(sum_svc/len(f1_score_outer_svc))
  ## in the outer loop for the two methods.     

  ## compute the ration of standard deviations of the two methods
  ratio = std_dev_primal / std_dev_svc
  ## ratio = std of primal / std of the svc.
  print(std_dev_primal,std_dev_svc,ratio)
  
    
  return

## ####################################################
## ###################################################
main()

