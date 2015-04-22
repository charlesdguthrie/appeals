import pandas as pd
import numpy as np
import pickler
import time
import sys
import extract_metadata
import join_data as jd

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV

#TODO: put this in evaluate_models.py
def train_test_split(X,y, ordered_case_ids,pct_train):
    train_rows = int(pct_train*len(y))
    y_train = np.array(y[:train_rows])
    y_test = np.array(y[train_rows:])
    X_train = X[:train_rows]
    X_test = X[train_rows:]
    case_ids_train = ordered_case_ids[:train_rows]
    case_ids_test = ordered_case_ids[train_rows:]
    return X_train,y_train,case_ids_train,X_test,y_test,case_ids_test

#TODO: create a script to limit the number of rows used

#TODO: run one model with a bunch of options.  
#   options include: regularization, scoring, words to throw out, rows of data
#   this is a model testing framework
#   what if I turn on TF-IDF
#TODO: run all the models with those options

def optimizeSVM(X_train, y_train, reg_min_log10=-2, reg_max_log10=2, regularization_type='l1'):
    '''
    Creates an SVM classifier trained on the given data with an optimized C parameter.
    Args:
      X_train: A dataframe on which to train the features
      y_train: A dataframe on which to evaluate the training data
      reg_min_log10: log base 10 of the low end of the regularization parameter range.  -2 means 10^-2
      reg_max_log10: log base 10 of the high end of the regularization parameter range.  2 means 10^2
    Returns:
      A fitted SVM classifier.
    '''
    
    model_to_set = LinearSVC(penalty=regularization_type,random_state=0, dual=False)
    # consider broadening the param_grid to include different SVM kernels and degrees.  See:
    # http://stackoverflow.com/questions/12632992/gridsearch-for-an-estimator-inside-a-onevsrestclassifier
    param_grid = {'C': [10**i for i in range(-reg_min_log10,reg_max_log10)] + [1e30]}
    model_tuning = GridSearchCV(model_to_set, scoring='f1_weighted',param_grid=param_grid)
    
    model_tuning.fit(X_train, y_train)
    print 'best C param for SVM classifier:', model_tuning.best_params_['C']
    print 'best_score: ', model_tuning.best_score_
        
    return model_tuning.best_estimator_

def optimizeLogistic(X_train, y_train, reg_min_log10=-2, reg_max_log10=2,regularization_type='l1'):
    '''
    Creates a logistic classifier trained on the given data with an optimized C parameter.
    Args:
      X_train: A dataframe on which to train the features
      y_train: A dataframe on which to evaluate the training data
      reg_min_log10: log base 10 of the low end of the regularization parameter range.  -2 means 10^-2
      reg_max_log10: log base 10 of the high end of the regularization parameter range.  2 means 10^2
      TODO: scoring
    Returns:
      A fitted logistic classifier.
    '''
    
    model_to_set = LogisticRegression(penalty=regularization_type)
    param_grid = {'C': [10**i for i in range(-reg_min_log10,reg_max_log10)] + [1e30]}
    model_tuning = GridSearchCV(model_to_set, param_grid=param_grid,
                             scoring='f1_weighted')
    
    model_tuning.fit(X_train, y_train)
    print 'best C param for LR classifier:', model_tuning.best_params_['C']
    print 'best params: ', model_tuning.best_params_
    print 'best_score: ', model_tuning.best_score_
        
    return model_tuning.best_estimator_

class MajorityClassifier:
    def fit(self,X_train,y_train):
        '''
        Takes training features and labels and outputs majority
        Classifier will predict the training majority every time
        '''
        y_df = pd.Series(y_train)
        counts = y_df.value_counts()
        self.majority=counts[counts==max(counts)].index[0]
        
    def predict(self,X_test):
        return [self.majority] * X_test.shape[0]

#TODO: move to evaluate_models.py
def evaluate_accuracy(y_true,y_pred):
    '''
    Prints out confusion matrix and returns percent accuracy

    Args:
        y_true: array of y values 
        y_pred: array of predicted y values

    Returns:
        percent of cases where y_pred == y_true
    '''
    cm = confusion_matrix(y_true,y_pred)
    height = cm.shape[0]
    width = cm.shape[1]
    uniques = np.unique(np.array([y_true,y_pred]))
    header = ""
    for u in uniques:
        header = header + " \t " + str(int(u))

    print "\t \t pred"
    print "true \t" + header
    for i,u in enumerate(uniques):
        row = ""
        for j in range(len(uniques)):
            row = row + " \t " + str(cm[i][j])
        print "\t " + str(int(u)) + row
    
    return (np.diagonal(cm).sum())*1.0/len(y_true)


def main(model,reg_low=0,reg_high=1):

    print "model: ",model
    reg_low = int(reg_low)
    reg_high = int(reg_high)
    start_time = time.time()

    print 'loading data...'
    CASE_DATA_FILENAME = 'merged_caselevel_data.csv'
    case_data_dir = '../data'
    cases_df = extract_metadata.extract_metadata(case_data_dir+'/'+CASE_DATA_FILENAME)
    num_shards = 1340
    X,case_ids,y = jd.load_data('../data/feature_matrix.svmlight',
              '../data/case_ids.p',
              cases_df, 
              '../data/docvec_text',
              num_opinion_shards=num_shards)

    print 'splitting...'
    X_train,y_train,case_ids_train,X_test,y_test,case_ids_test = train_test_split(X,y,case_ids,0.75)
    
    print 'fitting model...'
    if model=='svm':
        fitted_model = optimizeSVM(X_train,y_train,reg_low,reg_high)
    elif model=='logistic':
        fitted_model = optimizeLogistic(X_train,y_train,reg_low,reg_high)
    elif model=='baseline':
        fitted_model = MajorityClassifier()
        fitted_model.fit(X_train,y_train)
    else:
        print "error: model unknown"
        return

    print 'evaluating model...'
    fitted_model.score(X_test,y_test)
    evaluate_accuracy(fitted_model.predict(X_test),y_test)
    print 'total time:', time.time() - start_time

if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2],sys.argv[3])