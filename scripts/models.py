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

def train_test_split(X,y, ordered_case_ids,pct_train):
    train_rows = int(pct_train*len(y))
    y_train = np.array(y[:train_rows])
    y_test = np.array(y[train_rows:])
    X_train = X[:train_rows]
    X_test = X[train_rows:]
    case_ids_train = ordered_case_ids[:train_rows]
    case_ids_test = ordered_case_ids[train_rows:]
    return X_train,y_train,case_ids_train,X_test,y_test,case_ids_test

def subsample(X,y,case_ids, sample_pct):
    '''
    Take a random sub-sample of the rows in your data set.  NOTE: to keep it predictable, this is seeded.
    Args:
        X: feature matrix
        y: label array
        case_ids: list of case ids
        sample_pct: float between 0 and 1, determines what fraction of the data you want to keep. 
        
    Returns X,y, and case_ids, but filtered down to a random sample
    '''
    case_ids2 = np.array(case_ids)
    assert X.shape[0]==len(y), "X and y are not the same length"
    assert len(case_ids2)==len(y), "case_ids and y are not the same length"
    sample_size = int(sample_pct*len(y))
    np.random.seed(10)
    
    #Get random sub-sample of row indexes
    sample_indexes = sorted(np.random.choice(range(len(y)), size=sample_size,replace=False))
    return X[sample_indexes],y[sample_indexes],list(case_ids2[sample_indexes])


#TODO: run one model with a bunch of options.  
#   options include: regularization, scoring, words to throw out, rows of data
#   this is a model testing framework
#   what if I turn on TF-IDF
#TODO: run all the models with those options

def optimizeSVM(X_train, y_train, 
    scoring='f1_weighted',reg_min_log10=-2, reg_max_log10=2, regularization_type='l1'):
    '''
    Creates an SVM classifier trained on the given data with an optimized C parameter.
    Args:
      X_train: A dataframe on which to train the features
      y_train: A dataframe on which to evaluate the training data
      reg_min_log10: log base 10 of the low end of the regularization parameter range.  -2 means 10^-2
      reg_max_log10: log base 10 of the high end of the regularization parameter range.  2 means 10^2
      scoring: 'f1_weighted',
    Returns:
      A fitted SVM classifier.
    '''
    
    model_to_set = LinearSVC(penalty=regularization_type,random_state=0, dual=False)
    # consider broadening the param_grid to include different SVM kernels and degrees.  See:
    # http://stackoverflow.com/questions/12632992/gridsearch-for-an-estimator-inside-a-onevsrestclassifier
    param_grid = {'C': [10**i for i in range(-reg_min_log10,reg_max_log10)] + [1e30]}
    model_tuning = GridSearchCV(model_to_set, scoring=scoring,param_grid=param_grid)
    
    model_tuning.fit(X_train, y_train)
    print 'best C param for SVM classifier:', model_tuning.best_params_['C']
    print 'best_score: ', model_tuning.best_score_
        
    return model_tuning.best_estimator_

def optimizeLogistic(X_train, y_train, 
    scoring='f1_weighted',reg_min_log10=-2, reg_max_log10=2,regularization_type='l1'):
    '''
    Creates a logistic classifier trained on the given data with an optimized C parameter.
    Args:
      X_train: A dataframe on which to train the features
      y_train: A dataframe on which to evaluate the training data
      reg_min_log10: log base 10 of the low end of the regularization parameter range.  -2 means 10^-2
      reg_max_log10: log base 10 of the high end of the regularization parameter range.  2 means 10^2
      scoring: 'f1_weighted', 
    Returns:
      A fitted logistic classifier.
    '''
    
    model_to_set = LogisticRegression(penalty=regularization_type)
    param_grid = {'C': [10**i for i in range(-reg_min_log10,reg_max_log10)] + [1e30]}
    model_tuning = GridSearchCV(model_to_set, param_grid=param_grid,
                             scoring=scoring)
    
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

def evaluate_accuracy(y_pred,y_true):
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

def train_and_score_model(X,y,case_ids,model,
    subsample_pct=1.0, train_pct=0.75, reg_low=-3,reg_high=3, scoring='f1_weighted'):
    '''
    Train and score a model
    Args:
        X,y,case_ids - your data.  X is a matrix, y is an array, case_ids is a list
        model: string.  So far either 'baseline','logistic',or 'svm'
        subsample_pct: float between 0 and 1.  Fraction of your data rows to use.
        train_pct: float between 0 and 1.  Fraction of your subsample to use as training.
        reg_low, reg_high: integers.  The low and high of your grid search for regularization parameter.
        scoring: scoring method

    Returns:
        Model score

    Prints out: Model score and confusion matrix
    '''


    start_time = time.time()

    print 'Sampling data down to %i percent...' % (subsample_pct*100)
    if subsample_pct<=1.0 and subsample_pct>0:
        if subsample_pct<1.0:
            X,y,case_ids = subsample(X,y,case_ids,subsample_pct)
    else:
        print "Subsample percent must be between 0 and 1"
        return

    print 'splitting...'
    X_train,y_train,case_ids_train,X_test,y_test,case_ids_test = train_test_split(X,y,case_ids,0.75)
    
    reg_low = int(reg_low)
    reg_high = int(reg_high)

    print 'fitting chosen model: %s...' % model
    if model=='svm':
        fitted_model = optimizeSVM(X_train,y_train,scoring,reg_low,reg_high)
    elif model=='logistic':
        fitted_model = optimizeLogistic(X_train,y_train,scoring,reg_low,reg_high)
    elif model=='baseline':
        fitted_model = MajorityClassifier()
        fitted_model.fit(X_train,y_train)
    else:
        print "error: model unknown"
        return

    print 'evaluating model...'
    #fitted_model.score(X_test,y_test)
    print 'total time:', time.time() - start_time

    score = evaluate_accuracy(fitted_model.predict(X_test),y_test)
    print "Score = ",score
    return score
    print ""

def main():

    print 'loading data...'
    CASE_DATA_FILENAME = 'merged_caselevel_data.csv'
    case_data_dir = '../data'
    cases_df = extract_metadata.extract_metadata(case_data_dir+'/'+CASE_DATA_FILENAME)
    num_shards = 1340
    X,case_ids,y = jd.load_data('../data/feature_matrix_100.svmlight',
              '../data/case_ids.p',
              cases_df, 
              '../data/docvec_text',
              num_opinion_shards=num_shards)

    print 'training and scoring models...'
    train_and_score_model(X,y,case_ids,'baseline',subsample_pct=1)
    train_and_score_model(X,y,case_ids,'logistic',subsample_pct=0.5)
    train_and_score_model(X,y,case_ids,'svm',subsample_pct=0.1)

if __name__ == '__main__':
    main()