import pandas as pd
import numpy as np
import pickler
import time
import sys
import extract_metadata
import join_data as jd

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import sklearn.metrics
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


def optimizeSVM(X_train, y_train, 
                scoring, reg_min_log10, reg_max_log10, regularization_type):
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
    param_grid = {'C': [10**i for i in range(reg_min_log10, reg_max_log10 + 1)]}
    # NOTE: n_jobs=-1 means this will try to run the different folds in parallel
    model_tuning = GridSearchCV(model_to_set, scoring=scoring, param_grid=param_grid, verbose=1, n_jobs=-1)
    
    model_tuning.fit(X_train, y_train)
    print 'Fitting Complete!\n'
    print 'best C param for SVM classifier:', model_tuning.best_params_['C']
    print 'best_score: ', model_tuning.best_score_
        
    return model_tuning.best_estimator_


def optimizeLogistic(X_train, y_train, 
                     scoring, reg_min_log10, reg_max_log10, regularization_type):
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
    param_grid = {'C': [10**i for i in range(reg_min_log10, reg_max_log10 + 1)]}
    model_tuning = GridSearchCV(model_to_set, param_grid=param_grid,
                                scoring=scoring)
    
    model_tuning.fit(X_train, y_train)
    print 'Fitting Complete!\n'
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


def print_accuracy_info(y_pred, y_true):
    '''
    Prints out confusion matrix and returns percent accuracy

    Args:
        y_true: array of y values 
        y_pred: array of predicted y values
    '''
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
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

    print 'Percent Accuracy: %0.3f%%' % (100 * sklearn.metrics.accuracy_score(y_true, y_pred))


def train_and_score_model(X, y, case_ids, model,
                          subsample_pct, train_pct, reg_low, reg_high, scoring, regularization_type):
    '''
    Train and score a model
    Args:
        X,y,case_ids - your data.  X is a matrix, y is an array, case_ids is a list
        model: string.  So far either 'baseline','logistic',or 'svm'
        subsample_pct: float between 0 and 1.  Fraction of your data rows to use.
        train_pct: float between 0 and 1.  Fraction of your subsample to use as training.
        reg_low, reg_high: integers.  The low and high of your grid search for regularization parameter.
        scoring: scoring method
        regularization_type: the type of regularation to use. 'l1', 'l2' or None.

    Returns:
        Model score

    Prints out: Model score and confusion matrix
    '''
    start_time = time.time()

    print '\nFitting New Model'
    print 'Model:', model
    print 'Feature Matrix Info:'
    print '  Number of cases', X.shape[0]
    print '  Number of features', X.shape[1]
    print 'Training percentage', train_pct
    print 'Scoring used:', scoring
    print 'Regularization type:', regularization_type
    if reg_low and reg_high:
        print 'Regularization bounded between 10^(%d) and 10^(%d):' % (
            reg_low, reg_high)

    print ''
    if subsample_pct <= 1.0 and subsample_pct > 0:
        if subsample_pct < 1.0:
            print 'Sampling data down to %i percent...' % (subsample_pct*100)
            X, y, case_ids = subsample(X, y, case_ids, subsample_pct)
    else:
        print "ERROR: Subsample percent must be between 0 and 1"
        return

    print 'Splitting data into training and testing...'
    X_train, y_train, case_ids_train, X_test, y_test, case_ids_test = train_test_split(X, y, case_ids, train_pct)
    
    print 'Fitting model...'
    if model == 'svm':
        fitted_model = optimizeSVM(X_train, y_train, scoring, reg_low, reg_high, regularization_type)
    elif model == 'logistic':
        fitted_model = optimizeLogistic(X_train, y_train, scoring, reg_low, reg_high, regularization_type)
    elif model == 'baseline':
        fitted_model = MajorityClassifier()
        fitted_model.fit(X_train,y_train)
    else:
        print "ERROR: model unknown"
        return

    print 'Total time:', time.time() - start_time

    print "Training Accuracy"
    train_accuracy = print_accuracy_info(fitted_model.predict(X_train), y_train)
    print "Testing Accuracy"
    test_accuracy = print_accuracy_info(fitted_model.predict(X_test), y_test)


def main():
    # Data params
    INPUT_DATA_DIR = '/Users/pinesol/mlcs_data'
    OUTPUT_DATA_DIR = '/tmp'
    NUM_OPINION_SHARDS = 100 #1340
    MIN_REQUIRED_COUNT = 10 #150
    USE_TFIDF = True

    # Model params
    SUBSAMPLE_PCT = 1.0
    TRAIN_PCT = 0.75
    REG_LOW = -2
    REG_HIGH = 2
    SCORING = 'f1_weighted'
    REGULARIZATION_TYPE='l1'

    X, case_ids, y = jd.load_data(INPUT_DATA_DIR, OUTPUT_DATA_DIR,
                                  NUM_OPINION_SHARDS, MIN_REQUIRED_COUNT, USE_TFIDF)

    print 'Training and scoring models...'
    train_and_score_model(X, y, case_ids, 'baseline', subsample_pct=SUBSAMPLE_PCT, train_pct=TRAIN_PCT, 
                          reg_low=None, reg_high=None, scoring=None, regularization_type=None)
    train_and_score_model(X, y, case_ids, 'logistic', subsample_pct=SUBSAMPLE_PCT, train_pct=TRAIN_PCT,
                          reg_low=REG_LOW, reg_high=REG_HIGH, scoring=SCORING, regularization_type=REGULARIZATION_TYPE)
    train_and_score_model(X, y, case_ids, 'svm', subsample_pct=SUBSAMPLE_PCT, train_pct=TRAIN_PCT,
                          reg_low=REG_LOW, reg_high=REG_HIGH, scoring=SCORING, regularization_type=REGULARIZATION_TYPE)


    # TODO P0 Implement Multinomial Naive Bayes
    # http://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes

    # TODO P0 Implement Bernouilli Naive Bayes
    # http://scikit-learn.org/stable/modules/naive_bayes.html#bernoulli-naive-bayes

    # TODO P0 Create bar charts to visualize which classifiers are better


    # TODO P1 Calculate scores other than f1_weighted

    # TODO P1 Print top 50 most-used N-Grams for each classifier

    # TODO P1 Write a second base classifier
    # There should be a one that guesses randomly, but in proportion to the different classes
    # This will probably do better than the other base classifier. We have to beat this!

    # TODO P1 why are we using f1_weighted? We need to cite a justafication
    # We should be able to justify f1 weighted using this doc:
    # http://scikit-learn.org/stable/modules/model_evaluation.html#from-binary-to-multiclass-and-multilabel

    # TODO P1 Try LogisticRegressionCV, which uses a 'regularization path', so it's faster than grid search
    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV


    # TODO P2 Visualize the confustion matrix
    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#example-model-selection-plot-confusion-matrix-py

    # TODO P2 Look into using pipeline system for setting up experiments
    # THis will allows us to try different data sizes, data prep, and hyper params. 
    # Example: http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html#example-model-selection-grid-search-text-feature-extraction-py

    # TODO P2 try other model-specific cross validation techniques
    # http://scikit-learn.org/stable/modules/grid_search.html#model-specific-cross-validation



if __name__ == '__main__':
    main()
