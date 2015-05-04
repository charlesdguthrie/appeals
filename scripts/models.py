import pandas as pd
import numpy as np
import cPickle as pickle
import time
import sys
import extract_metadata
import join_data as jd

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import sklearn.metrics
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

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


def train_test_split(X,y, ordered_case_ids,pct_train):
    train_rows = int(pct_train*len(y))
    y_train = np.array(y[:train_rows])
    y_test = np.array(y[train_rows:])
    X_train = X[:train_rows]
    X_test = X[train_rows:]
    case_ids_train = ordered_case_ids[:train_rows]
    case_ids_test = ordered_case_ids[train_rows:]
    return X_train,y_train,case_ids_train,X_test,y_test,case_ids_test


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

    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)

    print 'Percent Accuracy: %0.3f%%' % (100 * accuracy)
    return accuracy

def log_results(locals_dict,log_path):
    '''
    Save model parameters, accuracy, and training time as a dict and append to pickled log file
    Args:
        locals_dict: dictionary of variables in parent function
        log_path: path to pickled log file
        
    Returns: None
    '''
    log_items = locals_dict
    
    #read results log
    try:
        log = pickle.load(open( log_path, "rb" ))
    except:
        print "Log file doesn't exist.  Creating from scratch."
        log = []
    
    #Remove raw data from local_vars_dict
    for key in ('X','y','case_ids','X_train', 'y_train', 'case_ids_train', 'X_test', 'y_test', 'case_ids_test'):
        if key in log_items:
            del log_items[key]
    
    #append result
    log.append(log_items)
    
    #write to results log
    pickle.dump( log, open( log_path, "wb" ) )

def log_to_csv(log_path,csv_path):
    '''
    Convert log pickle to a csv.  Dict keys become columns
    Args:
        log_path: path to log file pickle.  Log is saved as an array of dicts
        csv_path: csv save file
        
    Returns:
        df: pandas DataFrame with every dict key as a column.  
    '''
    try:
        log = pickle.load(open( log_path, "rb" ))
    except:
        print "Log file doesn't exist."
        return
    df = pd.DataFrame.from_dict(log)
    
    df.to_csv(csv_path)
    
    return df

def train_and_score_model(X, y, case_ids, model,
                          train_pct, reg_min_log10, reg_max_log10, scoring, regularization_type,
                          result_path):
    '''
    Train and score a model
    Args:
        X,y,case_ids - your data.  X is a matrix, y is an array, case_ids is a list
        model: string.  So far either 'baseline', 'logistic', or 'svm'
        train_pct: float between 0 and 1.  Fraction of your subsample to use as training.
        reg_min_log10, reg_max_log10: integers.  The low and high of your grid search for regularization parameter.
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
    if reg_min_log10 and reg_max_log10:
        print 'Regularization bounded between 10^(%d) and 10^(%d):' % (
            reg_min_log10, reg_max_log10)

    print 'Splitting data into training and testing...'
    X_train, y_train, case_ids_train, X_test, y_test, case_ids_test = train_test_split(X, y, case_ids, train_pct)
    
    print 'Fitting model...'

    if model == 'baseline':
        fitted_model = MajorityClassifier()
        fitted_model.fit(X_train, y_train)
    else:
        param_grid = dict()
        if model == 'svm':
            # random_state=0 so it always has the same seed so we get deterministic results.
            # Using dual=False b.c. there are lots of features.
            classifier = LinearSVC(penalty=regularization_type, random_state=0, dual=False)
            param_grid['classifier__C'] = [10**i for i in range(reg_min_log10, reg_max_log10 + 1)]
        elif model == 'logistic':
            classifier = LogisticRegression(penalty=regularization_type)
            param_grid['classifier__C'] = [10**i for i in range(reg_min_log10, reg_max_log10 + 1)]
        elif model == 'naive_bayes':
            # TODO why fit_prior? why not?
            classifier = MultinomialNB(fit_prior=True)
        elif model == 'bernoulli_bayes':
            # TODO trying aribtrary thresholds for binarize b.c. i'm assuming we've TF-IDF'd the data already.
            param_grid['classifier__binarize'] = [0.01, 0.1]
            classifier = BernoulliNB()
        else:
            print "ERROR: model unknown"
            return
        pipeline_steps = [('classifier', classifier)]

        fitted_model = GridSearchCV(Pipeline(pipeline_steps), scoring=scoring, param_grid=param_grid, 
                                    verbose=1, n_jobs=-1)
        fitted_model.fit(X_train, y_train)

        print 'Fitting Complete!\n'
        print 'best estimator:', fitted_model.best_estimator_
        print 'best params:', fitted_model.best_params_
        print 'best score from that estimator:', fitted_model.best_score_

    total_time = time.time() - start_time
    print 'Total time:', total_time

    print "Training Accuracy"
    train_accuracy = print_accuracy_info(fitted_model.predict(X_train), y_train)
    print "Testing Accuracy"
    test_accuracy = print_accuracy_info(fitted_model.predict(X_test), y_test)

    #log parameters and output
    log_results(locals(),result_path)

def main():
    # Data params
    #INPUT_DATA_DIR = '/Users/pinesol/mlcs_data'
    #OUTPUT_DATA_DIR = '/tmp'

    INPUT_DATA_DIR = '/Users/205341/Documents/git/machine-learning/appeals/data'
    OUTPUT_DATA_DIR = '/Users/205341/Documents/git/machine-learning/appeals/data'

    RESULT_PATH = '../results/model_results.pkl'
    NUM_OPINION_SHARDS = 100 #1340
    MIN_REQUIRED_COUNT = 10 #150
    USE_TFIDF = True

    # Model params
    TRAIN_PCT = 0.75
    REG_MIN_LOG10 = -2
    REG_MAX_LOG10 = 2
    SCORING = 'f1_weighted'
    REGULARIZATION_TYPE='l1'

    X, case_ids, y = jd.load_data(INPUT_DATA_DIR, OUTPUT_DATA_DIR,
                                  NUM_OPINION_SHARDS, MIN_REQUIRED_COUNT, USE_TFIDF)

    print 'Training and scoring models...'
    train_and_score_model(X, y, case_ids, 'baseline', train_pct=TRAIN_PCT, 
                          reg_min_log10=None, reg_max_log10=None, scoring=None, regularization_type=None,
                          result_path=RESULT_PATH)
    train_and_score_model(X, y, case_ids, 'logistic', train_pct=TRAIN_PCT,
                          reg_min_log10=REG_MIN_LOG10, reg_max_log10=REG_MAX_LOG10, scoring=SCORING, regularization_type=REGULARIZATION_TYPE,
                          result_path=RESULT_PATH)
    train_and_score_model(X, y, case_ids, 'svm', train_pct=TRAIN_PCT,
                          reg_min_log10=REG_MIN_LOG10, reg_max_log10=REG_MAX_LOG10, scoring=SCORING, regularization_type=REGULARIZATION_TYPE,
                          result_path=RESULT_PATH)
    train_and_score_model(X, y, case_ids, 'naive_bayes', train_pct=TRAIN_PCT,
                          reg_min_log10=REG_MIN_LOG10, reg_max_log10=REG_MAX_LOG10, scoring=SCORING, regularization_type=REGULARIZATION_TYPE,
                          result_path=RESULT_PATH)
    train_and_score_model(X, y, case_ids, 'bernoulli_bayes', train_pct=TRAIN_PCT,
                          reg_min_log10=REG_MIN_LOG10, reg_max_log10=REG_MAX_LOG10, scoring=SCORING, regularization_type=REGULARIZATION_TYPE,
                          result_path=RESULT_PATH)


    # TODO P0 Create bar charts to visualize which classifiers are better
    # TODO P0.1 Create a function to log results


    # TODO P1.0 Print top 50 most-used N-Grams for each classifier

    # TODO P1.1 Calculate scores other than f1_weighted

    # TODO P1 Write a second base classifier
    # There should be a one that guesses randomly, but in proportion to the different classes
    # This will probably do better than the other base classifier. We have to beat this!

    # TODO P1 why are we using f1_weighted? We need to cite a justification
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
