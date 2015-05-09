import pandas as pd
import numpy as np
import cPickle as pickle
import time
import sys
import extract_metadata
import join_data as jd
from datetime import datetime

from sklearn.feature_selection import chi2, SelectFpr
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
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

def log_results(locals_dict,log_path,parameters_dict):
    '''
    Save model parameters, accuracy, and training time as a dict and append to pickled log file
    Also pulls parameters from parameters_dict if it's there
    Args:
        locals_dict: dictionary of variables in parent function
        log_path: path to pickled log file
        parameters_dict: dictionary from parameters loaded into load_data
        
    Returns: None
    '''
    log_items = locals_dict.copy()
    if parameters_dict:
        log_items.update(parameters_dict)
    
    #read results log
    try:
        log = pickle.load(open( log_path, "rb" ))
    except:
        print "Log file doesn't exist.  Creating from scratch."
        log = []
    
    #Remove raw data from local_vars_dict
    for key in ('X','y','case_ids','X_train', 'y_train', 'case_ids_train', 
        'X_test', 'y_test', 'case_ids_test',
        'classifier','fitted_model'):
        if key in log_items:
            del log_items[key]

    #TODO: parse fitted_model sub-variables (grid_scores,best_estimator, best_params, best_score)

    #convert start timestamp to datetime
    log_items['start_date_time']=time.ctime(int(log_items['start_time']))
    
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
                          train_pct, reg_min_log10, reg_max_log10, scoring, feature_reduction_type,
                          result_path, description,parameters_dict):
    '''
    Train and score a model
    Args:
        X,y,case_ids - your data.  X is a matrix, y is an array, case_ids is a list
        model: string.  So far either 'baseline', 'logistic', or 'svm'
        train_pct: float between 0 and 1.  Fraction of your subsample to use as training.
        reg_min_log10, reg_max_log10: integers.  The low and high of your grid search for regularization parameter.
        scoring: scoring method
        feature_reduction_type: Either 'chi2', 'l1svc', or None.
        result_path: The path of the filename to which the results should be written.
        description: The experiment description, to be saved in the results file.
    Returns:
        Model score

    Prints out: Model score and confusion matrix
    '''
    start_time = time.time()

    num_cases = X.shape[0]
    num_features = X.shape[1]

    print '\nFitting New Model'
    print 'Model:', model
    print 'Feature Matrix Info:'
    print '  Number of cases', num_cases
    print '  Number of features', num_features
    print 'Training percentage', train_pct
    print 'Scoring used:', scoring
    if reg_min_log10 and reg_max_log10:
        print 'Regularization bounded between 10^(%d) and 10^(%d):' % (
            reg_min_log10, reg_max_log10)

    print 'Splitting data into training and testing...'
    X_train, y_train, case_ids_train, X_test, y_test, case_ids_test = train_test_split(X, y, case_ids, train_pct)

    pipeline_steps = list()
    param_grid = dict()

    if model == 'baseline':
        fitted_model = MajorityClassifier()
        fitted_model.fit(X_train, y_train)
    else:
        # Feature reduction step
        if feature_reduction_type == 'chi2':
            pipeline_steps.append(('feature_reduction', SelectFpr(chi2)))
            param_grid['feature_reduction__alpha'] = [0.4, 0.6, 0.8, 1.0]
        elif feature_reduction_type == 'l1svc':
            # TODO we can modify the stopping point with 'tol'
            pipeline_steps.append(('feature_reduction', LinearSVC(penalty="l1", dual=False)))
            param_grid['feature_reduction__C'] = [10**i for i in range(reg_min_log10, reg_max_log10 + 1)]
        # Model training step
        if model == 'svm':
            # random_state=0 so it always has the same seed so we get deterministic results.
            # Using dual=False b.c. there are lots of features.
            classifier = LinearSVC(penalty='l2', random_state=0, dual=False)
#            param_grid['classifier__penalty'] = ['l1', 'l2'] # TODO this makes it redoc slow
            param_grid['classifier__C'] = [10**i for i in range(reg_min_log10, reg_max_log10 + 1)]
        elif model == 'logistic':
            classifier = LogisticRegression(penalty='l2')
#            param_grid['classifier__penalty'] = ['l1', 'l2'] # TODO this makes it redonc slow
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
        pipeline_steps.append(('classifier', classifier))

        print 'Running Model Pipeline...'
        fitted_model = GridSearchCV(Pipeline(pipeline_steps), scoring=scoring, param_grid=param_grid, 
                                    verbose=1, n_jobs=-1)
        fitted_model.fit(X_train, y_train)

        #Save these to variables so the log can access
        grid_scores=[]
        for grid in fitted_model.grid_scores_:
            grid_scores.append({'parameters': grid.parameters, 'score':grid.mean_validation_score})
        best_estimator = fitted_model.best_estimator_
        best_params = fitted_model.best_params_
        best_score = fitted_model.best_score_

        print 'Fitting Complete!\n'
        print 'best estimator:', fitted_model.best_estimator_
        print 'best params:', fitted_model.best_params_
        print 'best score from that estimator:', fitted_model.best_score_

    total_time = time.time() - start_time
    print 'Total time:', total_time

    train_accuracy = print_accuracy_info(fitted_model.predict(X_train), y_train)
    print "Training Accuracy",train_accuracy
    test_accuracy = print_accuracy_info(fitted_model.predict(X_test), y_test)
    print "Testing Accuracy",test_accuracy

    #log parameters and output and return log
    return log_results(locals(),result_path,parameters_dict)

def main():
    # HPC Params
    #INPUT_DATA_DIR = '/scratch/akp258/ml_input_data'
    #OUTPUT_DATA_DIR = '/scratch/akp258/ml_output_data'
    #RESULT_PATH = '/scratch/akp258/ml_results/model_results.pkl'

    # Alex Data params
    #INPUT_DATA_DIR = '/Users/pinesol/mlcs_data'
    #OUTPUT_DATA_DIR = '/tmp'
    #RESULT_PATH = '/tmp/model_results.pkl'

    # Charlie Params
    INPUT_DATA_DIR = '/Users/205341/Documents/git/machine-learning/appeals/data'
    OUTPUT_DATA_DIR = '/Users/205341/Documents/git/machine-learning/appeals/data'
    RESULT_PATH = '../results/model_results.pkl.' + datetime.now().strftime('%Y%m%d-%H%M%S')

    NUM_OPINION_SHARDS = 10 #1340
    MIN_REQUIRED_COUNT = 1000 #50
    USE_TFIDF = True
    CODED_FEATURE_NAMES = 'geniss'


    # Model params
    TRAIN_PCT = 0.75
    REG_MIN_LOG10 = -2
    REG_MAX_LOG10 = 2
    SCORING = 'accuracy'
    # NOTE: this will be too slow to run locally if feature reduction is enabled
    FEATURE_REDUCTION_TYPE = 'chi2' # TODO try 'chi2' or l1svc

    DESCRIPTION = '.'.join([datetime.now().strftime('%Y%m%d-%H%M%S'), 'min_required_count', str(MIN_REQUIRED_COUNT), 
                            FEATURE_REDUCTION_TYPE if FEATURE_REDUCTION_TYPE else 'all_features', SCORING]) 
    RESULT_PATH = RESULT_PATH + '.' + DESCRIPTION

    print 'Experiment:', DESCRIPTION

    X, case_ids, y,PARAMETERS_DICT = jd.load_data(INPUT_DATA_DIR, OUTPUT_DATA_DIR,
                                  NUM_OPINION_SHARDS, MIN_REQUIRED_COUNT,
                                  USE_TFIDF, CODED_FEATURE_NAMES)

    # TODO write a loop for all of the params
    print 'Training and scoring models...'
    train_and_score_model(X, y, case_ids, 'baseline', train_pct=TRAIN_PCT, 
                          reg_min_log10=None, reg_max_log10=None, scoring=None, feature_reduction_type=FEATURE_REDUCTION_TYPE,
                          result_path=RESULT_PATH, description=DESCRIPTION,
                          parameters_dict = PARAMETERS_DICT)
    train_and_score_model(X, y, case_ids, 'logistic', train_pct=TRAIN_PCT,
                          reg_min_log10=REG_MIN_LOG10, reg_max_log10=REG_MAX_LOG10, scoring=SCORING, feature_reduction_type=FEATURE_REDUCTION_TYPE,
                          result_path=RESULT_PATH, description=DESCRIPTION,
                          parameters_dict = PARAMETERS_DICT)
    train_and_score_model(X, y, case_ids, 'naive_bayes', train_pct=TRAIN_PCT,
                          reg_min_log10=REG_MIN_LOG10, reg_max_log10=REG_MAX_LOG10, scoring=SCORING, feature_reduction_type=FEATURE_REDUCTION_TYPE,
                          result_path=RESULT_PATH, description=DESCRIPTION,
                          parameters_dict = PARAMETERS_DICT)
    train_and_score_model(X, y, case_ids, 'bernoulli_bayes', train_pct=TRAIN_PCT,
                          reg_min_log10=REG_MIN_LOG10, reg_max_log10=REG_MAX_LOG10, scoring=SCORING, feature_reduction_type=FEATURE_REDUCTION_TYPE,
                          result_path=RESULT_PATH, description=DESCRIPTION,
                          parameters_dict = PARAMETERS_DICT)
    train_and_score_model(X, y, case_ids, 'svm', train_pct=TRAIN_PCT,
                          reg_min_log10=REG_MIN_LOG10, reg_max_log10=REG_MAX_LOG10, scoring=SCORING, feature_reduction_type=FEATURE_REDUCTION_TYPE,
                          result_path=RESULT_PATH, description=DESCRIPTION,
                          parameters_dict = PARAMETERS_DICT)

    # TODO P0 Make regularization type something that varies in the pipline

    # TODO P0 Charlie Connect charts to real data

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
    # THis will allows us to try differentsssata sizes, data prep, and hyper params. 
    # Example: http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html#example-model-selection-grid-search-text-feature-extraction-py

    # TODO P2 try other model-specific cross validation techniques
    # http://scikit-learn.org/stable/modules/grid_search.html#model-specific-cross-validation



if __name__ == '__main__':
    main()
