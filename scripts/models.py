import pandas as pd
import numpy as np
import pickler
import time
import sys
import extract_metadata
import join_data as jd

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

    print 'Percent Accuracy: %0.3f%%' % (100 * sklearn.metrics.accuracy_score(y_true, y_pred))


def train_and_score_model(X, y, case_ids, model,
                          train_pct, reg_min_log10, reg_max_log10, scoring, feature_reduction_type):
    '''
    Train and score a model
    Args:
        X,y,case_ids - your data.  X is a matrix, y is an array, case_ids is a list
        model: string.  So far either 'baseline', 'logistic', or 'svm'
        train_pct: float between 0 and 1.  Fraction of your subsample to use as training.
        reg_min_log10, reg_max_log10: integers.  The low and high of your grid search for regularization parameter.
        scoring: scoring method
        feature_reduction_type: Either 'chi2', 'l1svc', or None.

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
    if reg_min_log10 and reg_max_log10:
        print 'Regularization bounded between 10^(%d) and 10^(%d):' % (
            reg_min_log10, reg_max_log10)

    print 'Splitting data into training and testing...'
    X_train, y_train, case_ids_train, X_test, y_test, case_ids_test = train_test_split(X, y, case_ids, train_pct)

    pipeline_steps = list()
    param_grid = dict()

    # Feature reduction step
    if feature_reduction_type == 'chi2':
        pipeline_steps.append(('feature_reduction', SelectFpr(chi2)))
        param_grid['feature_reduction__alpha'] = [0.6, 0.8, 1.0] # TODO check where this ends up, expand around where it lands
    elif feature_reduction_type == 'l1svc':
        # TODO we can modify the stopping point with 'tol'
        pipeline_steps.append(('feature_reduction', LinearSVC(penalty="l1", dual=False)))
        param_grid['feature_reduction__C'] = [10**i for i in range(reg_min_log10, reg_max_log10 + 1)] # TODO check where this ends up, expand around where it lands

    if model == 'baseline':
        # TODO This isn't taking part in feature reduction!
        fitted_model = MajorityClassifier()
        fitted_model.fit(X_train, y_train)
    else:
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

        print 'Fitting Complete!\n'
        print 'best estimator:', fitted_model.best_estimator_
        print 'best params:', fitted_model.best_params_
        print 'best score from that estimator:', fitted_model.best_score_

    print 'Total time:', time.time() - start_time

    print "Training Accuracy"
    train_accuracy = print_accuracy_info(fitted_model.predict(X_train), y_train)
    print "Testing Accuracy"
    test_accuracy = print_accuracy_info(fitted_model.predict(X_test), y_test)


def main():
    # Data params
    #INPUT_DATA_DIR = '/Users/pinesol/mlcs_data'
    #OUTPUT_DATA_DIR = '/tmp'

    INPUT_DATA_DIR = '/Users/205341/Documents/git/machine-learning/appeals/data'
    OUTPUT_DATA_DIR = '/Users/205341/Documents/git/machine-learning/appeals/data'
    NUM_OPINION_SHARDS = 100 #1340
    MIN_REQUIRED_COUNT = 2 #150
    USE_TFIDF = True

    # Model params
    TRAIN_PCT = 0.75
    REG_MIN_LOG10 = -2
    REG_MAX_LOG10 = 2
    SCORING = 'f1_weighted'
    # NOTE: this will be too slow to run locally if feature reduction is enabled
    FEATURE_REDUCTION_TYPE = None # TODO try 'chi2' or l1svc

    X, case_ids, y = jd.load_data(INPUT_DATA_DIR, OUTPUT_DATA_DIR,
                                  NUM_OPINION_SHARDS, MIN_REQUIRED_COUNT, USE_TFIDF)

    # TODO write a loop for all of the params
    print 'Training and scoring models...'
    train_and_score_model(X, y, case_ids, 'baseline', train_pct=TRAIN_PCT, 
                          reg_min_log10=None, reg_max_log10=None, scoring=None, feature_reduction_type=None)
    train_and_score_model(X, y, case_ids, 'logistic', train_pct=TRAIN_PCT,
                          reg_min_log10=REG_MIN_LOG10, reg_max_log10=REG_MAX_LOG10, scoring=SCORING, feature_reduction_type=FEATURE_REDUCTION_TYPE)
    train_and_score_model(X, y, case_ids, 'svm', train_pct=TRAIN_PCT,
                          reg_min_log10=REG_MIN_LOG10, reg_max_log10=REG_MAX_LOG10, scoring=SCORING, feature_reduction_type=FEATURE_REDUCTION_TYPE)
    train_and_score_model(X, y, case_ids, 'naive_bayes', train_pct=TRAIN_PCT,
                          reg_min_log10=REG_MIN_LOG10, reg_max_log10=REG_MAX_LOG10, scoring=SCORING, feature_reduction_type=FEATURE_REDUCTION_TYPE)
    train_and_score_model(X, y, case_ids, 'bernoulli_bayes', train_pct=TRAIN_PCT,
                          reg_min_log10=REG_MIN_LOG10, reg_max_log10=REG_MAX_LOG10, scoring=SCORING, feature_reduction_type=FEATURE_REDUCTION_TYPE)


    # TODO P0 Make linear svc and chi2 options for feature reduction

    # TODO P0 Make the baseline classifer work in the pipeline so it can take part in feature reduction.

    # TODO P0 Make regularization type something that varies in the pipline

    # TODO P0 Create bar charts to visualize which classifiers are better


    # TODO P1.0 Print top 50 most-used N-Grams for each classifier

    # TODO P1.1 Calculate scores other than f1_weighted

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
