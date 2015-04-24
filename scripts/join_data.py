"""Code for parsing docvec_text shards into a sparse matrix."""

__author__ = 'Alex Pine'

import collections
import cPickle as pickle
import extract_metadata
import numpy as np
import os
import random
import scipy.sparse
import sklearn.datasets
import sklearn.feature_extraction
import sys
import time

CASE_DATA_FILENAME = 'merged_caselevel_data.csv'
SHARD_FILE_PREFIX = 'part-'
# NOTE: change this to control the number of shard files to read. Max number is 1340.
TOTAL_NUM_SHARDS = 1340

# Stats:
# num unique case_ids:  17178
# Number of cases with 1 opinions: 15466
# Number of cases with 2 opinions: 1630
# Number of cases with 3 opinions: 82


def extract_docvec_lines(case_ids, opinion_data_dir, num_opinion_shards, 
                         print_stats=False):
    """
    Opens the court opinion n-gram files, and extracts the lines that have a
    case_id in the given list. 
    NOTE: Only case_ids with exactly one opinion are returned.

    Args:
      case_ids: an iterable of case ID strings.
      opinion_data_dir: string. The directory that contains the court opinion 
        n-grams.
      num_opinion_shards: The maximum number of opinion shard files to read.
      print_stats: If true, prints the number of opinions per case_id.

    Returns:
      A dict that maps a case_id string to a '||' delimited opinion n-gram line.
    """
    # NOTE: Throwing out cases with more than one opinion.
    case_ids = frozenset(case_ids)
    # convert case_ids to a hashset for fast lookup
    filenames = [opinion_data_dir + '/' + SHARD_FILE_PREFIX + '%05d' % (shardnum) 
                 for shardnum in range(num_opinion_shards)]
    case_id_counts = collections.defaultdict(int)
    case_id_opinion_lines = collections.defaultdict(list)

    for filename in filenames:
        with open(filename, 'rb') as f:
            for line in f:
                line = line.strip()
                case_id_end = line.find('||')
                assert case_id_end != -1
                assert line[:2] == "('" or line[:2] == "(\"", line[:2]
                assert line[-2:] == "')" or line[-2:] == "\")", line[-2:]
                case_id = line[2:case_id_end] # cut out the initial ('
                line = line[2:-2] # cut out the (' and ')
                if case_id in case_ids:
                    case_id_counts[case_id] += 1
                    case_id_opinion_lines[case_id].append(line)
    if print_stats:
        print 'num unique case_ids: ', len(case_id_opinion_lines)
        histogram = {}
        for case_id, count in case_id_counts.iteritems():
            if count not in histogram:
                histogram[count] = 0
            histogram[count] += 1 

        for num_counts, num in histogram.iteritems():
            print "Number of cases with %d opinions: %d" % (num_counts, num)
    # Deleting items with more than one opinion
    for case_id, count in case_id_counts.iteritems():
        if count > 1:
            del case_id_opinion_lines[case_id]
    # The return dict does case_id -> line.
    return {case_id: lines[0] for case_id, lines, in case_id_opinion_lines.iteritems()}


def print_opinion_data_stats(case_data_dir, opinion_data_dir):
    """Testing function"""
    cases_df = extract_metadata.extract_metadata(case_data_dir+'/'+CASE_DATA_FILENAME)
    case_ids = cases_df['caseid']
    extract_docvec_lines(case_ids, opinion_data_dir, TOTAL_NUM_SHARDS, print_stats=True)


# TODO Do experiments to find the best value for this.
def filter_infrequent_ngrams(rows_ngram_counts, case_ids, min_required_count):
    """
    Counts the number of times each n-gram occurs throughout the corpus, and 
    filters out the low-occuring ones.

    Args:
      rows_ngram_counts: The list of n-gram counts for each case.
      case_ids: The case IDs corresponding to each set of n-grams.
      min_required_count: Positive integer. The minimum number of times an
        n-gram must occur in total in order to be left in.

    Returns:
      filtered_rows_ngram_counts: The list of n-gram counts for each case. The
        infrequently occuring n-grams have been filtered out. If all of the 
          n-grams of a document were removed, the whole row was removed.
      filtered_case_ids: The case IDs corresponding to each set of filtered 
        n-grams.
    """
    total_ngram_counts = collections.defaultdict(int)
    for ngram_counts in rows_ngram_counts:
        for ngram_id, count in ngram_counts.iteritems():
            total_ngram_counts[ngram_id] += count
    ngrams_to_keep = set([])
    for ngram_id, count in total_ngram_counts.iteritems():
        if count >= min_required_count:
            ngrams_to_keep.add(ngram_id)

    # TODO This is really slow...
    filtered_rows_ngram_counts = []
    filtered_case_ids = []
    for i, ngram_counts in enumerate(rows_ngram_counts):
        filtered_ngram_counts = {}
        for ngram_id, count in ngram_counts.iteritems():
            if ngram_id in ngrams_to_keep:
                filtered_ngram_counts[ngram_id] = count
        if filtered_ngram_counts:
            filtered_rows_ngram_counts.append(filtered_ngram_counts)
            filtered_case_ids.append(case_ids[i])

    return filtered_rows_ngram_counts, filtered_case_ids


def parse_opinion_shards(case_ids, opinion_data_dir, num_opinion_shards):
    """
    Builds a dictionary containing the n-gram counts from the court opinion
    shard files.
    NOTE: This takes ~10 minutes to run on my macbook on all shards.

    Args:
      case_ids: The list of case_ids to extract.
      opinion_data_dir: The directory where the opinion n-gram shard files
        reside.
      num_opinion_shards: The number of opinion shard files to read in. Defaults
        to TOTAL_NUM_SHARDS.

    Returns:
      rows_ngram_counts: A list of dictionaries containing n-gram counts.
      ordered_case_ids: A list of case ID strings. The index of each case ID
        corresponds to the index of corresponding row of n-grams in
        sparse_feature_matrix.
    """
    # Extract only the n-grams with the given case IDs
    case_id_opinion_lines = extract_docvec_lines(case_ids, opinion_data_dir, 
                                                 num_opinion_shards)

    # Constants needed to parse opinion lines
    SEPARATOR = "', '"
    SEPARATOR_ALT = "\", '"
    NGRAM_SEPARATOR = "||"

    # This case_ids will be sorted in the order that the final matrix is sorted.
    filtered_case_ids = case_id_opinion_lines.keys()
    # List of dictionaries of ngram counts
    rows_ngram_counts = []

    for case_id in filtered_case_ids:
        opinion_line = case_id_opinion_lines[case_id]
        # Each line into metadata portion and ngram portion, separated by either
        # ', ' or ", '
        separator_index = opinion_line.find(SEPARATOR)
        if separator_index == -1:
            separator_index = opinion_line.find(SEPARATOR_ALT)
            assert separator_index != -1, 'Unparsable opinion line. Case ID: %s' % (case_id)
        ngram_line = opinion_line[separator_index+len(SEPARATOR):]
        assert len(ngram_line) > 0 and ngram_line[0] != "'" and ngram_line[-1] != "'", 'bad ngram line at case %s' % (case_id)
        ngrams = ngram_line.split('||')
        ngram_counts = {}
        for ngram in ngrams:
            ngram_id, count = ngram.split(':')
            assert ngram_id != '', 'Bad ngram ID: %s' % ngram_id
            count = int(count)
            assert count > 0, 'Bad ngram count %d' % count
            ngram_counts[ngram_id] = count
        rows_ngram_counts.append(ngram_counts)
    # Testing code
    assert len(rows_ngram_counts) == len(filtered_case_ids)
    for i in random.sample(range(len(rows_ngram_counts)), len(rows_ngram_counts)/100):
        case_id = filtered_case_ids[i]
        opinion_line = case_id_opinion_lines[case_id]
        ngram_counts = rows_ngram_counts[i]
        for ngram_id, count in ngram_counts.iteritems():
            assert opinion_line.find(case_id) != -1
            assert opinion_line.find(ngram_id) != -1
            assert opinion_line.find(str(count)) != -1

    return rows_ngram_counts, filtered_case_ids


def sort_case_lists(cases_df, rows_ngram_counts, case_ids):
    """
    Sorts the ngram_count dictionaries and case_ids by the date that each case 
    occured on, from oldest to newest.

    Args:
      cases_df: The dataframe of cases information. Used to get the case dates.
      rows_ngram_counts: The list of n-gram counts for each case.
      case_ids: The case IDs corresponding to each set of n-grams.

    Returns:
      rows_ngram_counts: The list of n-gram counts for each case, sorted by the
        date of the corresponding case.
      case_ids: The case IDs corresponding to each set of n-grams, sorted by the
        date of the corresponding case.
    """
    # Case ID -> date as integer YYYYMMDD.
    case_id_date_map = {}
    for index, row in cases_df.iterrows():
        date_int = int(str(row['year']) + str(int(float(row['month']))).zfill(2)
                       + str(int(float(row['day']))).zfill(2))
        case_id_date_map[row['caseid']] = date_int
    # This creates a list of (index, case_id) pairs, sorted by the date 
    # corresponding to the case ID.
    sorted_index_case_id_pairs = sorted(enumerate(case_ids), 
                                        key=lambda tup: case_id_date_map[tup[1]])

    # TODO this is not in place and will be memory intensive
    case_ids = [case_ids[i] for i, caseid in sorted_index_case_id_pairs]
    rows_ngram_counts = [rows_ngram_counts[i] for i, caseid in sorted_index_case_id_pairs]
    # testing code
    sorted_dateints = [case_id_date_map[caseid] for i, caseid in sorted_index_case_id_pairs]
    assert sorted_dateints == sorted(sorted_dateints)

    return rows_ngram_counts, case_ids


def create_valences(cases_df, case_ids):
    """Retreives the valences corresponding to case_ids. Recode unknown valences to neutral."""
    UNKNOWN_VALENCE = 0
    NEUTRAL_VALENCE = 2
    valences = []
    for case_id in case_ids:
        valence = int(cases_df[cases_df['caseid'] == case_id]['direct1'].values[0])
        # Replacing unknown valence variables with netural scores.
        if valence == UNKNOWN_VALENCE:
            valence = NEUTRAL_VALENCE
        valences.append(valence)
    return np.array(valences)

# TODO chi2
# http://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection
# TODO LARS path?
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_lars.html
def construct_sparse_opinion_matrix(cases_df, opinion_data_dir,
                                    num_opinion_shards=TOTAL_NUM_SHARDS,
                                    min_required_count=100,
                                    tfidf=True):
    """
    Builds a CSR sparse matrix containing the n-gram counts from the court
    opinion shard files. Also returns the coresponding case_ids and valences.
    The rows of these lists are sorted in order of the dates of the
    corresponding cases, oldest to newest.
    NOTE: This takes ~10 minutes to run with all shards on my macbook.

    Args:
      case_df: A dataframe containing the case variables. Must include caseid, 
        year, month, day, and direct1.
      opinion_data_dir: The directory where the opinion n-gram shard files
        reside.
      num_opinion_shards: The number of opinion shard files to read in. Defaults
        to TOTAL_NUM_SHARDS.
      min_required_count: The minimum number of of times an n-gram must appear
        throughout all documents in order to be included in the data.
      tfidf: Boolean. If set, the returned feature matrix has been normalized
        using TF-IDF.

    Returns:
      sparse_feature_matrix: A scipy.sparse.csr_matrix with n-gram counts.
      ordered_case_ids: A list of case ID strings. The index of each case ID
        corresponds to the index of corresponding row of n-grams in
        sparse_feature_matrix.
      valences: A list of valences as ints, with the unknown valences (0) 
        replaced with the neutral valence (2).
    """
    start_time = time.time()

    case_ids_df = cases_df['caseid']

    rows_ngram_counts, case_ids = parse_opinion_shards(case_ids_df, opinion_data_dir, num_opinion_shards)
    rows_ngram_counts, case_ids = filter_infrequent_ngrams(rows_ngram_counts, case_ids, min_required_count)
    rows_ngram_counts, case_ids = sort_case_lists(cases_df, rows_ngram_counts, case_ids)

    valences = create_valences(cases_df, case_ids)

    # Make sure the matrics created by this vectorizer are sparse.
    # set sort=False so that the ordering of the rows by date is preserved.
    dict_vectorizer = sklearn.feature_extraction.DictVectorizer(sparse=True, sort=False)
    sparse_feature_matrix = dict_vectorizer.fit_transform(rows_ngram_counts)

    if tfidf:
        transformer = sklearn.feature_extraction.text.TfidfTransformer()
        sparse_feature_matrix = transformer.fit_transform(sparse_feature_matrix)

    assert sparse_feature_matrix.get_shape()[0] == len(case_ids)
    assert len(valences) == len(case_ids)

    print 'shape: ', sparse_feature_matrix.get_shape()
    print 'number of cases', len(case_ids)
    print 'total time:', time.time() - start_time

    return sparse_feature_matrix, case_ids, valences


def load_data(matrix_data_filename,
              case_ids_filename,
              cases_df, 
              opinion_data_dir,
              num_opinion_shards=TOTAL_NUM_SHARDS,
              min_required_count=100,
              tfidf=True):
    """
    Looks to see if the file containing the feature matrix and target labels 
    exists, along with the file containing their corresponding case IDs. If so, 
    this loads them into memory and returns them. Otherwise, it calls
    construct_sparse_opinion_matrix to construct the them. Once they have been 
    constructed, it saves them to disk and returns them.

    Args:
      matrix_data_filename: The svmlight file containing the features and target
        variables.
      case_ids_filename: TODO
      case_df: A dataframe containing the case variables. Must include caseid, 
        year, month, day, and direct1.
      opinion_data_dir: The directory where the opinion n-gram shard files
        reside.
      num_opinion_shards: The number of opinion shard files to read in. Defaults
        to TOTAL_NUM_SHARDS.
      min_required_count: The minimum number of of times an n-gram must appear
        throughout all documents in order to be included in the data.
      tfidf: Boolean. If set, the returned feature matrix has been normalized
        using TF-IDF.

    Returns:
      sparse_feature_matrix: A scipy.sparse.csr_matrix with n-gram counts.
      ordered_case_ids: A list of case ID strings. The index of each case ID
        corresponds to the index of corresponding row of n-grams in
        sparse_feature_matrix.
      valences: A list of valences as ints, with the unknown valences (0) 
        replaced with the neutral valence (2).
    """
    if os.path.isfile(matrix_data_filename) and os.path.isfile(case_ids_filename):
        print 'Loading data from', matrix_data_filename, 'and', case_ids_filename
        with open(matrix_data_filename, 'rb') as f:
            sparse_feature_matrix, valences = sklearn.datasets.load_svmlight_file(f)
        with open(case_ids_filename, 'rb') as f:
            case_ids = pickle.load(f)
    else:
        print 'Constructing data from scratch...'
        sparse_feature_matrix, case_ids, valences = construct_sparse_opinion_matrix(
            cases_df, opinion_data_dir, num_opinion_shards=num_opinion_shards,
            min_required_count=min_required_count, tfidf=tfidf)
        # Save results to disk
        sklearn.datasets.dump_svmlight_file(sparse_feature_matrix, valences, matrix_data_filename)
        with open(case_ids_filename, 'wb') as f:
            pickle.dump(case_ids, f)
    return sparse_feature_matrix, case_ids, valences


if __name__ == '__main__':
#    print_opinion_data_stats('/Users/pinesol/mlcs_data/', 
#                             '/Users/pinesol/mlcs_data/docvec_text')
    # Read in cases, get te list of case IDs
    # case_data_dir = '/Users/pinesol/mlcs_data'
    # cases_df = extract_metadata.extract_metadata(case_data_dir+'/'+CASE_DATA_FILENAME)
    # num_shards = 1340
    # load_data('/tmp/feature_matrix.svmlight',
    #           '/tmp/case_ids.p',
    #           cases_df, 
    #           '/Users/pinesol/mlcs_data/docvec_text',
    #           num_opinion_shards=num_shards)

    case_data_dir = '../data'
    cases_df = extract_metadata.extract_metadata(case_data_dir+'/'+CASE_DATA_FILENAME)
    num_shards = 1340
    load_data(case_data_dir+'/feature_matrix_100.svmlight',
              case_data_dir+'/case_ids.p',
              cases_df, 
              case_data_dir+'/docvec_text',
              num_opinion_shards=num_shards,
              min_required_count=100)