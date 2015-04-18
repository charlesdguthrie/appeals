'''Code for parsing docvec_text shards into a sparse matrix.'''

__author__ = 'Alex Pine'

import collections
import extract_metadata
import numpy as np
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
    '''
    Opens the court opinion n-gram files, and extracts the lines that have a
    case_id in the given list. 
    NOTE: Only case_ids with exactly one opinion are returned.

    Args:
      case_ids: an iterable of case ID strings.
      opinion_data_dir: string. The directory that contains the court opinion 
        n-grams.
      print_stats: If true, prints the number of opinions per case_id.

    Returns:
      A dict that maps a case_id string to a '||' delimited opinion n-gram line.
    '''
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
    '''Testing function'''
    cases_df = extract_metadata.extract_metadata(case_data_dir+'/'+CASE_DATA_FILENAME)
    case_ids = cases_df['caseid']
    extract_docvec_lines(case_ids, opinion_data_dir, TOTAL_NUM_SHARDS, print_stats=True)


# TODO
# Remove the ones that barely ever occur
# do tfidf?
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer
# Then do chi2?
# TODO update comments
def construct_sparse_opinion_matrix(case_ids, opinion_data_dir,
                                    num_opinion_shards=TOTAL_NUM_SHARDS):
    '''
    Builds a CSR sparse matrix containing the n-gram counts from the court opinion
    shard files.
    NOTE: This takes ~6 minutes to run on my macbook.

    Args:
      case_ids: The list of case_ids to extract.
      opinion_data_dir: The directory where the opinion n-gram shard files
        reside.
      num_opinion_shards: The number of opinion shard files to read in. Defaults
        to TOTAL_NUM_SHARDS.

    Returns:
      sparse_feature_matrix: A scipy.sparse.csr_matrix with n-gram counts.
      ordered_case_ids: A list of case ID strings. The index of each case ID
        corresponds to the index of corresponding row of n-grams in
        sparse_feature_matrix.
    '''
    start_time = time.time()
    # Extract only the n-grams with the given case IDs
    case_id_opinion_lines = extract_docvec_lines(case_ids, opinion_data_dir, 
                                                 num_opinion_shards)

    # Constants needed to parse opinion lines
    SEPARATOR = "', '"
    SEPARATOR_ALT = "\", '"
    NGRAM_SEPARATOR = "||"

    # This case_ids will be sorted in the order that the final matrix is sorted.
    ordered_case_ids = sorted(case_id_opinion_lines.keys())
    # List of dictionaries of ngram counts
    rows_ngram_counts = []

    for case_id in ordered_case_ids:
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

    # Make sure the matrics created by this vectorizer are sparse.
    # TODO set sort=False?
    dict_vectorizer = sklearn.feature_extraction.DictVectorizer(sparse=True)
    sparse_feature_matrix = dict_vectorizer.fit_transform(rows_ngram_counts)
    # TODO
    assert sparse_feature_matrix.get_shape()[0] == len(ordered_case_ids)
    print 'shape: ', sparse_feature_matrix.get_shape()
    print 'number of cases', len(case_id_opinion_lines)
    print 'total time:', time.time() - start_time

    return sparse_feature_matrix, ordered_case_ids



# TODO Do best chi2 feature pruning
# http://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection
# TODO test that the ordered_case_ids really do index the right rows.
# TODO normalize the matrix. Do you normalize train and test?
# TODO make sure all test data occurs AFTER training data in time.
# TODO pull in the target variable, and write that and sparse matrix to files using the svmlight format
# instructions:
# http://scikit-learn.org/stable/datasets/#datasets-in-svmlight-libsvm-format
# http://www.reddit.com/r/MachineLearning/comments/l9j0e/ask_rml_i_am_extracting_ngrams_from_my_dataset/c2qzx42
def construct_and_write_sparse_opinion_matrix(case_data_dir, opinion_data_dir):
    sparse_feature_matrix, ordered_case_ids = construct_sparse_opinion_matrix(case_data_dir, opinion_data_dir)
    
    # sklearn.datasets.dump_svmlight_file


if __name__ == '__main__':
#    print_opinion_data_stats('/Users/pinesol/mlcs_data/', 
#                             '/Users/pinesol/mlcs_data/docvec_text')
    # Read in cases, get te list of case IDs
    case_data_dir = '/Users/pinesol/mlcs_data'
    cases_df = extract_metadata.extract_metadata(case_data_dir+'/'+CASE_DATA_FILENAME)
    case_ids = cases_df['caseid']
    num_shards = 100
    construct_sparse_opinion_matrix(case_ids, 
                                    '/Users/pinesol/mlcs_data/docvec_text',
                                    num_opinion_shards=num_shards)

    
