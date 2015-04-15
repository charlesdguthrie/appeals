'''Code for parsing docvec_text shards into a sparse matrix.'''

__author__ = 'Alex Pine'

import collections
import extract_metadata
import numpy as np
import scipy.sparse
import sys
import time

CASE_DATA_FILENAME = 'merged_caselevel_data.csv'
SHARD_FILE_PREFIX = 'part-'
# NOTE: change this to control the number of shard files to read. Max number is 1340.
NUM_SHARDS = 1340

# Stats:
# num unique caseids:  17178
# Number of cases with 1 opinions: 15466
# Number of cases with 2 opinions: 1630
# Number of cases with 3 opinions: 82

def extract_docvec_lines(caseids, opinion_data_dir, print_stats=False):
    '''
    Opens the court opinion n-gram files, and extracts the lines that have a
    caseid in the given list. 
    NOTE: Only caseids with exactly one opinion are returned.

    Args:
      caseids: an iterable of case ID strings.
      opinion_data_dir: string. The directory that contains the court opinion 
        n-grams.
      print_stats: If true, prints the number of opinions per caseid.

    Returns:
      A dict that maps a caseid string to a '||' delimited opinion n-gram line.
    '''
    # NOTE: Throwing out cases with more than one opinion.
    caseids = frozenset(caseids)
    # convert caseids to a hashset for fast lookup
    filenames = [opinion_data_dir + '/' + SHARD_FILE_PREFIX + '%05d' % (shardnum) 
                 for shardnum in range(NUM_SHARDS)]
    caseid_counts = collections.defaultdict(int)
    caseid_opinion_lines = collections.defaultdict(list)

    for filename in filenames:
        with open(filename, 'rb') as f:
            for line in f:
                line = line.strip()
                caseid_end = line.find('||')
                assert caseid_end != -1
                assert line[:2] == "('" or line[:2] == "(\"", line[:2]
                assert line[-2:] == "')" or line[-2:] == "\")", line[-2:]
                caseid = line[2:caseid_end] # cut out the initial ('
                line = line[2:-2] # cut out the (' and ')
                if caseid in caseids:
                    caseid_counts[caseid] += 1
                    caseid_opinion_lines[caseid].append(line)
    if print_stats:
        print 'num unique caseids: ', len(caseid_opinion_lines)
        histogram = {}
        for caseid, count in caseid_counts.iteritems():
            if count not in histogram:
                histogram[count] = 0
            histogram[count] += 1 

        for num_counts, num in histogram.iteritems():
            print "Number of cases with %d opinions: %d" % (num_counts, num)
    # Deleting items with more than one opinion
    for caseid, count in caseid_counts.iteritems():
        if count > 1:
            del caseid_opinion_lines[caseid]
    # The return dict does caseid -> line.
    return {caseid: lines[0] for caseid, lines, in caseid_opinion_lines.iteritems()}


def construct_sparse_opinion_matrix(case_data_dir, opinion_data_dir):
    '''
    Builds a CSR sparse matrix containing the n-gram counts from the court opinion
    shard files.
    NOTE: This takes ~6 minutes to run on my macbook.

    Args:
      case_data_dir: The directory where the case data file resides.
      opinion_data_dir :The directory where the opinion n-gram shard files reside.

    Returns:
      sparse_feature_matrix: A scipy.sparse.csr_matrix with n-gram counts.
      ordered_caseids: A list of case ID strings. The index of each case ID
        corresponds to the index of corresponding row of n-grams in sparse_feature_matrix.
    '''
    start_time = time.time()
    # Read in cases, get te list of case IDs
    cases_df = extract_metadata.extract_metadata(case_data_dir+'/'+CASE_DATA_FILENAME)
    caseids = cases_df['caseid']
    # Extract only the n-grams with the given case IDs
    caseid_opinion_lines = extract_docvec_lines(caseids, opinion_data_dir)

    # Constants needed to parse opinion lines
    SEPARATOR = "', '"
    SEPARATOR_ALT = "\", '"
    NGRAM_SEPARATOR = "||"

    # This caseids will be sorted in the order that the final matrix is sorted.
    ordered_caseids = []

    # Incrementally constructing csr sparse matrix using these instructions:
    # http://www.reddit.com/r/MachineLearning/comments/l9j0e/ask_rml_i_am_extracting_ngrams_from_my_dataset/c2qzx42
    values = []
    indices = []
    indptr = [0]
    for caseid, opinion_line in caseid_opinion_lines.iteritems():
        # Each line into metadata portion and ngram portion, separated by either
        # ', ' or ", '
        separator_index = opinion_line.find(SEPARATOR)
        if separator_index == -1:
            separator_index = opinion_line.find(SEPARATOR_ALT)
            assert separator_index != -1, 'Unparsable opinion line. Case ID: %s' % (caseid)
        ngram_line = opinion_line[separator_index+len(SEPARATOR):]
        assert len(ngram_line) > 0 and ngram_line[0] != "'" and ngram_line[-1] != "'", 'bad ngram line at case %s' % (caseid)
        ngrams = ngram_line.split('||')
        for ngram in ngrams:
            ngram_id, count = ngram.split(':')
            assert ngram_id != '', 'Bad ngram ID: %s' % ngram_id
            count = int(count)
            assert count > 0, 'Bad ngram count %d' % count
            ordered_caseids.append(caseid)
            indices.append(ngram_id)
            values.append(count)
            
        # Update row markers for sparse matrix
        indptr.append(indptr[-1] + len(ngrams)) 
    
    sparse_feature_matrix = scipy.sparse.csr_matrix((values, indices, indptr), dtype=np.int)

    print 'sparse matrix constructed!'
    print 'shape: ', sparse_feature_matrix.get_shape()
    print 'number of cases', len(caseid_opinion_lines)
    print 'total time:', time.time() - start_time

    return sparse_feature_matrix, ordered_caseids


def print_opinion_data_stats(case_data_dir, opinion_data_dir):
    cases_df = extract_metadata.extract_metadata(case_data_dir+'/'+CASE_DATA_FILENAME)
    caseids = cases_df['caseid']
    extract_docvec_lines(caseids, opinion_data_dir, print_stats=True)


if __name__ == '__main__':
#    print_opinion_data_stats('/Users/pinesol/mlcs_data/', 
#                             '/Users/pinesol/mlcs_data/docvec_text')
    construct_sparse_opinion_matrix('/Users/pinesol/mlcs_data/', 
                                    '/Users/pinesol/mlcs_data/docvec_text')

    
