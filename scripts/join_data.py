

import collections
import extract_metadata
import sys

CASE_DATA_FILENAME = 'merged_caselevel_data.csv'
SHARD_FILE_PREFIX = 'part-'
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


# TODO
def merge_data(case_data_dir, opinion_data_dir):
    cases_df = extract_metadata.extract_metadata(case_data_dir+'/'+CASE_DATA_FILENAME)
    caseids = cases_df['caseid']
    caseid_opinion_lines = extract_docvec_lines(caseids, opinion_data_dir)
    # TODO now take these lines, split them by '||', and expand the ngrams into their own columns
    # TODO best way to represent the sparse data?


def print_opinion_data_stats(case_data_dir, opinion_data_dir):
    cases_df = extract_metadata.extract_metadata(case_data_dir+'/'+CASE_DATA_FILENAME)
    caseids = cases_df['caseid']
    extract_docvec_lines(caseids, opinion_data_dir, print_stats=True)


if __name__ == '__main__':
    print_opinion_data_stats('/Users/pinesol/mlcs_data/', 
                             '/Users/pinesol/mlcs_data/docvec_text')
    
