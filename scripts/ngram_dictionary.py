
import cPickle as pickle
import numpy as np

def ngram_ids_to_strings(ngram_pickle_filepath, ngram_ids):
    '''Reads in pickle file containing an ngram ID to string dict, a replaces
    the ids in the given list with their corresponding strings.'''
    print 'Reading in ngram dictionary', ngram_pickle_filepath
    with open(ngram_pickle_filepath, 'rb') as f:
        id_ngram_dict = pickle.load(f)
        print 'Loading ngram dictionary with', len(id_ngram_dict), 'keys'
        return [id_ngram_dict.get(id, 'NOT FOUND') for id in ngram_ids]


# note don't know how to map the label index to the actual label
# They're probably in sorted order, though.
def print_important_ngrams(ngrams, coefficients, num_labels):
    # If there are only two labels, there is only one vector of coefs
    if num_labels == 2:
        num_labels = 1
    for i in range(num_labels):
        top10 = np.argsort(coefficients[i])[-10:] # TODO sort by absolute value?
        best_features = [ngrams[j] for j in top10]
        best_features_str = "\n  ".join(best_features)
        print("Label %s:\n  %s" % (i, best_features_str))
