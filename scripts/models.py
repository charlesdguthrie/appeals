import pandas as pd
from sklearn.metrics import confusion_matrix

class MajorityClassifier:
    def fit(self,X_train,y_train):
        '''
        Takes training features and labels and outputs majority
        Classifier will predict the training majority every time
        '''
        counts = y_train.value_counts()
        self.majority=counts[counts==max(counts)].index[0]
        
    def predict(self,X_test):
        return [self.majority] * X_test.shape[0]

def evaluate_accuracy(y_test,y_pred):
    cm = confusion_matrix(y_test,y_pred)
    print "\t \t pred"
    print "\t \t 0 \t 1"
    print "true \t 0 \t %d \t %d" %(cm[0][0],cm[0][1])
    print "\t 1 \t %d \t %d" %(cm[1][0],cm[1][1])
    
    return (cm[0][0] + cm[1][1])*1.0/len(y_test)