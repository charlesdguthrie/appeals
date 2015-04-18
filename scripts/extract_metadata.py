import pandas as pd
from course_utils import trainTest

def extract_metadata(inpath):
    '''
    extracts list of judges, their vote valences, and whether they voted with the majority;
    as well as information about their decisions from case

    args: 
        path of input file (ex: 'data/merged_caselevel_data.csv')

    returns:
        dataframe with only the relevant columns
    '''

    df = pd.read_csv(inpath,low_memory=False)

    #extract list of judges, their vote valences, and whether they voted with the majority
    assert df.shape[1]>160, "df is wrong shape"
    judges = df.iloc[:,160:229]

    #extract caseID and create classification labels
    y = df.loc[:,('caseid','direct1')]
    y['liberal'] = 0
    y.loc[y['direct1']==3,'liberal']=1
    y['conservative'] = 0
    y.loc[y['direct1']==1,'conservative']=1
    y['mixed'] = 0
    y.loc[y['direct1']==2,'mixed']=1
    y['unknown'] = 0
    y.loc[y['direct1']==0,'unknown']=1
    y.head()

    #extract other info about the decisions, starting with case ID and majority vote (direct1) 
    decisions = df.loc[:,('geniss','casetyp1','treat','majvotes','dissent','concur','casetyp2','direct2')]

    #merge labels, judges and decisions
    df2 = pd.concat([y,decisions,judges],axis=1)
    return df2

def label_feature_split(df,labelName, test_pct=0.25, first_feature_column=6):
    '''
    Splits a dataframe into X_train, X_test, y_train, y_test for a given y label
    
    Args:
        df: dataFrame
        labelName: string for the column representing y
        test_pct: fraction of the data set to devote to testing
        first_feature_column: column number of the first feature
        
    Returns: 
        X_train, X_test, y_train, y_test
    '''
    X = df.iloc[:,first_feature_column:]
    y = df.loc[:,labelName]
    
    X_train, X_test = trainTest(X,test_pct)
    y_train, y_test = trainTest(y,test_pct)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__": 
    df = extract_metadata('../data/merged_caselevel_data.csv')
    outpath = '../data/metadata_compact.csv'
    df.to_csv(outpath,index=False)
    print "File saved to %s" %outpath

    X_train, X_test, y_train, y_test=label_feature_split(df,'liberal')
    print ("X_train.shape",X_train.shape)
    print ("y_train.shape",y_train.shape)
    print ("X_test.shape",X_test.shape)
    print ("y_test.shape",y_test.shape)