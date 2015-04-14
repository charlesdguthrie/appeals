import pandas as pd

def extract_metadata(inpath):
    '''
    extracts list of judges, their vote valences, and whether they voted with the majority;
    as well as information about their decisions from case

    arguments: path of input file (ex: 'data/merged_caselevel_data.csv')

    returns: dataframe with only the relevant columns

    inputs: inpath 
    
    '''
    df = pd.read_csv(inpath)

    #extract list of judges, their vote valences, and whether they voted with the majority
    assert df.shape[1]>160, "df is wrong shape"
    judges = df.iloc[:,160:229]

    #extract info about the decisions, starting with case ID and majority vote (direct1) 
    decisions = df.loc[:,('caseid','direct1','geniss','casetyp1','treat','majvotes','dissent','concur','casetyp2','direct2')]

    #merge judges and decisions
    df2 = pd.concat([decisions,judges],axis=1)
    return df2

if __name__ == "__main__": 
    df = extract_metadata('../data/merged_caselevel_data.csv')
    outpath = '../data/metadata_compact.csv'
    df.to_csv(outpath,index=False)
    print "File saved to %s" %outpath