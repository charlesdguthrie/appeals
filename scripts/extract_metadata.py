import pandas as pd
df = pd.read_csv('data/merged_caselevel_data.csv')
#df2 = df.loc[:,('caseid','direct1')]

judges = df.iloc[:,160:229]
decisions = df.loc[:,('caseid','direct1','geniss','casetyp1','treat','majvotes','dissent','concur','casetyp2','direct2')]

#merge judges and decisions
df2 = pd.concat([decisions,judges],axis=1)
df2.to_csv('data/metadata_compact.csv',index=False)