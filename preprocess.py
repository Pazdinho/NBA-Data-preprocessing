import pandas as pd
import os
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Checking ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# Download data if it is unavailable.
if 'nba2k-full.csv' not in os.listdir('../Data'):
    print('Train dataset loading.')
    url = "https://www.dropbox.com/s/wmgqf23ugn9sr3b/nba2k-full.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/nba2k-full.csv', 'wb').write(r.content)
    print('Loaded.')

data_path = "../Data/nba2k-full.csv"



#The goal of this project:  to preprocess the data to prepare it for use in a machine learning model that predicts the salaries of NBA players.

#cleaning data
def clean_data(data_path):
    nba = pd.read_csv(data_path)
    nba['b_day'] = pd.to_datetime(nba['b_day'], format = '%m/%d/%y')
    nba['draft_year'] = pd.to_datetime(nba['draft_year'], format = '%Y')
    nba['team'].fillna('No Team', inplace=True)
    nba['height'] = nba['height'].str[-4:]
    nba['weight'] = nba['weight'].str.split('/').str[1].str.lstrip().str.rstrip(' kg.')
    nba['salary'] = nba['salary'].str.strip('$')
    nba[['height', 'weight', 'salary']] = nba[['height', 'weight', 'salary']].astype('float')
    nba.loc[nba['country'] != 'USA', 'country'] = 'Not-USA'
    nba.loc[nba['draft_round'] == 'Undrafted', 'draft_round'] = "0"
    return nba

#upgrading data and adding new features
def feature_data():
    nba = clean_data(data_path)
    nba.version = nba.version.apply(lambda x: '2020' if x == 'NBA2k20' else '2021')
    nba.version = pd.to_datetime(nba['version'], format = '%Y')
    nba['age'] = (nba.version.dt.year - nba.b_day.dt.year)
    nba['experience'] = (nba.version.dt.year - nba.draft_year.dt.year)
    nba['bmi'] = nba.weight/nba.height**2
    nba.drop(columns=['version', 'b_day', 'draft_year', 'weight', 'height'], inplace = True)

    #removing high cardinality categorical features
    for i in nba.columns.to_list():
        if nba[f"{i}"].dtype == object and nba[f"{i}"].nunique() > 50:
            nba.drop(columns = [f"{i}"],inplace = True)


    return nba

#to drop multicollinear features from the DataFrame
def multicol_data():
    nba = feature_data()
    cor = nba.corr()
    mlist = []
    #checking corelation between features and listing significant pairs
    for i in cor.columns.to_list():
        for j in cor.columns.to_list():
            if ((cor.loc[i,j] > 0.5 and cor.loc[i,j] < 1) or (cor.loc[i,j] < -0.5)):
                mlist.append(i)
                mlist.append(j)
    mlist = list(dict.fromkeys(mlist))

    #dropping multicollinear features
    for i in cor.index.to_list():
        if i in mlist:
            pass
        else:
            cor.drop(index = f"{i}", inplace = True)

    nba.drop(columns = [f"{cor['salary'].idxmin()}"], inplace = True)
    return nba

# transforming data into data sets ready to implement
def transform_data():
    nba = multicol_data()
    #spliting data into numeric and categorical features
    num_nba =nba.select_dtypes('number').drop(columns =['salary'])
    cat_nba =nba.select_dtypes('object')

    #standarazing numerical features using StandardScaler()
    scaler_std = StandardScaler()
    num_nba_standard = scaler_std.fit_transform(num_nba)
    num_nba_standard = pd.DataFrame(num_nba_standard, columns=num_nba.columns)

    #Transforming categorical variables using OneHotEncoder
    enc = OneHotEncoder(sparse = False)
    cat_nba_standard = enc.fit_transform(cat_nba)
    labels = [j for i in enc.categories_ for j in i]
    cat_nba_standard = pd.DataFrame(cat_nba_standard, columns =labels)

    X = pd.concat([num_nba_standard,cat_nba_standard], axis = 1)
    y = nba['salary']
    return X,y


transform_data()