# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 21:49:29 2019

@author: VStore
"""

import numpy as np
import pandas as pd


from Cases import *
from ML import ML
from ML_gallop import ML_gallop
from ML_position import ML_position
from DataHandler import DataHandler
from LogReg import LogReg
from SVC_ import SVC_
from OddsChecker import OddsChecker
from TodayDataHandler import TodayDataHandler
from presentation import show
import webbrowser

"""
class MLobject:

    def __init__(self,case,X_train,X_test,Y_train,Y_test):
        self.case = case

    def fit(self):
    def predict(self,data):
    def get_order(self, prediction):


    # Model save load
    def save_model(self,filenamepath):
    def load_model(self,filenamepath):

    # Model tuning  (xgboost specific)
    def print_attributes(self)

    def save_tuned(self,eval_limit,filenamepath)


 """
def save_recreate_datasets(X_train,X_test,Y_train,Y_test,trainInfo,testInfo):

    case = case1()
    dh = DataHandler(case)



    X_train, X_test, Y_train, Y_test = dh.get_XY_test_train('2019-06-01')
    trainInfo, testInfo = dh.get_raceInfo('2019-06-01')


    X_train.to_parquet('Datasets/X_train.gzip',compression='gzip',index=True)
    X_test.to_parquet('Datasets/X_test.gzip',compression='gzip',index=True)
    Y_train.to_parquet('Datasets/Y_train.gzip',compression='gzip',index=True)
    Y_test.to_parquet('Datasets/Y_test.gzip',compression='gzip',index=True)
    trainInfo.to_parquet('Datasets/trainInfo.gzip',compression='gzip',index=True)
    testInfo.to_parquet('Datasets/testInfo.gzip',compression='gzip',index=True)

def load_datasets():
    X_train = pd.read_parquet('Datasets/X_train.gzip')
    X_test = pd.read_parquet('Datasets/X_test.gzip')
    Y_train = pd.read_parquet('Datasets/Y_train.gzip')
    Y_test = pd.read_parquet('Datasets/Y_test.gzip')
    trainInfo = pd.read_parquet('Datasets/trainInfo.gzip')
    testInfo = pd.read_parquet('Datasets/testInfo.gzip')

    return X_train,X_test,Y_train,Y_test,trainInfo,testInfo

def save_datasets(X_train,X_test,Y_train,Y_test,trainInfo,testInfo):
    X_train.to_parquet('Datasets/X_train.gzip',compression='gzip',index=True)
    X_test.to_parquet('Datasets/X_test.gzip',compression='gzip',index=True)
    Y_train.to_parquet('Datasets/Y_train.gzip',compression='gzip',index=True)
    Y_test.to_parquet('Datasets/Y_test.gzip',compression='gzip',index=True)
    trainInfo.to_parquet('Datasets/trainInfo.gzip',compression='gzip',index=True)
    testInfo.to_parquet('Datasets/testInfo.gzip',compression='gzip',index=True)



class ML_Wrapper:

    def __init__(self):


        self.models = {}

    def insert_model(self,name,model):
        self.models[name] = model

    def fit(self, name):
        self.models[name].fit()

    def predict(self,name,data):
        prediction = self.models[name].predict(data)

        return prediction

    def predict_all(self,data):
        a = 0


    def save_model(self, name, path):
        path = path.format(name)
        self.models[name].save_model(path)

    def read_model(self, name, path):
        path = path.format(name)
        self.models[name].read_model(path)


    def fit_save_all(self, path):

        for name, model in self.models.items():
            model.fit()
            model.save_model(path.format(name))

    def fit_read_all(self, path):
        for name, model in self.models.items():
            model.read_model(path.format(name))



def Regressor():

    X_train,X_test,Y_train,Y_test,trainInfo,testInfo = load_datasets()

    case = case1()
    MLW = ML_Wrapper()

    MLW.insert_model('regressor',ML(case,X_train,X_test,Y_train,Y_test))
    MLW.fit('regressor')
    MLW.save_model('regressor','Models/{}')
    MLW.models['regressor'].print_attributes()

#    for i in range(len(MLW.models['regressor'].model)):
#        pred = MLW.models['regressor'].predict_1_booster(X_test,i)
#        test['position'].fillna(20,inplace=True)
#        print(i,ndcg(test,pred))

    MLW.models['regressor'].save_tuned(0.893,'Models/regressor')
    MLW.read_model('regressor','Models/{}_tuned')
    MLW.models['regressor'].print_attributes()

def Winner():

    X_train,X_test,Y_train,Y_test,trainInfo,testInfo = load_datasets()
    Y_train['position.BOOL.1.0'] = (Y_train['position.BOOL.1.0']!=1).astype(int)
    Y_test['position.BOOL.1.0'] = (Y_test['position.BOOL.1.0']!=1).astype(int)
    case = case1()
    MLW = ML_Wrapper()

    MLW.insert_model('winner',ML_position(case,X_train,X_test,Y_train,Y_test))
    MLW.fit('winner')
    MLW.save_model('winner','Models/{}')
    MLW.models['winner'].print_attributes()

    MLW.models['winner'].save_tuned(0.892,'Models/winner')
    MLW.read_model('winner','Models/{}_tuned')
    MLW.models['winner'].print_attributes()

def Galloper():

    X_train,X_test,Y_train,Y_test,trainInfo,testInfo = load_datasets()
    case = case1()
    MLW = ML_Wrapper()

    MLW.insert_model('galloper',ML_gallop(case,X_train,X_test,Y_train,Y_test))
    MLW.fit('galloper')
    MLW.save_model('galloper','Models/{}')
    MLW.models['galloper'].print_attributes()

    MLW.models['galloper'].save_tuned(0.22,'Models/galloper')
    MLW.read_model('galloper','Models/{}_tuned')
    MLW.models['galloper'].print_attributes()


def get_probable(testInfo):
    P = OddsChecker()

    df = P._odds.copy()
    df = df[df.poolType=='VOI']
    df = df[['poolId','probable','runnerNumber']]
    df = df.drop_duplicates(['poolId','runnerNumber'],keep='last')
    df.probable = df.probable/100.0

    df.columns = ['poolId','probable','startNumber']

    return testInfo.merge(df,how='left',left_on=['poolId','startNumber'], right_on=['poolId','startNumber']).copy()

def get_odd(testInfo):

    P = OddsChecker()

    df = P._odds.copy()
    df = df[df.poolType=='VOI']
    df = df[['poolId','probable','runnerNumber']]
    df = df.drop_duplicates(['poolId','runnerNumber'],keep='last')
    df.probable = df.probable/100.0

    df.columns = ['poolId','probable','startNumber']

    testInfo = testInfo.merge(df,how='left',left_on=['poolId','startNumber'], right_on=['poolId','startNumber'])
    testInfo.loc[:,'probable'] = 1/(1+testInfo.loc[:,'probable'])
#    testInfo.loc[:,'probable'] = 1/(testInfo.loc[:,'probable'])
    return testInfo

def get_predInfo(testInfo,case,X_train,X_test,Y_train,Y_test, MLW):

    MLW.insert_model('regressor',ML(case,X_train,X_test,Y_train,Y_test))
    MLW.read_model('regressor','Models/{}_tuned')
    pred1 = MLW.predict('regressor',X_test)

    MLW.insert_model('galloper',ML_gallop(case,X_train,X_test,Y_train,Y_test))
    MLW.read_model('galloper','Models/{}_tuned')
    pred2 = MLW.predict('galloper',X_test)

    MLW.insert_model('winner',ML_position(case,X_train,X_test,Y_train,Y_test))
    MLW.read_model('winner','Models/{}_tuned')
    pred3 = MLW.predict('winner',X_test)




    # Näitä ei tarvitse kääntää koska ndcg ei lasketa ennusteesta.
    testInfo['pred1'] = pred1    # regressio, pienempi parempi
    testInfo['pred2'] = pred2    # tod näk laukata
    testInfo['pred3'] = pred3    # tod näk voittaa
    testInfo = get_odd(testInfo) # tod näk voittaa



    return testInfo


def analysis(df):
    print('\n\nDifferent datatypes:\n\n')
    c = df.applymap(lambda x : type(x))
    for col in c.columns:

        if len(c[col].value_counts()) > 1:
            print(col)
            print(c[col].value_counts())




    print('\n\nNull values:\n\n')
    c = df.applymap(lambda x : x in ['',' ','NaN', 'nan', 'NAN', 'Nan', None , 'none' , 'None', 'NONE', np.nan])
    for col in c.columns:

        if len(c[col].value_counts()) > 1:
            print(col)
            print(c[col].value_counts())

def cvx_mtr(mu):


    n = mu.shape[1]
    k = mu.shape[0]
    
    mu = mu - mu.min() + 1
    R = (0.5 * (n**2 + n)) / mu[1,:].sum()
    mu = mu *  R

#    assert  np.min(mu) >= 1 and np.max(mu) <= n

    a = np.arange(1,n+1)
    A= block_diag(*[a for i in range(len(a))])

    b = np.ones(n)
    B= block_diag(*[b for i in range(len(b))])

    C = np.hstack([np.eye(n) for a in np.arange(n)])



    X = np.vstack([A,A,B,C])


    m = np.ones(n*(2+k))
    m[0:n*k] = mu.ravel()


    P = cp.Variable((n**2))
    objective = cp.Minimize(cp.sum_squares(X@P-m))
    constraints = [0 <= P, P <= 1]

    prob = cp.Problem(objective, constraints)


    prob.solve(max_iter=100000)

    return P.value.reshape(n,n)


import cvxpy as cp
from scipy.linalg import block_diag


def testcvx():
    case = case1()
    MLW = ML_Wrapper()


    X_train,X_test,Y_train,Y_test,trainInfo,testInfo = load_datasets()
    testInfo = get_predInfo(testInfo,case,X_train,X_test,Y_train,Y_test, MLW)

    testInfo = testInfo[~testInfo.probable.isnull()]


    testInfo =  testInfo.loc[testInfo.disqualificationCode != 'p']

    indices = testInfo.groupby('raceId').indices
    
    raceIds = testInfo.raceId.unique()
    testInfo.probable.replace(np.nan,0,inplace=True)
    testInfo['orderProbable'] = testInfo.groupby('raceId')['probable'].apply(lambda x: (1-x).argsort().argsort()+1)   



    
    
    for column in ['pred1','pred2']:
        testInfo[column+'_order'] = testInfo.groupby('raceId')[column].apply(lambda x: x.argsort().argsort()+1)
        testInfo[column+'_order'] = testInfo.groupby('raceId')[column].apply(lambda x: x.argsort().argsort()+1)
    
    for column in ['pred3','probable']:
        testInfo[column+'_order'] = testInfo.groupby('raceId')[column].apply(lambda x: (1-x).argsort().argsort()+1)
        testInfo[column+'_order'] = testInfo.groupby('raceId')[column].apply(lambda x: (1-x).argsort().argsort()+1)

    cvx_pred = np.zeros(len(testInfo))

#    b = testInfo.probable_order.values
#    c = testInfo.pred1_order.values
#    d = testInfo.pred2_order.values
#    e = testInfo.pred3_order.values
    b = testInfo.probable.values
    c = testInfo.pred1.values
    d = testInfo.pred2.values
    e = testInfo.pred3.values
    for raceId in indices.keys():
        
        idx = indices[raceId]
        
#        mu = testInfo.loc[idx,'probable']
#        a = cvx_mtr(1-mu)
        mu = np.vstack([b[idx], e[idx]])     
        a = cvx_mtr(mu)
     
        cvx_pred[idx] = np.sum(a*np.arange(1,a.shape[0]+1),axis=1)
        
    testInfo['prediction'] = cvx_pred
    testInfo['order'] = testInfo.groupby('raceId')['prediction'].apply(lambda x: (1-x).argsort().argsort()+1)
    showOff = testInfo

def Final():
    X_train,X_test,Y_train,Y_test,trainInfo,testInfo = load_datasets()
    

    
#    analysis(X_train)
#    analysis(X_test)
#    analysis(Y_train)
#    analysis(Y_test)
#    analysis(trainInfo)    
#    analysis(trainInfo)
    
    
    case = case1()
    MLW = ML_Wrapper()

    testInfo = get_predInfo(testInfo,case,X_train,X_test,Y_train,Y_test, MLW)

    testInfo = testInfo[~testInfo.probable.isnull()]


    testInfo =  testInfo.loc[testInfo.disqualificationCode != 'p']

    # Töstä saadaan ennuste > 90
#    testInfo =  testInfo.loc[testInfo.disqualificationCode.isnull()] # tämä antaa tietoa ennustajalle mitä sillä ei todellisuudessa tule olemaan



    X_train = testInfo.iloc[0:20000][['raceId','pred1','pred2','pred3','probable']]
    X_test = testInfo.iloc[20000:][['raceId','pred1','pred2','pred3','probable']]



    for column in ['pred1','pred2']:
        X_train[column+'_order'] = X_train.groupby('raceId')[column].apply(lambda x: x.argsort().argsort()+1)
        X_test[column+'_order'] = X_test.groupby('raceId')[column].apply(lambda x: x.argsort().argsort()+1)

    for column in ['pred3','probable']:
        X_train[column+'_order'] = X_train.groupby('raceId')[column].apply(lambda x: (1-x).argsort().argsort()+1)
        X_test[column+'_order'] = X_test.groupby('raceId')[column].apply(lambda x: (1-x).argsort().argsort()+1)


    for column in ['pred1','pred2','pred3','probable']:

        X_train[column+'-s'] = X_train.groupby('raceId')[column].apply(lambda x: x/x.sum() )
        X_test[column+'-s'] = X_test.groupby('raceId')[column].apply(lambda x:  x/x.sum() )


    X_train = X_train.values[:,-12:]
    X_test = X_test.values[:,-12:]

    Y_train = testInfo.iloc[0:20000]
    Y_train = (Y_train.position==1).astype(int) # ennuste käännetään (ndcg)

    Y_test =  testInfo.iloc[20000:][['raceId','position']].copy()
    Y_test = Y_test.replace(np.nan, 20)

    
    for mat in [X_train,X_test,Y_train,Y_test]:
        print('any is nan', np.any( np.isnan(mat) ) )
        print('all are finite', np.all( np.isfinite(mat) ) )

    MLW.insert_model('SVC',SVC_(case,X_train,X_test,Y_train,Y_test))
    MLW.fit('SVC')
    a = MLW.models['SVC'].result  

        
        

    MLW.insert_model('LogReg',LogReg(case,X_train,X_test,Y_train,Y_test))
    MLW.fit('LogReg')
    a = MLW.models['LogReg'].result



    MLW.insert_model('LogReg',LogReg(case,X_train,X_test,Y_train,Y_test))
    MLW.models['LogReg'].set_model({ 'solver':'saga', 'C':5.0, 'max_iter':100000, 'tol':1e-5, 'penalty':'l1' })
    MLW.save_model('LogReg','Models/{}_tuned')
    
    MLW.read_model('LogReg','Models/{}_tuned')
    pred = MLW.predict('LogReg',X_test)


    showOff = testInfo.iloc[20000:].copy()
    showOff = get_probable(showOff.drop(columns='probable'))
    showOff['prediction'] = pred
    showOff['order'] = showOff.groupby('raceId')['prediction'].apply(lambda x: (1-x).argsort().argsort()+1)
#    showOff['order'] = showOff.groupby('raceId')['pred1'].apply(lambda x: (1-x).argsort().argsort()+1)

    view = showOff[(showOff.position==1)]
    print(view.order.value_counts()/view.order.value_counts().sum())
    print( view.order.value_counts().values[:3].sum()/ view.order.value_counts().sum())

    print(len(view))
    
    print(view.country.value_counts())


    view = showOff[(showOff.order==1)]
    print(view.position.value_counts()/view.position.value_counts().sum())

    view = showOff[(showOff.order==1)&(showOff.probable_order!=1)]
    print(view.position.value_counts()/view.position.value_counts().sum())

    view = showOff[(showOff.order!=1)&(showOff.probable_order==1)]
    print(view.position.value_counts()/view.position.value_counts().sum())

    view = showOff[(showOff.order==1)|(showOff.probable_order==1)]
    print(view.position.value_counts()/view.position.value_counts().sum())


    # kuinka paljon logreg on yhtenäinen
    showOff['order1'] = showOff.groupby('raceId')['pred1'].apply(lambda x: x.argsort().argsort()+1)
    showOff['order2'] = showOff.groupby('raceId')['pred2'].apply(lambda x: x.argsort().argsort()+1)
    showOff['order3'] = showOff.groupby('raceId')['pred3'].apply(lambda x: (1-x).argsort().argsort()+1)
    showOff['order4'] = showOff.groupby('raceId')['probable'].apply(lambda x: x.argsort().argsort()+1)
    view = showOff[(showOff.order==1)]
    print(view.order1.value_counts()/view.order1.value_counts().sum())
    print(view.order2.value_counts()/view.order2.value_counts().sum())
    print(view.order3.value_counts()/view.order3.value_counts().sum())
    print(view.order4.value_counts()/view.order4.value_counts().sum())


    #____ sijavedot:

    view = showOff[(showOff.prediction > 0.4)&(showOff.order==1)]
    print(view.position.value_counts()/view.position.value_counts().sum())
    print(len(view))
    print( view.position.value_counts().values[:3].sum()/ view.position.value_counts().sum())
    
    
    view = showOff[(showOff.prediction > 0.25)&(showOff.order==2)]
    print(view.position.value_counts()/view.position.value_counts().sum())
    print(len(view))
    print( view.position.value_counts().values[:3].sum()/ view.position.value_counts().sum())
    
 
    

    


    view = showOff[(showOff.country=='FI')&(showOff.order==1)]
    print(view.position.value_counts()/view.position.value_counts().sum())
    print( view.position.value_counts().values[:3].sum()/ view.position.value_counts().sum())
    
    view = showOff[(showOff.country=='SE')&(showOff.order==1)]
    print(view.position.value_counts()/view.position.value_counts().sum())
    print( view.position.value_counts().values[:3].sum()/ view.position.value_counts().sum())
    

    view = showOff[(showOff.country=='FI')&(showOff.order==2)]
    print(view.position.value_counts()/view.position.value_counts().sum())
    print( view.position.value_counts().values[:3].sum()/ view.position.value_counts().sum())
    
    view = showOff[(showOff.country=='SE')&(showOff.order==2)]
    print(view.position.value_counts()/view.position.value_counts().sum())
    print( view.position.value_counts().values[:3].sum()/ view.position.value_counts().sum())
    

    view = showOff[(showOff.country=='FI')&(showOff.order==3)]
    print(view.position.value_counts()/view.position.value_counts().sum())
    print( view.position.value_counts().values[:3].sum()/ view.position.value_counts().sum())
    
    view = showOff[(showOff.country=='SE')&(showOff.order==3)]
    print(view.position.value_counts()/view.position.value_counts().sum())
    print( view.position.value_counts().values[:3].sum()/ view.position.value_counts().sum())


    view = showOff[(showOff.order==2)]
    print(view.position.value_counts()/view.position.value_counts().sum())
    print( view.position.value_counts().values[:3].sum()/ view.position.value_counts().sum())
    
    view = showOff[(showOff.order==3)]
    print(view.position.value_counts()/view.position.value_counts().sum())
    print( view.position.value_counts().values[:3].sum()/ view.position.value_counts().sum())


    #___ Kaksaria
    
    view = showOff[(showOff.position==1)|(showOff.position==2)]
    raceIds = view.raceId.value_counts()[view.raceId.value_counts() == 2].index
    view = view[view.raceId.isin(raceIds)]
    print(view.order.value_counts()/view.order.value_counts().sum())
    print(view.raceId.value_counts())
    a = view.order.astype(str).values[0:-1:2]+'-' + view.order.astype(str).values[1::2]
    unique, counts = np.unique(a, return_counts=True)
    df = pd.DataFrame({'unique':unique,'counts':counts})
    
    v2 = view[['raceId','position','order']]
    view = showOff[(showOff.order==1)|(showOff.order==2)|(showOff.order==3)|(showOff.position==1)|(showOff.position==2)]
    print ( view.raceId.value_counts().value_counts())
#    MLW.read_model('LogReg','Models/{}_tuned')

    view = showOff[(showOff.position==1)]
    print(view.order.value_counts()/view.order.value_counts().sum())
    print(view.prediction.describe())

    view = showOff[(showOff.order==1) & (showOff.prediction > 0.3)]
    print(view.position.value_counts()/view.position.value_counts().sum())


    x = 0.29
    y = 2.0
    view = showOff[(showOff.country=='FI')&(showOff.order==1) & (showOff.prediction > x) & (showOff.probable  > y)]
    print(view.position.value_counts()/view.position.value_counts().sum())
    print(len(view))

    view = showOff[(showOff.country=='FI')&(showOff.order==1) & (showOff.prediction > x)& (showOff.position == 1)  & (showOff.probable  > y)]
    print(view.position.value_counts()/view.position.value_counts().sum())
    print(len(view))
    print(np.sum(view.probable.values))


    #___ Sweden

    x = 0.40
    y = 2.0
    view = showOff[(showOff.country=='SE')&(showOff.order==1) & (showOff.prediction > x) & (showOff.probable  > y)]
    print(view.position.value_counts()/view.position.value_counts().sum())
    print(len(view))

    view = showOff[(showOff.country=='SE')&(showOff.order==1) & (showOff.prediction > x)& (showOff.position == 1)  & (showOff.probable  > y)]
    print(view.position.value_counts()/view.position.value_counts().sum())
    print(len(view))
    print(np.sum(view.probable.values))

def get_poolId(df,Event,startNumber):
    return int(df[ (df.trackName==Event) & (df.number==startNumber)]['poolId'])

def predict_current(O,raceInfo,Event,startNumber,MLW):
    betList = raceInfo[['trackName','number','poolId','raceId']].drop_duplicates().copy()
    poolId = get_poolId(betList,Event,startNumber)
    odds = O.request_odds(poolId)
    odds.probable = odds.probable/100.0
    odds.loc[:,'probable'] = 1/(1+odds.loc[:,'probable'])
    odds = odds[['poolId','probable','runnerNumber']]

    toBet = raceInfo[raceInfo.poolId==poolId].copy()
    toBet.drop(columns=['probable'],inplace=True)
    toBet = toBet.merge(odds,how='left',left_on=['poolId','startNumber'], right_on=['poolId','runnerNumber'])
    toBet.fillna(0.0,inplace=True)



    for column in ['pred1','pred2']:
        toBet[column+'_order'] = toBet.groupby('raceId')[column].apply(lambda x: x.argsort().argsort()+1)
        toBet[column+'_order'] = toBet.groupby('raceId')[column].apply(lambda x: x.argsort().argsort()+1)

    for column in ['pred3','probable']:
        toBet[column+'_order'] = toBet.groupby('raceId')[column].apply(lambda x: (1-x).argsort().argsort()+1)
        toBet[column+'_order'] = toBet.groupby('raceId')[column].apply(lambda x: (1-x).argsort().argsort()+1)


    for column in ['pred1','pred2','pred3','probable']:

        toBet[column+'-s'] = toBet.groupby('raceId')[column].apply(lambda x: x/x.sum() )
        toBet[column+'-s'] = toBet.groupby('raceId')[column].apply(lambda x:  x/x.sum() )




    X_data = toBet[['pred1','pred2','pred3','probable','pred1_order','pred2_order','pred3_order','probable_order','pred1-s','pred2-s','pred3-s','probable-s']].values
    
    prediction = MLW.predict('LogReg',X_data)


    toBet['prediction'] = prediction

    toBet['order'] = toBet[['prediction']].apply(lambda x: (1-x).argsort().argsort()+1)

    return toBet.copy()



def today():
    X_train,_,_,_,_,_ = load_datasets()
    O = OddsChecker(False)
    MLW = ML_Wrapper()
    case = case1()

    MLW.insert_model('LogReg',LogReg(case,None,None,None,None))
    MLW.read_model('LogReg','Models/{}_tuned')

    TDH = TodayDataHandler(case)
    TDH.data_from_disk()
    raceInfo, X_today = TDH.get_X()
    if 'startType.BOOL.UNKNOWN' not in X_today.columns:
        X_today['startType.BOOL.UNKNOWN'] = np.zeros(len(X_today))

    if 'breed.BOOL.S' not in X_today.columns:
        X_today['breed.BOOL.S'] = np.zeros(len(X_today))

    # Sort columns
    X_today = X_today[X_train.columns]


    #___ pitäisikö tässä olla jokin hakukriteeri (tai no oikeastaan kaikissa)
    # nyt haetaan _tuned nimistä
    # pitäisikö jatkossa olla vaikka _rev1
    raceInfo = get_predInfo(raceInfo,case,None,X_today,None,None, MLW)

    betList = raceInfo[['trackName','number','poolId']].drop_duplicates().copy()

#    poolId = get_poolId(betList, 'Kuopio',1)


    bet = predict_current(O,raceInfo,'Ylivieska',5,MLW)
    show(bet)

    for i in np.arange(4,8):
        bet = predict_current(O,raceInfo,'Mikkeli',i,MLW)
        show(bet)


    bet = predict_current(O,raceInfo,'Dannero',2,MLW)
    show(bet)
