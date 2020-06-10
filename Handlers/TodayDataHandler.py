# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 21:25:21 2019

@author: VStore
"""

import os
import time
import datetime

import numpy as np
import pandas as pd

from DataHandler import DataHandler
from df_utils import *
from Cases import *
from DataParser import *
from FetchWrapper import Fetcher
from JSONnormalizer import JSONnormalizer

from common import *

class TodayDataHandler(DataHandler):


    def __init__(self,case):
        self.today_ = '{0:%Y-%m-%d}'.format(datetime.datetime.now().date())
        self.today_ = '2019-08-13'
        case['filterDate'] = self.today_  
        case['today'] = True
        
        self.debug_ = True
        
        super().__init__(case,True)
        self.data_ = {}

    def run_formats(self):
        self._basic_filter()    

        self._format_stats()
        
        self._format_prevStarts()  
        
        self._transpose_prevStarts()

        self._format_pool_runners()
        
        self._filter_data()
        
        self._format_data()
        
        self._create_X()

     
    def data_from_disk(self):
        
        
        dataparser = DataParser()
        self.data_['events'] = dataparser.get_events(self.today_)
        self.data_['datamaatti'] = dataparser.get_datamaatti(self.today_)
        self.data_['pools'] = dataparser.get_pools(self.today_)
        self.data_['odds'] = dataparser.get_odds(self.today_)
        self.data_['prevStarts'] = dataparser.get_prevStarts(self.today_)
        self.data_['races'] = dataparser.get_races(self.today_)
        self.data_['stats'] = dataparser.get_stats(self.today_)
        self.data_['aggregates'] = dataparser.get_aggregates(self.today_)
        self.data_['pool_runners'] = dataparser.get_pool_runners(self.today_)

        self.data_['race_results'] = pd.DataFrame()
        self.data_['card_results'] = pd.DataFrame()

        self.run_formats()

    def data_from_web(self, eventFilter=None, raceFilter=None, poolFilter=None):
        
        
        F = Fetcher(save=False)
        J = JSONnormalizer()

        

        self.data_['events'] = J.normalize_events(F.get_events())
        if eventFilter != None:
            self._keep_data('events',None,{'trackName':eventFilter})

        cardIds = self.data_['events']['cardId'].unique()

        self.data_['races'] = J.normalize_races(F.get_races( cardIds ))
        if raceFilter != None:
            self._keep_data('races',None,{'number':raceFilter})


        self.data_['pools'] = J.normalize_pools(F.get_pools(cardIds))
        if poolFilter != None:
            self._keep_data('pools',None,{'poolType':poolFilter})
        
        poolIds = self.data_['pools']['poolId'].unique()

        self.data_['pool_runners']  = J.normalize_pool_runners(F.get_pool_runners( poolIds ))

        # no need to load several times
        runnerIds = self.data_['pool_runners']['runnerId'].unique()
        stats = F.get_stats(runnerIds)

        self.data_['stats'] = J.normalize_stats(stats)
        self.data_['prevStarts'] = J.normalize_prevStarts(stats)

        
        df_racedata = create_event_runner_list(self.data_['events'], self.data_['races'], self.data_['pool_runners'])

        df = self.data_['prevStarts'].copy()

        df = df.merge(df_racedata,how='inner', on='runnerId', validate='m:1')
        
        
        df = handle_result_column(df)            
        df = handle_kmTime_column(df)
        df = calculate_aggregates(df)
        self.data_['aggregates'] = df.copy()

 

        self.run_formats()


def test():  
    case = case1()     
    case['minimumFirstPrize'] = False
    TDH = TodayDataHandler(case)
    TDH.data_from_disk()
#    TDH.data_from_web('Jokimaa',4,'VOI')
    events= TDH.data_['events']
    races= TDH.data_['races']    
    

    
    odds= TDH.data_['odds']
    pools= TDH.data_['pools']
    pool_runners= TDH.data_['pool_runners']
    prevStarts= TDH.data_['prevStarts']
    prevStarts_transposed= TDH.data_['prevStarts_transposed']

    stats= TDH.data_['stats']
    driverStats = TDH.data_['driverStats']
    aggregates = TDH.data_['aggregates']
    
    missing = missing_values_table(TDH.data_['X'])
    
    len(TDH.data_['X'])
    len(prevStarts[(prevStarts.timeDifference < 30)&(prevStarts['kmTime.BOOL.a']==0)])
    len(prevStarts[(prevStarts.timeDifference < 30)&(prevStarts['kmTime.BOOL.a']==1)])
    #
    #card_results= TDH.data_['card_results']
    #race_results= TDH.data_['race_results']
#    #
#    #
    X = TDH.data_['X']
    view = X[['raceId','startNumber','stats.previousYear.starts','stats.previousYear.starts_scaled']]
    view = X[['raceId','startNumber','aggregates.kmTime.15.CAR_START.max','aggregates.kmTime.15.CAR_START.max_scaled']]
#    ##Y = TDH.data_['Y']
#    #
#    #
#    raceInfo, X_today = TDH.get_X()
