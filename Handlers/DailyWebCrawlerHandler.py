# -*- coding: utf-8 -*-

from Wrappers.RequestWrapper import RequestWrapper
from Wrappers.FetchWrapper import FetchWrapper
from Wrappers.gzipWrapper import gzipWrapper
from Parsers.JSONParser import json2DataFrame

import pathlib
import pandas as pd
import datetime


class DailyWebCrawlerHandler():



    def __init__(self):
        self.path = pathlib.Path('D:/Ravi_json/raw_json_RD_v4')
        # self.session_ = RequestWrapper()
        # self.session_.SetPath(path)
        self.Fetcher_ = FetchWrapper(requestwrapper=None, save=False)
        self.J_ = json2DataFrame()
        
        self.data_ =  {
                    'events':[],
                    'eventPools':[],
                    'eventResults':[],
                    'eventRaces':[],
                    'eventRaceinfo':[],
                    'raceRunners':[],
                    'racePools':[],
                    'raceResults':[],
                    # 'runnerStats':[],
                    # 'runnerDatamaatti':[],
                    'poolRunners':[],
                    'poolOdds':[],
                      }

        
        self.fetced_ = False
        


        
    def fetch_results(self):
        date = self.get_yesterday()
        self.data_['events'] = self.Fetcher_.get_eventsDate(date)
        
        cardIds = self.get_cardIds()
        self.data_['eventPools'] = self.Fetcher_.get_eventPools(cardIds)
        self.data_['eventResults'] = self.Fetcher_.get_eventResults(cardIds)
        self.data_['eventRaces'] = self.Fetcher_.get_eventRaces(cardIds)
        self.data_['eventRaceinfo'] = self.Fetcher_.get_eventRaceinfo(cardIds)
        
        raceIds = self.get_raceIds()
        self.data_['raceResults'] = self.Fetcher_.get_raceResults(raceIds)
        self.data_['raceRunners'] = self.Fetcher_.get_raceRunners(raceIds)
        self.data_['racePools'] = self.Fetcher_.get_racePools(raceIds)
        
        poolIds = self.get_poolIds()
        self.data_['poolRunners']  = self.Fetcher_.get_poolRunners(poolIds)
        self.data_['poolOdds']  = self.Fetcher_.get_poolOdds(poolIds)
        
        self.fetced_ = True
        
        
    def get_yesterday(self):
        date = datetime.datetime.now().date()-datetime.timedelta(days=1)
        return '{0:%Y-%m-%d}'.format(date)
        
    def get_cardIds(self):        
        cardIds = self.J_.normalize_events(self.data_['events'])['cardId'].unique()
        return cardIds
    
    def get_raceIds(self):        
        raceIds = self.J_.normalize_eventRaces(self.data_['eventRaces'])['raceId'].unique()
        return raceIds
    
    def get_runnerIds(self):
        df = self.J_.normalize_raceRunners(self.data_['raceRunners'])
        return df['runnerId'].unique()

    
    def get_poolIds(self):
        df = self.J_.normalize_eventPools(self.data_['eventPools'])
        return df['poolId'].unique()

    
    def get_data(self):
        if not self.fetced_:
            self.fetch_results()
        return self.data_

    def save_gzip(self):
        gW = gzipWrapper()
        for key, value in self.data_.items():
            path = self.path.joinpath(key,self.get_yesterday()+'.gz')
            gW.save(path,value)


# def main():
#     DWCH = DailyWebCrawlerHandler()
#     var = DWCH.get_data()
#     DWCH.save_gzip()    


# main()
