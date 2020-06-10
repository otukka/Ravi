# -*- coding: utf-8 -*-

import time
from Wrappers.RequestWrapper import RequestWrapper

class FetchWrapper:

    def __init__(self, requestwrapper=None, save=True, path='D:/Ravi_json/raw_json_v2'):
        
        if requestwrapper == None:
            self.RW_ = RequestWrapper()
            self.RW_.set_path(path)
        else:
            self.RW_ = requestwrapper

        self.save_ = save

    def check_correctness_(self, return_list):

        if len(return_list) == 0:
            raise ValueError('DataFetcher returned empty dataset.')

        # if not isinstance(return_list, list):
        #     return_list = list(return_list)
            
        return return_list


    def get_looper(self, IDs, requestType):

        if not isinstance(IDs,list):
            IDs = list(IDs)
        
        return_list = []
        for id_ in IDs:
            return_value = self.RW_.fetch_data( requestType, id_, self.save_ )
            return_list.append( return_value )
            time.sleep(0.01)
            
        return self.check_correctness_( return_list )



    # Initial step, done once: event data:

    def get_events(self):
        return self.check_correctness_( self.RW_.fetch_data( 'events', save=self.save_ ) )
    
    def get_eventsDate(self, date):
        return self.check_correctness_( self.RW_.fetch_data( 'eventsDate', date, save=self.save_ ) )


    # looping starts here, done for every ID:


    # data from events

    def get_eventPools(self, cardIds):
        return self.get_looper( cardIds, 'eventPools' )
    def get_eventRaces(self, cardIds):
        return self.get_looper( cardIds, 'eventRaces' )
    def get_eventRaceinfo(self, cardIds):
        return self.get_looper( cardIds, 'eventRaceinfo' )
    def get_eventResults(self, cardIds):
        return self.get_looper( cardIds, 'eventResults' )

    # data from pools

    def get_poolRunners(self, poolIds):
        return self.get_looper( poolIds, 'poolRunners' )
    def get_poolOdds(self, poolIds):
        return self.get_looper( poolIds, 'poolOdds' )

    # data from races

    def get_raceRunners(self, raceIds):
        return self.get_looper( raceIds, 'raceRunners' )
    def get_racePools(self, raceIds):
        return self.get_looper( raceIds, 'racePools' )
    def get_raceResults(self, raceIds):
        return self.get_looper( raceIds, 'raceResults' )

    # data from runners

    def get_runnerStats(self, runnerIds):
        return self.get_looper( runnerIds, 'runnerStats' )
    def get_runnerDatamaatti(self, runnerIds):
        return self.get_looper( runnerIds, 'runnerDatamaatti' )


