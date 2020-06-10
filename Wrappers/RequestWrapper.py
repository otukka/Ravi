# -*- coding: utf-8 -*-


import os
import time
import datetime
import json
import requests
import pathlib
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter



class RequestWrapper:

    def __init__(self):
        self.session = requests.Session()
        self.path_ = ''

        # retry if failed
        # self.failed_requests_ = 0
        # retries = Retry(total=2, backoff_factor=0.1,)#status_forcelist=[ 500, 502, 503, 504 ]
        # self.session.mount('https://',HTTPAdapter(max_retries=retries))


        self.headers = {
            'Content-type':'application/json',
            'Accept':'application/json',
            'X-ESA-API-Key':'ROBOT'
        }

        self.urls = {
                    'host':'https://veikkaus.fi',

                    # card -> event

                    'events':'/api/toto-info/v1/cards/today', # kuluvan päivän tapahtumat
#                   'future':'/api/toto-info/v1/cards/future', # tulevie npäivien tapahtumat
#                   'active':'/api/toto-info/v1/cards/active', # kuluvan ja tulevien päivien tapahtumat
                    'eventsDate':'/api/toto-info/v1/cards/date/{}', # annetun päivän tapahtumat esim 2017-04-18

                    'eventPools':'/api/toto-info/v1/card/{}/pools', # tapahtuman kaikki pelikohteet
                    'eventResults':'/api/toto-info/v1/card/{}/results', # tapahtuman tulokset
                    'eventRaces':'/api/toto-info/v1/card/{}/races', # tapahtuman lähdöt
                    'eventRaceinfo':'/api/toto-info/v1/card/{}/raceinfo', # tapahtuman perustietoja


                    'raceRunners':'/api/toto-info/v1/race/{}/runners', # lähdön osallistujat 
                    'racePools':'/api/toto-info/v1/race/{}/pools', # yhden lähdön pelikohteet
                    'raceResults':'/api/toto-info/v1/race/{}/competition-results', # yhden lähdön lopputulokset



                    'runnerStats':'/api/toto-info/v1/runner/{}/stats', # löytyy myös race/runners
                    'runnerDatamaatti':'/api/toto-info/v1/runner/{}/datamaatti', # toimii vain suomalaisissa lähdöissä

                    'poolRunners':'/api/toto-info/v1/pool/{}/runners', # pelikohteen osallistujat
                    'poolOdds':'/api/toto-info/v1/pool/{}/odds', # pelikohteen kertoimet
                    }

    def set_path(self, path):
        self.path_ = pathlib.Path(path)
        assert self.path_.is_dir() == True, "Path not directory in RequestWrapper.set_path()"


    def get_date(self):
        date = datetime.datetime.now()
        date = '{0:%Y-%m-%d}'.format(date)
        return date


    def format_save_filename(self, requestType, ID):
        
        if ID == None:
            ID = ''
        date = self.get_date()
        return '{}_{}_{}.json'.format(date, requestType, ID)
        

    def save_raw_json(self, content, requestType, ID):

        content = json.dumps(content)

        p = self.path_.joinpath(self.format_save_filename(requestType, ID))
        
        with open(p, 'w', encoding="utf-8") as f:
            f.write(content)


    def fetch_data(self, requestType, ID=None, save=True):

        assert (requestType == 'events') == (ID == None)
        
        url = self.urls['host'] + self.urls[requestType]

        if ID != None:
            url = url.format( str(ID) )


        print(url)

        r = self.session.get(url, headers=self.headers)

        if r.status_code == 200:
            # content = r.json()            
            content = {'requestType':requestType, 'ID':str(ID), 'content':r.json()}
        else:
            raise Exception('Query failed', requestType, ID, r.status_code)

        if save:
            self.save_raw_json(content, requestType, ID)

        return content




def test():
    RW = RequestWrapper()
    RW.set_path('D:/Ravi_json/raw_json_RD')
    events = RW.fetch_data('events', ID=None, save=False)
    events = RW.fetch_data('events', ID=None, save=True)
    

    raceRunner = RW.fetch_data('raceRunners', 4887419, False)
    runnerDatamaatti = RW.fetch_data('runnerDatamaatti', 59280041, False)
    








