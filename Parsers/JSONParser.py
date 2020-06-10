
import time
import pandas as pd


class json2DataFrame:

    def drop_columns_(self, df, columns):
        if len(df) > 0:
            for column in columns:
                if column in df.columns:
                    df.drop(columns=column, inplace=True)
        return df
    
    def basic_normalization_(self, json_file, record_path=None, meta=None):
        df = pd.json_normalize(data=json_file, record_path=record_path)
        if len(df) > 0:
            return df
    
    def loop_normalization_(self, json_file_array, record_path=None, meta=None, max_level=None):
        result = []
        for d in json_file_array:
            result.append(pd.json_normalize(
                                    data=d['content']['collection'],
                                    record_path=record_path,
                                    meta=meta,
                                    max_level=max_level))
        if len(result) > 0:
            df = pd.concat(result,sort=True)
            return df
        
        
    def normalize_events(self, json_array):        
        df = self.basic_normalization_(json_array['content'], record_path='collection')
        # df = self.drop_columns_(df, columns=['bonusPools','jackpotPools','totoPools'])
        return df


    def normalize_eventPools(self, json_file_array):
        return self.loop_normalization_(json_file_array)
    def normalize_eventRaces(self, json_file_array):
        return self.loop_normalization_(json_file_array)
    def normalize_eventRaceinfo(self, json_file_array):
        return self.loop_normalization_(json_file_array)
    
    def normalize_eventResults(self, json_file_array):
        df = self.loop_normalization_(
                                        json_file_array,
                                        record_path=['results'],
                                        meta=['cancelledRace','card','raceId','raceNumber','raceStatus','scratched'],
                                        )
        df = pd.concat([df.drop(['card'], axis=1), df['card'].apply(pd.Series)], axis=1)
        return df

    def normalize_poolRunners(self, json_file_array):
        return self.loop_normalization_(json_file_array)
    def normalize_poolOdds(self, json_file_array):
        df = self.basic_normalization_(
                                        json_file_array, 
                                        record_path='odds',
                                        meta=['poolId','netSales','netPool','updated'],
                                        )
        if len(df) > 0:
            df['updatedStr'] = df['updated'].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(x)/1000)))
            return df
    



                                        
    def normalize_raceResults(self, json_file_array):
        return self.basic_normalization_(
                                        json_file_array, 
                                        record_path='runners',
                                        meta=['raceId','cardId','raceNumber','startType','monte','breed','startTime'],
                                        )

    def normalize_raceRunners(self, json_file_array):
        return self.basic_normalization_(
                                        json_file_array, 
                                        record_path='collection',
                                        )

    def normalize_runnerDatamaatti(self, json_file_array):
        return self.basic_normalization_(
                                        json_file_array,
                                        record_path=['races'],
                                        meta=['horseName', 'runnerNumber'],
                                        )

    def normalize_runnerStats(self, json_array):
        individual_runners = []
        for arr in json_array:
            keys = []
            values = []
    
            keys.append('runnerId')
            values.append(arr['runnerId'])
    
            for k1 in arr['stats'].keys():
                for k2 in arr['stats'][k1].keys():
                    keys.append('{}.{}.{}'.format('stats',k1,k2))
                    values.append(arr['stats'][k1][k2])  
    
            individual_runners.append(dict(zip(keys,values)))
    
        df = pd.DataFrame(individual_runners)

        if len(df) > 0:
            return df     


    def normalize_prevStarts(self, json_file_array):
        return self.basic_normalization_(
                                        json_file_array,
                                        record_path='prevStarts',
                                        meta=['runnerId'],
                                        )


def test():
    
    import pickle
    with open('../Handlers/RD_request.dat','rb') as f:
        var = pickle.load(f)
        


    J = json2DataFrame()
    
    df_events= J.normalize_events(var['events'])

    df_eventPools = J.normalize_eventPools(var['eventPools'])
    df_eventResults = J.normalize_eventResults(var['eventResults'])
    df_eventRaces = J.normalize_eventRaces(var['eventRaces'])
    df_eventRaceinfo = J.normalize_eventRaceinfo(var['eventRaceinfo'])
    
    # Tässä
    df_raceRunners = J.normalize_raceRunners(var['raceRunners'])
    df_racePools = J.normalize_racePools(var['racePools'])
    df_raceResults = J.normalize_raceResults(var['raceResults'])

    df_poolRunners = J.normalize_poolRunners(var['poolRunners'])
    df_poolOdds = J.normalize_poolOdds(var['poolOdds'])



        