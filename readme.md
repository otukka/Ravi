# Predicting harness racing results

This repository contains mainly automatic data collection modules and few example ML-models.

## Automatic data collection

daily_data_fetch.py collects automatically previous date harness racing results from www.veikkaus.fi. JSON data collected from source is stored as gzip files for each day. This is because we want to store the original data from source in case we make error parsing data in some point.

![Alt text](UML/Daily%20Request.png?raw=true "")


## ML models

Few example ML models are concluded and they can be trained if some wants to implement data parsing.

## Data parsing

Under construction.

Basic idea: Varbiables X : history data for race runner. Target Y : position in race.


