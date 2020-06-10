ECHO off
CALL "D:\Anaconda\Scripts\activate.bat"
CALL conda activate RAVI
CALL python daily_data_fetch.py
pause