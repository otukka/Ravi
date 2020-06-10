
call D:\Anaconda\Scripts\activate.bat D:\Anaconda\

SET env_name=RAVIt


call conda create --name %env_name% -y python=3.7
call conda install -n %env_name% -y  spyder==4.0.1
call conda install -n %env_name% -y numpy scipy ipython jupyter pandas sympy nose scikit-learn
call conda install -n %env_name% -y -c conda-forge matplotlib
call conda install -n %env_name% -y -c conda-forge pyarrow
call conda install -n %env_name% -y -c anaconda flask
call conda install -n %env_name% -y -c conda-forge tensorflow-gpu

REM Not yet windows build in conda
REM call conda install -n %env_name% -y -c conda-forge xgboost
call conda activate %env_name%
call pip install xgboost-1.0.2-py3-none-win_amd64.whl
call conda deactivate


REM If interactive plots don't work in spyder:
REM set Spyder > Preferences > IPython console > Graphics > Graphics backend = Automatic 
pause

