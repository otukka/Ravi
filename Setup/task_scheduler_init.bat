ECHO off
CALL SCHTASKS /CREATE /TN ravi_python /XML ravi_python.xml
pause