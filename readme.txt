Anomaly Detector

Setup:
Windows
- extract "test-dataset.csv.zip"
- create folders "TestingData" and "TrainingData"
- copy the extracted file into both the folders
- copy "metadata.json" into TrainingData folder

Other OS
- make sure there are no sub directories within the AnomalyDetector folder

Running:
Windows
- in command console, navigate to AnomalyDetector folder and execute "run.bat"

Other OS
- in command console, navigate to AnomalyDetector folder and execute "run.sh"

Description:
The anomaly detector application takes in a csv file and breaks it into individual datasets. 
These are saved as files in the TestingData folder which can be stopped by removing lines
62 and 63 in DataManipulationService.py. The neural network models will be saved in the
NetworkModels folder along with their weights. When an anomaly is found, a JSON file will be
created in the Logs folder which details the datetimes of all the anomalies and their value. 
A plot is also generated in the plots folder for testing purposes to make sure the application
is working properly.