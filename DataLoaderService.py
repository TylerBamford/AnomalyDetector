from os import listdir
import DataManipulationService
import pandas as pd
import json

def getDataFiles(directoryName):
    """
    Gets the files from a given directory
    
    @param directoryName: string of directory to find files
    
    @returns list: list of all the files in the directory
    """
    
    return listdir(directoryName)

def loadData(directoryName, fileName):
    """
    Changes a given data file to include the UUID in the header and
        passes it on to be manipulated
    
    @param directoryName: directory of files
    @param fileName: name of the file that contains the data
    
    @returns dictionary: dictionary of UUID's to their corresponding data
    """

    dataset = pd.read_csv(directoryName + "/" + fileName, header = None)
    with open(directoryName + "/" + "metadata.json") as json_file:
        metadata = json.load(json_file)
    for uuid in metadata:
        for column in metadata[uuid]:
            index = int(metadata[uuid][column])
            dataset.iloc[0, index] = uuid
    
    return DataManipulationService.createDictionary(dataset)
