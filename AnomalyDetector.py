import os
import DataLoaderService
import AnomalyDetectionService
    
if __name__ == "__main__":
    print("Manipulating Data into known format...")
    directory = "TrainingData"
    modelsDirectory = "NetworkModels"
    dataFiles = DataLoaderService.getDataFiles(directory)
    for file in dataFiles:
        if file.endswith(".csv"):
            dataDictionary = DataLoaderService.loadData(directory, file)
            #os.remove(directory + "/" + file)
            #os.remove(directory + "\metadata.json")

    print("Training and Detecting Models...")
    AnomalyDetectionService.trainModel(dataDictionary.dictionary, modelsDirectory, "TestingData\dataset1.csv")