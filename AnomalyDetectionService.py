from os import listdir
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

def trainModel(uuidToDataMap, modelsDirectory, testData):
    """
    Trains a model based on the given data
    
    @param {uuidToDataMap<string, dataframe>} uuidToDataMap: dictionary of uuid's to their data
    @param modelsDirectory: directory of where the data is stored
    """

    files = listdir(modelsDirectory)
    for uuid in uuidToDataMap:
        if(uuidToDataMap[uuid].empty):
            break

        foundModel = False
        for file in files:
            if(uuid + ".json" == file):
                foundModel = True

        if(foundModel):
            network = loadModel("NetworkModels/" + uuid + ".json", uuid)
            print("loaded " + uuid)
        else:
            network = createModel("NetworkModels/" + uuid + ".json", uuid)
            print("created " + uuid)
    
        trainValues = uuidToDataMap[uuid]

        X_train = trainValues.iloc[1:int(trainValues.shape[0] * .5), 2:5]
        Y_train = trainValues.iloc[1:int(trainValues.shape[0] * .5), 5]

        network.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
        network.fit(X_train, Y_train, batch_size = 10, epochs = 10)
            
        model_json = network.to_json()
        with open("NetworkModels/" + uuid + ".json", "w") as json_file:
            json_file.write(model_json)
        network.save_weights("NetworkModels/" + uuid + ".h5")
                
        testModel(network, uuid)


def testModel(network, uuid):
    """
    Tests the model on the given testing dataset
    
    @param network: network model to test
    @param testData: the 
    """

    testData = ("TestingData/" + uuid + ".csv")

    testDataset = pd.read_csv(testData, header = None)

    X_test = testDataset.loc[:testDataset.shape[0] - 1, 3:5]
    Y_test = testDataset.loc[:testDataset.shape[0] - 1, 6]
    standDev = testDataset.loc[:testDataset.shape[0] - 1, 7]

    Y_pred = network.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred)
    standDev = pd.DataFrame(standDev)

    X_values = testDataset.iloc[:, 1]

    Y_test = Y_test.to_numpy()
    standDev = standDev.to_numpy()
    standardDeviationHigh = np.zeros(Y_test.shape)
    standardDeviationLow = np.zeros(Y_test.shape)

    for row in range(1, standardDeviationHigh.shape[0]):
        standardDeviationHigh[row] = float(standDev[row])
        standardDeviationLow[row] = float(standDev[row])
    Y_pred = pd.DataFrame(Y_pred)
    X_values = pd.DataFrame(X_values)
    standardDeviationHigh = pd.DataFrame(standardDeviationHigh)
    standardDeviationLow = pd.DataFrame(standardDeviationLow)

    X_values.reset_index(inplace = True, drop = True)

    standardDeviationHigh.insert(loc = 0, column = "value", value = testDataset.iloc[:, 0])
    standardDeviationLow.insert(loc = 0, column = "value", value = testDataset.iloc[:, 0])

    plotGraphs(uuid, Y_pred, Y_test, standardDeviationHigh, standardDeviationLow, X_values)

def createModel(modelFile, uuid):
    """
    Creates a model with the name as the given uuid
    
    @param modelFile: model file name
    @param uuid: uuid of the data
    
    @returns network: returns the created model
    """
    
    network = Sequential()
    network.add(Dense(activation = "relu", input_dim = 3, units = 10, kernel_initializer = "uniform"))
    network.add(Dense(activation = "relu", units = 15, kernel_initializer = "uniform"))
    network.add(Dense(activation = "relu", units = 10, kernel_initializer = "uniform"))
    network.add(Dense(activation = "linear", units = 1, kernel_initializer = "uniform"))
    
    network_json = network.to_json()
    with open(modelFile, "w") as json_file:
        json_file.write(network_json)
    network.save_weights("NetworkModels/" + uuid + ".h5")
    return network
    
def loadModel(modelFile, uuid):
    """
    Loads the given model file and sets it to the network
    
    @param modelFile: model file name
    @param uuid: uuid of the data
    
    @returns network: returns the loaded model
    """
    
    json_file = open(modelFile, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    network = model_from_json(loaded_model_json)
    network.load_weights("NetworkModels/" + uuid + ".h5")
    return network

def plotGraphs(uuid, Y_pred, Y_test, standardDeviationHigh, standardDeviationLow, X_values):
    """
    Compares the actual values to the standard deviation bounds to
    determine the anomalies. If there are anomalies, a plot and JSON
    object are created.

    Parameters
    ----------
    uuid : string
        The UUID of the dataset being worked on
    Y_pred : pandas dataframe
        The predicted values generated by the nerual network
    Y_test : pandas dataframe
        The actual values at the given points
    standardDeviationHigh : pandas dataframe
        A dataframe of the high bounds created from the standard deviation
    standardDeviationLow : pandas dataframe
        A dataframe of the low bounds created from the standard deviation
    X_values : pandas dataframe
        A dataframe of the date times where the points are located

    Returns
    -------
    None.

    """

    Y_test = pd.DataFrame(Y_test)
    plt.figure(figsize=(20,10))

    newIndex = 0
    secondIndex = 0
    indexColumn = standardDeviationHigh.loc[:, "value"]
    for index in range(indexColumn.shape[0]):
        if(indexColumn.loc[index] >= 1121177):
            newIndex = index;
            break;
    for index in range(indexColumn.shape[0]):
        if(indexColumn.loc[index] >= 1127305):
            secondIndex = index;
            break;

    Y_pred = Y_pred.loc[newIndex:secondIndex, :]
    Y_test = Y_test.loc[newIndex:secondIndex, :]
    standardDeviationHigh = standardDeviationHigh.loc[newIndex:secondIndex, :]
    standardDeviationLow = standardDeviationLow.loc[newIndex:secondIndex, :]
    indexColumn= indexColumn.loc[newIndex:secondIndex]
    X_values = X_values.loc[newIndex:secondIndex, :]

    standardDeviationHigh = pd.DataFrame(standardDeviationHigh)
    standardDeviationLow = pd.DataFrame(standardDeviationLow)
    Y_pred = pd.DataFrame(Y_pred)
    X_values = pd.DataFrame(X_values)

    #iterates through the graph points and marks the anomalies based
    #on a set bound
    print("creating deviations")
    for index in range(newIndex, secondIndex):
        standardDeviationHigh.loc[index, 0] = Y_pred.loc[index, 0] + (standardDeviationHigh.loc[index, 0] * 4)
        standardDeviationLow.loc[index, 0] = Y_pred.loc[index, 0] - (standardDeviationLow.loc[index, 0] * 4)

    notAnomaly = 0
    anomalyCount = 0
    print("plotting")
    anomalies = []

    for row in range(newIndex + 1, secondIndex):
        anomaly = {}
        highValue = standardDeviationHigh.loc[row, 0]
        lowValue = standardDeviationLow.loc[row, 0]
        actualValue = Y_test.loc[row, 0]

        if(highValue > actualValue and lowValue < actualValue):
            notAnomaly += 1
        else:
            plt.axvspan(row - 2, row + 2, facecolor = "#a70000", alpha = .5)
            print(row)
            anomaly["x"] = X_values.at[row, 1]
            anomaly["y"] = actualValue
            anomalies.append(anomaly)
            anomalyCount += 1

    standardDeviationHigh.drop(standardDeviationHigh.tail(1).index, inplace = True)
    standardDeviationLow.drop(standardDeviationLow.tail(1).index, inplace = True)

    plt.plot(Y_pred.loc[1:, 0], color = "black", label = "Predicted Values", linewidth = 1)
    plt.plot(Y_test.loc[1:, 0], color = "blue", label = "Actual Values", linewidth = 1)
    plt.plot(standardDeviationHigh.loc[1:, 0], color = "yellow", label = "Standard Deviation High", linewidth = 1)
    plt.plot(standardDeviationLow.loc[1:, 0], color = "green", label = "Standard Deviation Low", linewidth = 1)

    plt.legend(loc = "upper left")
    plt.xticks(np.arange(newIndex,secondIndex,15), X_values.loc[newIndex:secondIndex:15, 1], rotation = 45)
    print("done marking")
    print("Non-Anomylous Points: ", notAnomaly)
    print("Anomylous Points: ", anomalyCount)

    if(not anomalyCount == 0):
        plt.savefig("plots/" + uuid + ".png")

        data = {
            "mnemonicUuid": uuid,
            "anomalies": anomalies
        }
        with open("Logs/" + uuid + "_anomaly_log.json", "w") as write_file:
            json.dump(data, write_file)


