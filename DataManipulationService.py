import Dictionary
import numpy as np
import pandas as pd

def createDictionary(dataset):
    """
    Creates a dictionary of UUID's to their corresponding data
    
    @param dataset: a 2-D array of data to be manipulated
    
    @returns Dictionary: object containing the UUID dictionary
    """
    for columnNumber in range(2, dataset.shape[1]):
        print("manipulating ", dataset.at[0, columnNumber])
        manipulateData(columnNumber, dataset)
    return Dictionary

def manipulateData(columnNumber, dataset):
    """
    Manipulates given data to create the data that will be tested
    
    @param {int} columnNumber: column index where the needed data is
    @param dataset: 2-D array of all data in the original file
    
    @returns Dictionary: object that contains the dictionary of UUID's
    """

    #Make data the first colum and columnNumber column only
    uuid = dataset.at[0, columnNumber]
    newData = dataset.iloc[1:, 0: columnNumber + 1: columnNumber]
    timesColumn = dataset.loc[1:, 1]
    newData = newData.astype(float)
    newData = newData[~np.isnan(newData.iloc[:, 1])]

    newData.insert(loc = 0, column = "Time(UTC)", value = timesColumn)
    standardDeviationColumn = newData.iloc[:, 2].rolling(100).std()

    #Reset column index number to be sequential
    newData.columns = range(newData.shape[1])

    #Create a new column and initialize values to 0
    #lastColumn = dataset.iloc[:, dataset.shape[1] - 1]
    newData.insert(newData.shape[1], newData.shape[1], 0)
    newData.insert(newData.shape[1], newData.shape[1], 0)
    newData.insert(newData.shape[1], newData.shape[1], 0)

    #make new column as the previous value
    originalColumn = newData.loc[1:, 2]
    newData = newData.astype({3:'float64', 4:'float64', 5:'float64'})
    for index in range(newData.index.shape[0] - 1):
        dataIndex = newData.index[index]
        newData.at[dataIndex, 3] = originalColumn.at[newData.index[index - 1]]
        newData.at[dataIndex, 4] = originalColumn.at[newData.index[index - 2]]
        newData.at[dataIndex, 5] = originalColumn.at[newData.index[index + 1]]

    newData = pd.DataFrame(newData)

    newData.insert(loc = newData.shape[1], column = "Standard Deviation", value = standardDeviationColumn)
    newData = newData.iloc[newData.index[99]: , :]

    #save the file as the testing dataset -- FOR TESTING
    testDataString = "TestingData/" + uuid + ".csv"
    newData.to_csv(testDataString)


    Dictionary.add(dataset.loc[0, columnNumber], newData)
    return newData

def deleteTimeRows(timeDataframe, otherDataframe):
    otherDataframe = otherDataframe.astype('int')
    for timeIndex in range(timeDataframe.shape[0]):
        for otherIndex in otherDataframe.iloc[:, 0]:
            if(not timeIndex == otherIndex):
                timeDataframe.drop(otherIndex, inplace = True)
    return timeDataframe


