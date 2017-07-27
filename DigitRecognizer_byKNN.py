import numpy as np
import operator
import csv

def toInt(arr):
    print("toInt")
    #将arr变成mat，arr强制变为二维。 如果arr是label，label会从一维变为二维
    arr = np.mat(arr)
    m,n = np.shape(arr)
    newArr = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            newArr[i,j] = int(arr[i,j])
    return newArr

def nomalizing(arr):
    print("nomalizing")
    m,n = np.shape(arr)
    for i in range(m):
        for j in range(n):
            if arr[i,j]!=0:
                arr[i,j] = 1
    return arr

def loadTrainData():
    print("loadTrainData")
    l = []
    with open('/Users/sanshanxiashi/Desktop/kaggle/data/train.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)

    l.remove(l[0])
    l = np.array(l)
    label = l[:,0]  #42000*1
    data =  l[:,1:]
    return nomalizing(toInt(data)), toInt(label) #label 1*42000 data 42000*784

def loadTestData():
    print("loadTestData")
    l = []
    with open('/Users/sanshanxiashi/Desktop/kaggle/data/test.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    data = np.array(l)

    return nomalizing(toInt(data)) # data 28000*784


def loadTestLabel():
    print("loadTestLabel")
    l = []
    with open('/Users/sanshanxiashi/Desktop/kaggle/data/knn_benchmark.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)
    #28001*2
    l.remove(l[0])
    label = np.array(l)
    return toInt(label[:,1]) #label 28000*1

def saveResult(result):
    print("saveResult")
    with open('result.csv','w') as myFile:
        myWriter=csv.writer(myFile)
        for i in result:
            tmp=[]
            tmp.append(i)
            myWriter.writerow(tmp)

#inX:1*n    dataSet: m*n   labels:m*1
def classify(inX, dataSet, labels, k):
    print("classify")
    inX=np.mat(inX)
    dataSet=np.mat(dataSet)
    labels=np.mat(labels)
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = np.array(diffMat)**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i],0]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items() , key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def handwritingClassTest():
    print("handwritingClassTest")
    trainData, trainLabel = loadTrainData()
    testData = loadTestData()
    testLabel = loadTestLabel()
    m,n = np.shape(testData)
    errorCount = 0
    resultList = []
    for i in range(m):
        #classifierResult = np.zeros(1)
        print("i:",i)
        classifierResult = classify(testData[i], trainData, trainLabel.transpose(), 5)
        resultList.append(classifierResult)
        print("the classifier came back with: %d, the real answer is : %d" % (classifierResult, testLabel[0,i]))
        if (classifierResult != testLabel[0,i]): errorCount +=1.0
    print("\nthe total number of errors is : %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(m)))
    saveResult(resultList)

handwritingClassTest()
