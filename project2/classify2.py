import numpy
import matplotlib.pyplot as plt

def Percep(DataMat,LearningRate,weightVec):

    x = numpy.arange(-8, 10, 0.01)
    y = numpy.arange(-8, 10, 0.01)
    X, Y = numpy.meshgrid(x, y)

    while True:
        resultMat = weightVec.T.dot(DataMat) < 0
        if resultMat.sum() == 0:
            break
        weightVec += DataMat[:, resultMat.A1].sum(axis=1) * LearningRate

    print("weight vector is \n",weightVec)

    F = numpy.multiply(weightVec[1][0], X) + numpy.multiply(weightVec[2][0], Y)
    plt.contour(X, Y, F + weightVec[0][0], [0], colors='r')


def Relax(Margin,DataMat,LearningRate,weightVec):

    x = numpy.arange(-8, 10, 0.01)
    y = numpy.arange(-8, 10, 0.01)
    X, Y = numpy.meshgrid(x, y)
    i = 0

    while i < 1000:
        resultMat = weightVec.T.dot(DataMat) <= Margin
        if resultMat.sum() == 0:
            break
        temp = ((Margin - weightVec.T.dot(DataMat[:, resultMat.A1]))/
                ((numpy.linalg.norm(DataMat[:, resultMat.A1]))**2)).sum(axis =1)* LearningRate
        weightVec += numpy.multiply(temp,(DataMat[:, resultMat.A1].sum(axis=1)))
        i += 1

    print("weight vector is \n", weightVec)
    G = numpy.multiply(weightVec[1][0], X) + numpy.multiply(weightVec[2][0], Y)
    plt.contour(X, Y, G + weightVec[0][0], [0], colors='b')

def LMS(Margin,DataMat,LearningRate,weightVec,threshold):

    x = numpy.arange(-8, 10, 0.01)
    y = numpy.arange(-8, 10, 0.01)
    X, Y = numpy.meshgrid(x, y)
    iter = 0
    while iter < 20:
        resultMat = ((DataMat[:,[iter,iter+1]])*
                     (Margin - weightVec.T.dot(DataMat[:,[iter,iter+1]])).T).sum(axis= 1) * LearningRate
        if resultMat.T.sum(axis=1) >= threshold:
            break
        weightVec += resultMat
        iter += 2

    print("weight vector is \n",weightVec)
    H = numpy.multiply(weightVec[1][0], X) + numpy.multiply(weightVec[2][0], Y)
    plt.contour(X, Y, H + weightVec[0][0], [0], colors='Y')





InputDataFile = open("data.txt", "r")

w1x1 = []
w1x2 = []

w2x1 = []
w2x2 = []

w3x1 = []
w3x2 = []

while True:
    DataLine = InputDataFile.readline()
    if not DataLine: break

    SplitLine = DataLine.split()
    if SplitLine == []:
        continue

    w1x1.append(float(SplitLine[0]))
    w1x2.append(float(SplitLine[1]))
    w2x1.append(float(SplitLine[2]))
    w2x2.append(float(SplitLine[3]))
    w3x1.append(float(SplitLine[4]))
    w3x2.append(float(SplitLine[5]))

InputDataFile.close()

weightVec = numpy.matrix([[1.0],[1.0],[-1.0]])
DataList = []

for iterator in range(0, len(w1x1)):
    tempList1 = [1,w1x1[iterator],w1x2[iterator]]
    plt.scatter(w1x1[iterator], w1x2[iterator], marker="+", color="r")

    tempList2 = [-1,-1*w2x1[iterator],-1*w2x2[iterator]]
    plt.scatter(w2x1[iterator], w2x2[iterator], marker="+", color="b")

    DataList.append(tempList1)
    DataList.append(tempList2)

DataMat = numpy.matrix(DataList)
DataMat = DataMat.T
LearningRate = 0.0095

#Percep(DataMat,LearningRate,weightVec)

DataList1 = []
for iterator in range(0, len(w1x1)):
    tempList1 = [1,w1x1[iterator],w1x2[iterator]]
    plt.scatter(w1x1[iterator], w1x2[iterator], marker="+", color="r")

    tempList2 = [-1,-1*w3x1[iterator],-1*w3x2[iterator]]
    plt.scatter(w3x1[iterator], w3x2[iterator], marker="+", color="g")

    DataList1.append(tempList1)
    DataList1.append(tempList2)

DataMat1 = numpy.matrix(DataList1)
DataMat1 = DataMat1.T

Margin = 0.5
LearningRate = 0.01
weightVec = numpy.matrix([[0.0],[0.0],[0.0]])

#Relax(Margin,DataMat1,LearningRate,weightVec)

Margin = 0.5
LearningRate = 0.00095
weightVec = numpy.matrix([[1.0],[1.0],[-1.0]])
threshold = 0.1

LMS(Margin,DataMat,LearningRate,weightVec,threshold)

plt.show()


