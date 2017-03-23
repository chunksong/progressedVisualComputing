import numpy
from numpy import linalg as LA
import matplotlib.pyplot as plt

InputTrainingFile = open("Iris_train.dat","r")

SetoSepalLength = []
SetoSepalWidth = []

VersSepalLength = []
VersSepalWidth = []

VirgSepalLength = []
VirgSepalWidth = []

SetoAttrMean = []
VersAttrMean = []
VirgAttrMean = []

SetoCovMat = numpy.matrix([[0,0],[0,0]])
VersCovMat = numpy.matrix([[0,0],[0,0]])
VirgCovMat = numpy.matrix([[0,0],[0,0]])

while True:
	TrainingLine = InputTrainingFile.readline()
	if not TrainingLine: break

	SplitLine = TrainingLine.split()
	if SplitLine == []:
		continue	
	
	if SplitLine[4] == '1':			# Setosa
		SetoSepalLength.append(float(SplitLine[0]))
		SetoSepalWidth.append(float(SplitLine[1]))
		plt.scatter(SetoSepalLength[-1],SetoSepalWidth[-1],marker = ".",color = "r")
		
	elif SplitLine[4] == '2':			#Versicolor
		VersSepalLength.append(float(SplitLine[0]))
		VersSepalWidth.append(float(SplitLine[1]))
		plt.scatter(VersSepalLength[-1],VersSepalWidth[-1],marker = ".",color = "b")
		
	elif SplitLine[4] == '3':			#Virginica
		VirgSepalLength.append(float(SplitLine[0]))
		VirgSepalWidth.append(float(SplitLine[1]))
		plt.scatter(VirgSepalLength[-1],VirgSepalWidth[-1],marker = ".",color = "y")

	else:
		continue




SetoAttrMean.append(numpy.mean(SetoSepalLength))
SetoAttrMean.append(numpy.mean(SetoSepalWidth))
plt.scatter(SetoAttrMean[0],SetoAttrMean[1], marker = "o",color = "k")

VersAttrMean.append(numpy.mean(VersSepalLength))
VersAttrMean.append(numpy.mean(VersSepalWidth))
plt.scatter(VersAttrMean[0],VersAttrMean[1], marker = "o",color = "k")

VirgAttrMean.append(numpy.mean(VirgSepalLength))
VirgAttrMean.append(numpy.mean(VirgSepalWidth))
plt.scatter(VirgAttrMean[0],VirgAttrMean[1], marker = "o",color = "k")

SetoMeanVec = numpy.matrix([SetoAttrMean[0],SetoAttrMean[1]])
VersMeanVec = numpy.matrix([VersAttrMean[0],VersAttrMean[1]])
VirgMeanVec = numpy.matrix([VirgAttrMean[0],VirgAttrMean[1]])

print("Mean Vector :")
print("Setosa : ", SetoMeanVec)
print("Versicolor : ", VersMeanVec)
print("Virginica : ", VirgMeanVec)
print()



for iterator in range(0,len(SetoSepalLength)):
	SetoTempVec = numpy.matrix([SetoSepalLength[iterator],SetoSepalWidth[iterator]])
	VersTempVec = numpy.matrix([VersSepalLength[iterator],VersSepalWidth[iterator]])
	VirgTempVec = numpy.matrix([VirgSepalLength[iterator],VirgSepalWidth[iterator]])
	
	SetoCovMat = numpy.add(SetoCovMat,(SetoTempVec - SetoMeanVec).T.dot((SetoTempVec - SetoMeanVec)))
	VersCovMat = numpy.add(VersCovMat,(VersTempVec - VersMeanVec).T.dot((VersTempVec - VersMeanVec)))
	VirgCovMat = numpy.add(VirgCovMat,(VirgTempVec - VirgMeanVec).T.dot((VirgTempVec - VirgMeanVec)))




SetoCovMat /= (len(SetoSepalLength) - 1)
VersCovMat /= (len(VersSepalLength) - 1)
VirgCovMat /= (len(VirgSepalLength) - 1)

print("Setsa Covariance Matrix : ")
print(SetoCovMat)

print("Versicolor Covariance Matrix : ")
print(VersCovMat)

print("Virginica Covariance Matrix : ")
print(VirgCovMat)
print()

x = numpy.arange(0,8,0.01)
y = numpy.arange(0,5,0.01)
X,Y = numpy.meshgrid(x,y)

F = LA.inv(SetoCovMat)[0,0]*(X-SetoMeanVec[0,0])*(X-SetoMeanVec[0,0]) + (LA.inv(SetoCovMat)[0,1] + LA.inv(SetoCovMat)[1,0])*(X-SetoMeanVec[0,0])*(Y-SetoMeanVec[0,1]) + LA.inv(SetoCovMat)[1,1]*(Y-SetoMeanVec[0,1])*(Y-SetoMeanVec[0,1])
plt.contour(X, Y, F-2, [0],colors ='r')
G = LA.inv(VersCovMat)[0,0]*(X-VersMeanVec[0,0])*(X-VersMeanVec[0,0]) + (LA.inv(VersCovMat)[0,1] + LA.inv(VersCovMat)[1,0])*(X-VersMeanVec[0,0])*(Y-VersMeanVec[0,1]) + LA.inv(VersCovMat)[1,1]*(Y-VersMeanVec[0,1])*(Y-VersMeanVec[0,1])
plt.contour(X, Y, G-2, [0],colors ='b')
H = LA.inv(VirgCovMat)[0,0]*(X-VirgMeanVec[0,0])*(X-VirgMeanVec[0,0]) + (LA.inv(VirgCovMat)[0,1] + LA.inv(VirgCovMat)[1,0])*(X-VirgMeanVec[0,0])*(Y-VirgMeanVec[0,1]) + LA.inv(VirgCovMat)[1,1]*(Y-VirgMeanVec[0,1])*(Y-VirgMeanVec[0,1])
plt.contour(X, Y, H-2, [0],colors ='y')

plt.contour(X,Y, F - G, [0])
plt.contour(X,Y, G - H, [0])
plt.contour(X,Y, H - F, [0])


for iterator in range(0, len(SetoSepalLength)):
	SetoTempVec = numpy.matrix([SetoSepalLength[iterator], SetoSepalWidth[iterator]])
	VersTempVec = numpy.matrix([VersSepalLength[iterator], VersSepalWidth[iterator]])
	VirgTempVec = numpy.matrix([VirgSepalLength[iterator], VirgSepalWidth[iterator]])

InputTrainingFile.close()

InputTestFile = open("Iris_test.dat", "r")

ConfusionMat = numpy.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

while True:
	TestingLine = InputTestFile.readline()
	if not TestingLine: break

	SplitLine = TestingLine.split()
	if SplitLine == []:
		continue

	TestVec = numpy.matrix([float(SplitLine[0]), float(SplitLine[1])])

	SetoDistance = LA.inv(SetoCovMat)[0,0]*(TestVec[0,0]-SetoMeanVec[0,0])*(TestVec[0,0]-SetoMeanVec[0,0]) + (LA.inv(SetoCovMat)[0,1] + LA.inv(SetoCovMat)[1,0])*(TestVec[0,0]-SetoMeanVec[0,0])*(TestVec[0,1]-SetoMeanVec[0,1]) + LA.inv(SetoCovMat)[1,1]*(TestVec[0,1]-SetoMeanVec[0,1])*(TestVec[0,1]-SetoMeanVec[0,1])
	VersDistance = LA.inv(VersCovMat)[0,0]*(TestVec[0,0]-VersMeanVec[0,0])*(TestVec[0,0]-VersMeanVec[0,0]) + (LA.inv(VersCovMat)[0,1] + LA.inv(VersCovMat)[1,0])*(TestVec[0,0]-VersMeanVec[0,0])*(TestVec[0,1]-VersMeanVec[0,1]) + LA.inv(VersCovMat)[1,1]*(TestVec[0,1]-VersMeanVec[0,1])*(TestVec[0,1]-VersMeanVec[0,1])
	VirgDistance = LA.inv(VirgCovMat)[0,0]*(TestVec[0,0]-VirgMeanVec[0,0])*(TestVec[0,0]-VirgMeanVec[0,0]) + (LA.inv(VirgCovMat)[0,1] + LA.inv(VirgCovMat)[1,0])*(TestVec[0,0]-VirgMeanVec[0,0])*(TestVec[0,1]-VirgMeanVec[0,1]) + LA.inv(VirgCovMat)[1,1]*(TestVec[0,1]-VirgMeanVec[0,1])*(TestVec[0,1]-VirgMeanVec[0,1])

	SetoDF = SetoDistance - VersDistance
	VersDF = VersDistance - VirgDistance
	VirgDF = VirgDistance - SetoDistance

	if SetoDistance > VersDistance:
		if VersDistance > VirgDistance:
			ConfusionMat[int(SplitLine[4]) - 1, 2] += 1
			if(int(SplitLine[4]) == 3):
				plt.scatter(TestVec[0,0], TestVec[0,1], marker="*", color="y")
			else:
				plt.scatter(TestVec[0,0], TestVec[0, 1], marker="+", color="k")

		else:
			ConfusionMat[int(SplitLine[4]) - 1, 1] += 1
			if(int(SplitLine[4]) == 2):
				plt.scatter(TestVec[0,0], TestVec[0,1], marker="*", color="b")
			else:
				plt.scatter(TestVec[0,0], TestVec[0, 1], marker="+", color="k")

	else:
		if SetoDistance > VirgDistance:
			ConfusionMat[int(SplitLine[4]) - 1, 2] += 1
			if(int(SplitLine[4]) == 3):
				plt.scatter(TestVec[0,0], TestVec[0,1], marker="*", color="y")
			else:
				plt.scatter(TestVec[0,0], TestVec[0, 1], marker="+", color="k")

		else:
			ConfusionMat[int(SplitLine[4]) - 1, 0] += 1
			if(int(SplitLine[4]) == 1):
				plt.scatter(TestVec[0,0], TestVec[0,1], marker="*", color="r")
			else:
				plt.scatter(TestVec[0,0], TestVec[0, 1], marker="+", color="k")

print("Confusion Matrix :")
print(ConfusionMat)

InputTestFile.close()

plt.show()
