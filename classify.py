import numpy
from numpy import linalg as LA

InputTrainingFile = open("Iris_train.dat","r")

SetoSepalLength = []
SetoSepalWidth = []
SetoPetalLength = []
SetoPetalWidth = []

VersSepalLength = []
VersSepalWidth = []
VersPetalLength = []
VersPetalWidth = []

VirgSepalLength = []
VirgSepalWidth = []
VirgPetalLength = []
VirgPetalWidth = []

SetoAttrMean = []
VersAttrMean = []
VirgAttrMean = []

SetoCovMat = numpy.matrix([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
VersCovMat = numpy.matrix([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
VirgCovMat = numpy.matrix([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])

while True:
	TrainingLine = InputTrainingFile.readline()
	if not TrainingLine: break

	SplitLine = TrainingLine.split()
	if SplitLine == []:
		continue	
	
	if SplitLine[4] == '1':			# Setosa
		SetoSepalLength.append(float(SplitLine[0]))
		SetoSepalWidth.append(float(SplitLine[1]))
		SetoPetalLength.append(float(SplitLine[2]))
		SetoPetalWidth.append(float(SplitLine[3]))

	elif SplitLine[4] == '2':			#Versicolor
		VersSepalLength.append(float(SplitLine[0]))
		VersSepalWidth.append(float(SplitLine[1]))
		VersPetalLength.append(float(SplitLine[2]))
		VersPetalWidth.append(float(SplitLine[3]))

	elif SplitLine[4] == '3':			#Virginica
		VirgSepalLength.append(float(SplitLine[0]))
		VirgSepalWidth.append(float(SplitLine[1]))
		VirgPetalLength.append(float(SplitLine[2]))
		VirgPetalWidth.append(float(SplitLine[3]))
	else:
		continue

SetoAttrMean.append(numpy.mean(SetoSepalLength))
SetoAttrMean.append(numpy.mean(SetoSepalWidth))
SetoAttrMean.append(numpy.mean(SetoPetalLength))
SetoAttrMean.append(numpy.mean(SetoPetalWidth))

VersAttrMean.append(numpy.mean(VersSepalLength))
VersAttrMean.append(numpy.mean(VersSepalWidth))
VersAttrMean.append(numpy.mean(VersPetalLength))
VersAttrMean.append(numpy.mean(VersPetalWidth))

VirgAttrMean.append(numpy.mean(VirgSepalLength))
VirgAttrMean.append(numpy.mean(VirgSepalWidth))
VirgAttrMean.append(numpy.mean(VirgPetalLength))
VirgAttrMean.append(numpy.mean(VirgPetalWidth))

SetoMeanVec = numpy.matrix([SetoAttrMean[0],SetoAttrMean[1],SetoAttrMean[2],SetoAttrMean[3]])
VersMeanVec = numpy.matrix([VersAttrMean[0],VersAttrMean[1],VersAttrMean[2],VersAttrMean[3]])
VirgMeanVec = numpy.matrix([VirgAttrMean[0],VirgAttrMean[1],VirgAttrMean[2],VirgAttrMean[3]])

print("Mean Vector :")
print("Setosa : ", SetoMeanVec)
print("Versicolor : ", VersMeanVec)
print("Virginica : ", VirgMeanVec)
print()


for iterator in range(0,len(SetoSepalLength)):
	SetoTempVec = numpy.matrix([SetoSepalLength[iterator],SetoSepalWidth[iterator],SetoPetalLength[iterator],SetoPetalWidth[iterator]])
	VersTempVec = numpy.matrix([VersSepalLength[iterator],VersSepalWidth[iterator],VersPetalLength[iterator],VersPetalWidth[iterator]])
	VirgTempVec = numpy.matrix([VirgSepalLength[iterator],VirgSepalWidth[iterator],VirgPetalLength[iterator],VirgPetalWidth[iterator]])
	
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

InputTrainingFile.close()

InputTestFile = open("Iris_test.dat","r")

ConfusionMat = numpy.matrix([[0,0,0],[0,0,0],[0,0,0]])

while True:
	TestingLine = InputTestFile.readline()
	if not TestingLine: break
	
	SplitLine = TestingLine.split()
	if SplitLine == []:
		continue	

	TestVec = numpy.matrix([float(SplitLine[0]),float(SplitLine[1]),float(SplitLine[2]),float(SplitLine[3])])
	
	SetoDistanceVec = (LA.inv(SetoCovMat)).dot((TestVec - SetoMeanVec).T)	
	VersDistanceVec = (LA.inv(VersCovMat)).dot((TestVec - VersMeanVec).T)	
	VirgDistanceVec = (LA.inv(VirgCovMat)).dot((TestVec - VirgMeanVec).T)	
	SetoNorm = LA.norm(SetoDistanceVec)
	VersNorm = LA.norm(VersDistanceVec)
	VirgNorm = LA.norm(VirgDistanceVec)

	SetoDF = SetoNorm - VersNorm
	VersDF = VersNorm - VirgNorm
	VirgDF = VirgNorm - SetoNorm

	if SetoNorm > VersNorm:
		if VersNorm > VirgNorm:
			ConfusionMat[int(SplitLine[4])-1,2] += 1
		else:
			ConfusionMat[int(SplitLine[4])-1,1] += 1
	else:
		if SetoNorm > VirgNorm:
			ConfusionMat[int(SplitLine[4])-1,2] += 1
		else:
			ConfusionMat[int(SplitLine[4])-1,0] += 1

print("Confusion Matrix :")
print(ConfusionMat)

InputTestFile.close()
