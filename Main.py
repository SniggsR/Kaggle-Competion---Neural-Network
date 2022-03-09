import numpy as np
import pandas as pd
from numpy import array
# from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

trainData = pd.read_csv('trainData.csv')
testData = pd.read_csv('testData.csv')
trainDataWithOutFeatures = pd.read_csv("train.csv")
testDataWithOutFeatures = pd.read_csv('test.csv')

trainDataSeasons = array(trainData['season'])
trainDataPackage = array(trainData['package'])
trainDataSeats = array(trainData['no.seats'])
trainDataLocation = array(trainData['location'])
trainDataSection = array(trainData['section'])
trainDataPriceLevel = array(trainData['price.level'])
trainDataSubsTier = array(trainData['subscription_tier'])

testDataSeasons = array(testData['season'])
testDataPackage = array(testData['package'])
testDataSeats = array(testData['no.seats'])
testDataLocation = array(testData['location'])
testDataSection = array(testData['section'])
testDataPriceLevel = array(testData['price.level'])
testDataSubsTier = array(testData['subscription_tier'])

donations = array(trainData[['amount.donated.2013','amount.donated.lifetime','no.donations.lifetime']])

#%% Integer Encoder
label_encoder = LabelEncoder()
trainDataSeasons_intEnc = label_encoder.fit_transform(trainDataSeasons)
trainDataSeasons_intEnc = trainDataSeasons_intEnc.reshape(len(trainDataSeasons_intEnc), 1)

trainDataPackage_intEnc = label_encoder.fit_transform(trainDataPackage)
trainDataPackage_intEnc = trainDataSeasons_intEnc.reshape(len(trainDataPackage_intEnc), 1)

trainDataSeats_intEnc = label_encoder.fit_transform(trainDataSeats)
trainDataSeats_intEnc = trainDataSeats_intEnc.reshape(len(trainDataSeats_intEnc), 1)

trainDataLocation_intEnc = label_encoder.fit_transform(trainDataLocation)
trainDataLocation_intEnc = trainDataLocation_intEnc.reshape(len(trainDataLocation_intEnc), 1)

trainDataSection_intEnc = label_encoder.fit_transform(trainDataSection)
trainDataSection_intEnc = trainDataSection_intEnc.reshape(len(trainDataSection_intEnc), 1)

trainDataPriceLevel_intEnc = label_encoder.fit_transform(trainDataPriceLevel)
trainDataPriceLevel_intEnc = trainDataPriceLevel_intEnc.reshape(len(trainDataPriceLevel_intEnc), 1)

trainDataSubsTier_intEnc = label_encoder.fit_transform(trainDataSubsTier)
trainDataSubsTier_intEnc = trainDataSeasons_intEnc.reshape(len(trainDataSubsTier_intEnc), 1)



testDataSeasons_intEnc = label_encoder.fit_transform(testDataSeasons)
testDataSeasons_intEnc = testDataSeasons_intEnc.reshape(len(testDataSeasons_intEnc), 1)

testDataPackage_intEnc = label_encoder.fit_transform(testDataPackage)
testDataPackage_intEnc = testDataPackage_intEnc.reshape(len(testDataPackage_intEnc), 1)

testDataSeats_intEnc = label_encoder.fit_transform(testDataSeats)
testDataSeats_intEnc = testDataSeats_intEnc.reshape(len(testDataSeats_intEnc), 1)

testDataLocation_intEnc = label_encoder.fit_transform(testDataLocation)
testDataLocation_intEnc = testDataLocation_intEnc.reshape(len(testDataLocation_intEnc), 1)

testDataSection_intEnc = label_encoder.fit_transform(testDataSection)
testDataSection_intEnc = testDataSection_intEnc.reshape(len(testDataSection_intEnc), 1)

testDataPriceLevel_intEnc = label_encoder.fit_transform(testDataPriceLevel)
testDataPriceLevel_intEnc = testDataPriceLevel_intEnc.reshape(len(testDataPriceLevel_intEnc), 1)

testDataSubsTier_intEnc = label_encoder.fit_transform(testDataSubsTier)
testDataSubsTier_intEnc = testDataSubsTier_intEnc.reshape(len(testDataSubsTier_intEnc), 1)


#%% OneHot encoder
onehot_encoder = OneHotEncoder(sparse=False)
trainDataSeasons_oneHot = onehot_encoder.fit_transform(trainDataSeasons_intEnc)
testDataSeasons_oneHot = onehot_encoder.fit_transform(testDataSeasons_intEnc)

trainDataAccountID = array(trainData['account.id'])
trainDataAccountIDUnique = array(trainDataWithOutFeatures['account.id'])

testDataAccountID = array(testData['ID'])
testDataAccountIDUnique = array(testDataWithOutFeatures['account.id'])

#%% Combining all encoded data for training
newTrainData = []

for i in range(len(trainDataAccountIDUnique)):
    indices = np.where(trainDataAccountID == trainDataAccountIDUnique[i])
    if len(indices[0])==0:
        print('{} not found'.format(trainDataAccountID[i]))
    else:
        season = np.reshape(np.sum(trainDataSeasons_oneHot[indices],axis=0),[1,22])
        package = np.zeros([1,22])
        seats = np.zeros([1,22])
        location = np.zeros([1,22])
        section = np.zeros([1,22])
        priceLevel = np.zeros([1,22])
        subsTier = np.zeros([1,22])
        for index in indices[0]:
            nonZeroArray = np.where(trainDataSeasons_oneHot[index]==1)
            package[0,nonZeroArray] = trainDataPackage_intEnc[index]
            seats[0,nonZeroArray] = trainDataSeats_intEnc[index]
            location[0,nonZeroArray] = trainDataLocation_intEnc[index]
            section[0,nonZeroArray] = trainDataSection_intEnc[index]
            priceLevel[0,nonZeroArray] = trainDataPriceLevel_intEnc[index]
            subsTier[0,nonZeroArray] = trainDataSubsTier_intEnc[index]
        donation = donations[index:index+1,:]
        newTrainData.append(np.concatenate((season,package,seats,location,
                                                  section,priceLevel,subsTier,donation),axis=1))
newTrainData = np.reshape(array(newTrainData),[len(newTrainData),157])

#%% Combining all data for testing

newTestData = []
for i in range(len(testDataAccountIDUnique)):
    indices = np.where(testDataAccountID == testDataAccountIDUnique[i])
    if len(indices[0])==0:
        print('{} not found'.format(testDataAccountID[i]))
    else:
        season = np.reshape(np.sum(testDataSeasons_oneHot[indices],axis=0),[1,22])
        package = np.zeros([1,22])
        seats = np.zeros([1,22])
        location = np.zeros([1,22])
        section = np.zeros([1,22])
        priceLevel = np.zeros([1,22])
        subsTier = np.zeros([1,22])
        for index in indices[0]:
            nonZeroArray = np.where(testDataSeasons_oneHot[index]==1)
            package[0,nonZeroArray] = testDataPackage_intEnc[index]
            seats[0,nonZeroArray] = testDataSeats_intEnc[index]
            location[0,nonZeroArray] = testDataLocation_intEnc[index]
            section[0,nonZeroArray] = testDataSection_intEnc[index]
            priceLevel[0,nonZeroArray] = testDataPriceLevel_intEnc[index]
            subsTier[0,nonZeroArray] = testDataSubsTier_intEnc[index]
        donation = donations[index:index+1,:]
        newTestData.append(np.concatenate((season,package,seats,location,
                                                  section,priceLevel,subsTier,donation),axis=1))
newTestData = np.reshape(array(newTestData),[len(newTestData),157])

#%% Data Normalization
allData = np.concatenate((newTrainData,newTestData),axis=0)
allDataMax = np.reshape(np.max(allData,axis=0),[1,157])
allDataMax[0,np.where(allDataMax[0,:]==0)[0]]=1
allDataMin = np.reshape(np.min(allData,axis=0),[1,157])

newTrainData = (newTrainData-allDataMin)/(allDataMax-allDataMin)
newTestData = (newTestData-allDataMin)/(allDataMax-allDataMin)

#%%

import tensorflow as tf
trainInputs = newTrainData
trainOutputs = np.reshape(array(trainDataWithOutFeatures['label']),[6941,1])
# np.random.shuffle(trainOutputs)
allTrainData = np.concatenate((trainInputs,trainOutputs),axis=1)
np.random.shuffle(allTrainData)

trainInputs = allTrainData[:5000,:-1]
trainOutputs = allTrainData[:5000,-1:]

valInputs = allTrainData[5000:,:-1]
valOutputs = allTrainData[5000:,-1:]

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(1,157)),
  # tf.keras.layers.Dense(256, activation='sigmoid',kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
  # tf.keras.layers.Dense(8, activation='sigmoid',kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
   tf.keras.layers.Dense(1, activation='sigmoid',kernel_regularizer=tf.keras.regularizers.l2(0.001)),
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])

model.fit(trainInputs, trainOutputs, epochs=100,validation_split=0.2)

#%%
from sklearn import metrics

pred = model.predict(trainInputs)
pred[np.where(pred[:,0]<0.5)[0],0] = 0
pred[np.where(pred[:,0]>0.5)[0],0] = 1
cm = metrics.confusion_matrix(trainOutputs,pred)
trainAccuracy = cm[1][1]/(cm[1][0]+cm[1][1])
print('Train Accuracy = {}'.format(trainAccuracy))
cmPlot = metrics.ConfusionMatrixDisplay(cm)
cmPlot.plot()

pred = model.predict(valInputs)
pred[np.where(pred[:,0]<0.5)[0],0] = 0
pred[np.where(pred[:,0]>0.5)[0],0] = 1
cm = metrics.confusion_matrix(valOutputs,pred)
valAccuracy = cm[1][1]/(cm[1][0]+cm[1][1])
print('Validation Accuracy = {}'.format(valAccuracy))
cmPlot = metrics.ConfusionMatrixDisplay(cm)
cmPlot.plot()

testPred = model.predict(newTestData)
