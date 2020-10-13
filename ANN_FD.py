#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 23:55:52 2020

@author: Dicken
"""

# Importer les librairies nécessaires 
import matplotlib.pyplot as plt # visualiser des données sous formes de graphiques
import numpy as np # manipuler des matrices ou tableaux 
import pandas as pd # analyse, nettoyage et préparation des données
import seaborn as sns # création de tracés statistiques communs
#scikit-learn
import sklearn
# Importation dataset
#data = pd.read_csv('bank-full.csv', sep=';') // Premier dataset avec 45.000 observations
data = pd.read_csv('bank.csv')
data.rename(columns={'deposit': 'y'}, inplace=True)

# Variables independantes
X = data.iloc[:, :-1].values

# Variables dependantes
y = data.iloc[:, -1].values

# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from  sklearn.compose import ColumnTransformer

# Variable independante
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Variables dependantes a coder en 0 et 1

# Variable default
labelencoder_X_default = LabelEncoder()
X[:, 4] = labelencoder_X_default.fit_transform(X[:, 4])

# Variable housing
labelencoder_X_housing = LabelEncoder()
X[:, 6] = labelencoder_X_housing.fit_transform(X[:, 6])

# Variable loan
labelencoder_X_loan = LabelEncoder()
X[:, 7] = labelencoder_X_loan.fit_transform(X[:, 7])

# Variables a coder en plusieurs
# Utilisation OneHotEncoder() et columnTransform
labelencoder_X_job = LabelEncoder()
X[:, 1] = labelencoder_X_job.fit_transform(X[:, 1])

# Utilisation OneHotEncoder() et columnTransform
labelencoder_X_marital = LabelEncoder()
X[:, 2] = labelencoder_X_marital.fit_transform(X[:, 2])

labelencoder_X_education = LabelEncoder()
X[:, 3] = labelencoder_X_education.fit_transform(X[:, 3])

labelencoder_X_contact = LabelEncoder()
X[:, 8] = labelencoder_X_education.fit_transform(X[:, 8])

labelencoder_X_month = LabelEncoder()
X[:, 10] = labelencoder_X_education.fit_transform(X[:, 10])

labelencoder_X_poutcome = LabelEncoder()
X[:, 15] = labelencoder_X_education.fit_transform(X[:, 15])

#t = X.select_dtypes(exclude=['int64'])


columtransform =  ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [1, 2, 3, 8, 10, 15])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)

X = np.array(columtransform.fit_transform(X), dtype=np.float)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print (X_train)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialiser le reseau
classifier = Sequential()

classifier.add(Dense(units=8, activation="relu", kernel_initializer="uniform", input_dim=48))

# deuxieme couche cachée
classifier.add(Dense(units=8, activation="relu", kernel_initializer="uniform"))

# Couche de sortie
classifier.add(Dense(units=1, activation="sigmoid", kernel_initializer="uniform"))

# Compiler le reseau le neurone
classifier.compile(optimizer='adam', loss="binary_crossentropy", metrics=["accuracy"])

# resume du model
print(classifier.summary())

# Entrainer le reseaux de neurone
classifier.fit(X_train, y_train, batch_size=10, epochs=1000)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Classifier avec un seuil de 50%
y_pred= (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
