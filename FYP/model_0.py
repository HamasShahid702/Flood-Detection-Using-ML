# Importing the Libraries
import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import scikitplot as skplt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Importing the Dataset
dataset = pd.read_csv('kerala (1).csv')

# Separating the Dependent & Independent Variables
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

# Label Encoding of Dependent Variables
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Splitting the Dataset
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.20,random_state=1)

# Feature Scaling of Dataset
Standard_Scaler = StandardScaler()
X_train = Standard_Scaler.fit_transform(X_train)
X_test = Standard_Scaler.transform(X_test)

# Training Artificial Neural Network Model as Classifier
ANN = tf.keras.models.Sequential()

# Adding Input Layer of Classifier
ANN.add(tf.keras.layers.Dense(units=10,activation='relu'))

# Adding Hidden Layer of Classifier
ANN.add(tf.keras.layers.Dense(units=10,activation='relu'))

# Adding Output Layer of Classifier
ANN.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

# Compiling Artificial Neural Network Classifier
ANN.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Fit Artificial Neural Network Classifier
history = ANN.fit(X_train,y_train,batch_size=10,epochs=1000)

# Prediction on X_Test 
Prediction_X_Test = ANN.predict(X_test)
Prediction_X_Test = (Prediction_X_Test>0.5)

# Accuracy of Artificial Neural Network Classifier
Accuracy = accuracy_score(y_test, Prediction_X_Test)*100
print("Accuracy score of Classifier is : ", Accuracy ,"%")

# Classification Report for Artifical Neural Network Classifier
print(f'Classification Report:\n{classification_report(y_test, Prediction_X_Test)}')

# Plot the Loss Curve
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
# Plot the Accuracy Curve
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

# Confusion Matrix for Artificial Neural Network Classification
skplt.metrics.plot_confusion_matrix(y_test, Prediction_X_Test, normalize=False)

# Normalized Confusion Matrix
skplt.metrics.plot_confusion_matrix(y_test, Prediction_X_Test, normalize=True)

