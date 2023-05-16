# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
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

# Object of 1st Ensemble Model
ensemble_1 = LogisticRegression(random_state=0)

# Object of 2nd Ensemble Model
ensemble_2 = DecisionTreeClassifier(random_state=0)

# Object of 3rd Ensemble Model
ensemble_3 = GaussianNB()

# Training the Classifier using Voting_Classifier
Classifier = VotingClassifier(estimators=[ ('Logistic_Regression', ensemble_1), ('Decision_Tree', ensemble_2), ('Gaussian_NB', ensemble_3)], voting='soft')
Classifier.fit(X_train,y_train)

# Prediction on X_Test 
Prediction_X_Test = Classifier.predict(X_test)

# Confusion Matrix for Classification
skplt.metrics.plot_confusion_matrix(y_test, Prediction_X_Test, normalize=False)

# Normalized Confusion Matrix
skplt.metrics.plot_confusion_matrix(y_test, Prediction_X_Test, normalize=True)

# Accuracy of Classifier
Accuracy = accuracy_score(y_test, Prediction_X_Test)*100
print("Accuracy score of Classifier is : ", Accuracy ,"%")

# Classification Report for Voting Classifier
print(f'Classification Report:\n{classification_report(y_test, Prediction_X_Test)}')

# Training Score Of Classifier
trainscore = Classifier.score(X_train, y_train)
print("Training Score is : ", trainscore)

# Testing Score Of Classifier
testscore = Classifier.score(X_test, y_test)
print("Testing Score is : ", testscore)

# Training the Model for 1st Ensemble Model
ensemble_1.fit(X_train,y_train)

# Training the Model for 2nd Ensemble Model
ensemble_2.fit(X_train,y_train)

# Training the Model for 3rd Ensemble Model
ensemble_3.fit(X_train,y_train)

# Prediction on X_Test  for 1st Ensemble Model
ensemble_1_pred1 = ensemble_1.predict(X_test)

# Prediction on X_Test  for 2nd Ensemble Model
ensemble_2_pred1 = ensemble_1.predict(X_test)

# Prediction on X_Test  for 3rd Ensemble Model
ensemble_3_pred1 = ensemble_1.predict(X_test)

# K-Fold Cross-Validation 
CV = KFold(n_splits=4, random_state=1, shuffle=True)

# Evaluate model using K-Fold Cross-Validation 
Scores = cross_val_score(Classifier, X, y, scoring='accuracy', cv=CV, n_jobs=-1)

# Evaluting the Performance of  K-Fold Cross-Validation 
print("%0.2f accuracy with a standard deviation of %0.2f" % (Scores.mean(), Scores.std()))

# Predict the class probabilities of the test set using the voting classifier
y_proba = Classifier.predict_proba(X_test)

# Compute the ROC curve and AUC for each class
fpr = {}
tpr = {}
roc_auc = {}
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(y_test, y_proba[:, i], pos_label=i)
    roc_auc[i] = roc_auc_score(y_test, y_proba[:, i])

# Plot the ROC curves for all classes
plt.figure()
for i in range(2):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Voting Classifier')
plt.legend(loc="lower right")
plt.show()