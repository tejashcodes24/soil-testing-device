# %%
# Importing libraries

from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')

# %%
df = pd.read_csv('../dataset/crop_recommendation.csv')

# %%
df.head()

# %%
df.tail()

# %%
df.size

# %%
df.shape

# %%
df.columns

# %%
df['label'].unique()

# %%
df.dtypes

# %%
df['label'].value_counts()

# %%
sns.heatmap(df.corr(),annot=True)

# %% [markdown]
# ### Seperating features and target label

# %%
features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']
#features = df[['temperature', 'humidity', 'ph', 'rainfall']]
labels = df['label']

# %%
# Initialzing empty lists to append all model's name and corresponding name
acc = []
model = []

# %%
# Splitting into train and test data

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)

# %% [markdown]
# # Decision Tree

# %%
from sklearn.tree import DecisionTreeClassifier

DecisionTree = DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)

DecisionTree.fit(Xtrain,Ytrain)

predicted_values = DecisionTree.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Decision Tree')
print("DecisionTrees's Accuracy is: ", x*100)

print(classification_report(Ytest,predicted_values))

# %%
from sklearn.model_selection import cross_val_score

# %%
# Cross validation score (Decision Tree)
score = cross_val_score(DecisionTree, features, target,cv=5)

# %%
score

# %% [markdown]
# ### Saving trained Decision Tree model

# %%
import pickle
# Dump the trained Naive Bayes classifier with Pickle
DT_pkl_filename = '../dataset/DecisionTree.pkl'
# Open the file to save as pkl file
DT_Model_pkl = open(DT_pkl_filename, 'wb')
pickle.dump(DecisionTree, DT_Model_pkl)
# Close the pickle instances
DT_Model_pkl.close()

# %% [markdown]
# # Guassian Naive Bayes

# %%
from sklearn.naive_bayes import GaussianNB

NaiveBayes = GaussianNB()

NaiveBayes.fit(Xtrain,Ytrain)

predicted_values = NaiveBayes.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Naive Bayes')
print("Naive Bayes's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))

# %%
# Cross validation score (NaiveBayes)
score = cross_val_score(NaiveBayes,features,target,cv=5)
score

# %% [markdown]
# ### Saving trained Guassian Naive Bayes model

# %%
import pickle
# Dump the trained Naive Bayes classifier with Pickle
NB_pkl_filename = '../dataset/NBClassifier.pkl'
# Open the file to save as pkl file
NB_Model_pkl = open(NB_pkl_filename, 'wb')
pickle.dump(NaiveBayes, NB_Model_pkl)
# Close the pickle instances
NB_Model_pkl.close()

# %% [markdown]
# # Support Vector Machine (SVM)

# %%
from sklearn.svm import SVC
# data normalization with sklearn
from sklearn.preprocessing import MinMaxScaler
# fit scaler on training data
norm = MinMaxScaler().fit(Xtrain)
X_train_norm = norm.transform(Xtrain)
# transform testing dataabs
X_test_norm = norm.transform(Xtest)
SVM = SVC(kernel='poly', degree=3, C=1)
SVM.fit(X_train_norm,Ytrain)
predicted_values = SVM.predict(X_test_norm)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('SVM')
print("SVM's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))

# %%
# Cross validation score (SVM)
score = cross_val_score(SVM,features,target,cv=5)
score

# %%
#Saving trained SVM model

# %%
import pickle
# Dump the trained SVM classifier with Pickle
SVM_pkl_filename = '../dataset/SVMClassifier.pkl'
# Open the file to save as pkl file
SVM_Model_pkl = open(SVM_pkl_filename, 'wb')
pickle.dump(SVM, SVM_Model_pkl)
# Close the pickle instances
SVM_Model_pkl.close()

# %% [markdown]
# # Logistic Regression

# %%
from sklearn.linear_model import LogisticRegression

LogReg = LogisticRegression(random_state=2)

LogReg.fit(Xtrain,Ytrain)

predicted_values = LogReg.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Logistic Regression')
print("Logistic Regression's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))

# %%
# Cross validation score (Logistic Regression)
score = cross_val_score(LogReg,features,target,cv=5)
score

# %% [markdown]
# ### Saving trained Logistic Regression model

# %%
import pickle
# Dump the trained Naive Bayes classifier with Pickle
LR_pkl_filename = '../dataset/LogisticRegression.pkl'
# Open the file to save as pkl file
LR_Model_pkl = open(DT_pkl_filename, 'wb')
pickle.dump(LogReg, LR_Model_pkl)
# Close the pickle instances
LR_Model_pkl.close()

# %% [markdown]
# # Random Forest

# %%
from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain,Ytrain)

predicted_values = RF.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('RF')
print("RF's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))

# %%
# Cross validation score (Random Forest)
score = cross_val_score(RF,features,target,cv=5)
score

# %% [markdown]
# ### Saving trained Random Forest model

# %%
import pickle
# Dump the trained Naive Bayes classifier with Pickle
RF_pkl_filename = '../dataset/RandomForest.pkl'
# Open the file to save as pkl file
RF_Model_pkl = open(RF_pkl_filename, 'wb')
pickle.dump(RF, RF_Model_pkl)
# Close the pickle instances
RF_Model_pkl.close()

# %% [markdown]
# # XGBoost

# %%
%pip install xgboost

# %%
# Import necessary libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import xgboost as xgb

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the Ytrain and Ytest labels
Ytrain_encoded = label_encoder.fit_transform(Ytrain)
Ytest_encoded = label_encoder.transform(Ytest)

# Initialize and train the XGBoost Classifier
XB = xgb.XGBClassifier()
XB.fit(Xtrain, Ytrain_encoded)

# Predict the values for Xtest
predicted_values = XB.predict(Xtest)

# Calculate the accuracy score
x = metrics.accuracy_score(Ytest_encoded, predicted_values)
acc.append(x)
model.append('XGBoost')

# Print the accuracy
print("XGBoost's Accuracy is: ", x)

# Print the classification report with the actual and predicted labels
print(classification_report(Ytest_encoded, predicted_values, target_names=label_encoder.classes_))


# %%
# Cross validation score (XGBoost)
from sklearn.preprocessing import LabelEncoder

# Create an instance of LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the target variable (which contains string labels)
encoded_target = label_encoder.fit_transform(target)

# Use the encoded target for cross-validation
score = cross_val_score(XB, features, encoded_target, cv=5)
print(score)


# %% [markdown]
# ### Saving trained XGBoost model

# %%
import pickle
# Dump the trained Naive Bayes classifier with Pickle
XB_pkl_filename = '../dataset/XGBoost.pkl'
# Open the file to save as pkl file
XB_Model_pkl = open(XB_pkl_filename, 'wb')
pickle.dump(XB, XB_Model_pkl)
# Close the pickle instances
XB_Model_pkl.close()

# %% [markdown]
# ## Accuracy Comparison

# %%
plt.figure(figsize=[10,5],dpi = 100)
plt.title('Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Algorithm')
sns.barplot(x = acc,y = model,palette='dark')

# %%
accuracy_models = dict(zip(model, acc))
for k, v in accuracy_models.items():
    print (k, '-->', v)

# %% [markdown]
# ## Making a prediction

# %%
data = np.array([[104,18, 30, 23.603016, 60.3, 6.7, 140.91]])
prediction = RF.predict(data)
print(prediction)

# %%
data = np.array([[83, 45, 60, 28, 70.3, 7.0, 150.9]])
prediction = RF.predict(data)
print(prediction)

# %%
data = np.array([[90, 42, 43, 20, 82, 6.5, 202]])
prediction = RF.predict(data)
print(prediction)

# %%
data = np.array([[76, 51, 18, 26, 71, 6, 79]])
prediction = RF.predict(data)
print(prediction)


