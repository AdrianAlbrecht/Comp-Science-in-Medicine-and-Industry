#Binary classifier
# Classifiers = {Support Vector Machine with Kernels = {linear, polynomial, radial}, Decison Tree {C 4.5, C5.0}, Artificial Neural Network (ANN),
# Rule- based classifier {Lem2, Covering, Exhaustive}, Random forest, k Nearest Neighbours (knn), metrics = {Euclidease, Manhattan, Cabocca, Minkowski: power =3, Epsilon max-min,
# hamminga, Chebyszev}, Naive Bayes Classifiers (NB), Classifiere Committree, Ada- Boost}
###########################################################################################
# Models: Train&Test, Cross-validation, Internal Cross-validation, Leave one out, Monte Carlo Cross-validation (Multipt T&T)
###########################################################################################
# 1) Find proper data
# 2) Apply preporcessing:
# For numerical data:
# - normalization,
# - standarization,
# - discretization if needed.
# For symbolic data:
# - create Dummy vairables if needed,
# - missing values absorption.
# 3) Split data according to model Cross-validation
# 4) Learn model
# 5) Create confusion matrix
# 6) Compute Accuracy, Balanced accuracy, Coverage, precision, recall, F1 Score
# 7) If possible apply ROC, PR-curve, G-Mean,
# 8) Compare with random (.5)
# 9) Create raport
# 10) Present (posible to do it via MS Teams)
#############################################################################################
# 6 points for 3, 7 => 3.5, 8 => 4, 9 => 4.5, 10 => 5
# Random forest with model Cross-validation


# importing required libraries
# # importing Scikit-learn library and datasets package
from sklearn import datasets
# # Loading the iris plants dataset 
iris = datasets.load_iris()
# dividing the datasets into two parts i.e. training datasets and test datasets
X, y = datasets.load_iris( return_X_y = True)
# Splitting arrays or matrices into random train and test subsets
from sklearn.model_selection import train_test_split
# i.e. 70 % training dataset and 30 % test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
# importing random forest classifier from assemble module
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
# creating dataframe of IRIS dataset
data = pd.DataFrame({'sepallength': iris.data[:, 0], 'sepalwidth': iris.data[:, 1],'petallength': iris.data[:, 2], 'petalwidth': iris.data[:, 3],'species': iris.target})
print(data.head())
# creating a RF classifier
clf = RandomForestClassifier(n_estimators = 100)
# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train)
# performing predictions on the test dataset
y_pred = clf.predict(X_test)
# metrics are used to find accuracy or error
from sklearn import metrics
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))
print(clf.predict([[3, 3, 2, 2]]))



# Create a Random forest Classifier
clf = RandomForestClassifier(n_estimators = 100)
# Train the model using the training sets
clf.fit(X_train, y_train)
# using the feature importance variable
feature_imp = pd.Series(clf.feature_importances_, index = iris.feature_names).sort_values(ascending = False)
print(feature_imp)





