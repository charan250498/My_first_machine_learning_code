'''MACHINE LEARNING IN PYTHON'''


'''This is my first machine learning program(on a set of data about iris flower)
   this was purely a guided peice of code and not my own one
   link for the detailed explanation of each step is given below
   and to understand each step please uncomment the commented ones(commented lines below) to understand

   link:http://machinelearningmastery.com/machine-learning-in-python-step-by-step/

   In case of doubts in use of any function use //>>>help(function_name)// for a better understanding of functions
   in a descriptive way.

'''


# importing all required libraries and modules
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import  GaussianNB

# data set loading
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# check
# print(dataset)
# print(dataset.shape)
# print(dataset.head(20))
# print(dataset.describe())
# print(dataset.groupby('class').size())

# univariate plots
'''dataset.plot(kind="box", subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()'''

# histograms
'''dataset.hist()
plt.show()'''

# scatter plots
'''scatter_matrix(dataset)
plt.show()'''

# Split-out validation dataset
array = dataset.values
# print(array)
X = array[:,0:4]
# print(X)
Y = array[:,4]
# print(Y)
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
# print(X_train)
# print(X_validation)
# print(Y_train)
# print(Y_validation)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# print(models)

# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	# print(msg)
# print(names)
# print(results)

# compare algorithms
fig = plt.figure()
fig.suptitle("Algorithm Comparison")
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
# plt.show()

# Make predictions on validation dataset(data set for KNN ALOGRITHM)
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# Make predictions on validation dataset(data set for NB ALGORITHM)
gnb = GaussianNB()
gnb.fit(X_train, Y_train)
predictions = gnb.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))