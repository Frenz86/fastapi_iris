### just to create and test the model pkl

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib


# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(filepath_or_buffer=url,header=None,sep=',',names=names)
# Split-out validation dataset
array = dataset.values

X = array[:,0:4]
y = array[:,4]
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=667, 
                                                    shuffle=True)
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

#save the model to disk
joblib.dump(classifier,'LRClassifier.pkl')

#load the model from disk
loaded_model = joblib.load('LRClassifier.pkl')
result = loaded_model.score(X_test, y_test)
print(result)
pred = loaded_model.predict([[3,5,4,3]])
print(pred)