import pandas as pd

url = 'https://gist.githubusercontent.com/guilhermesilveira/4d1d4a16ccbf6ea4e0a64a38a24ec884/raw/afd05cb0c796d18f3f5a6537053ded308ba94bf7/car-prices.csv' 
data = pd.read_csv(url)


change = {
    'no' : 0,
    'yes': 1
}

data['sold'] = data.sold.map(change)


from datetime import datetime
data['years_old'] = datetime.today().year - data.model_year
data['km_per_year'] = 1.60934*data.mileage_per_year

data = data.drop(columns = ['Unnamed: 0', 'mileage_per_year', 'model_year'], axis = 1)

x = data[['price', 'years_old', 'km_per_year']]
y = data['sold']


###################### Linear SVC Estimator #####################
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import numpy as np

SEED = 5
np.random.seed(SEED)
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.25,
                                                   stratify = y)
print('We are going to train with %d elements and test with %d elements' %(len(train_x), len(test_x)))

model = LinearSVC()
model.fit(train_x, train_y)
predict = model.predict(test_x)
accuracy = accuracy_score(test_y, predict)
print('the accuracy is %.2f%%' %(accuracy*100))

###################### Dummy Classifier #####################
from sklearn.dummy import DummyClassifier
dummy_stratified = DummyClassifier(strategy='stratified')
dummy_stratified.fit(train_x, train_y)
dummy_accuracy = dummy_stratified.score(test_x, test_y)
#dummy_predict = dummy_stratified.predict(test_x)
#dummy_accuracy = accuracy_score(test_y, dummy_predict)
print('the dummy stratified accuracy is %.2f%%' %(dummy_accuracy*100))


from sklearn.dummy import DummyClassifier
dummy_mostfrequent = DummyClassifier(strategy="most_frequent")
dummy_mostfrequent.fit(train_x, train_y)
dummy_accuracy = dummy_mostfrequent.score(test_x, test_y)
#dummy_predict = dummy_mostfrequent.predict(test_x)
#dummy_accuracy = accuracy_score(test_y, dummy_predict)
print('the dummy most frequent accuracy is %.2f%%' %(dummy_accuracy*100))


###################### SVC Estimator #####################
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

SEED = 5
np.random.seed(SEED)
raw_train_x, raw_test_x, train_y, test_y = train_test_split(x, y, test_size = 0.25,
                                                   stratify = y)
print('We are going to train with %d elements and test with %d elements' %(len(train_x), len(test_x)))

scaler = StandardScaler()
scaler.fit(raw_train_x)
train_x = scaler.transform(raw_train_x)
test_x = scaler.transform(raw_test_x)

model = SVC()
model.fit(train_x, train_y)
predict = model.predict(test_x)
accuracy = accuracy_score(test_y, predict)
print('the accuracy is %.2f%%' %(accuracy*100))


###################### Decision Tree Classifier #####################
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

SEED = 5
np.random.seed(SEED)
raw_train_x, raw_test_x, train_y, test_y = train_test_split(x, y, test_size = 0.25,
                                                   stratify = y)
print('We are going to train with %d elements and test with %d elements' %(len(train_x), len(test_x)))

model = DecisionTreeClassifier(max_depth=3)
model.fit(raw_train_x, train_y)
predict = model.predict(raw_test_x)
accuracy = accuracy_score(test_y, predict)
print('the accuracy is %.2f%%' %(accuracy*100))



from sklearn.tree import export_graphviz
import graphviz
features = x.columns
dot_data = export_graphviz(model, out_file = None,
                           filled = True,
                           rounded = True,
                           feature_names = features,
                           class_names = ['no', 'yes'])
graph = graphviz.Source(dot_data)
graph

