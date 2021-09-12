import pandas as pd
data = pd.read_csv("/content/drive/MyDrive/Placement_Data_Full_Class.csv")
data.head()

data1 = pd.get_dummies(data,drop_first=True)
data1.head()
data1.columns

data1.describe()

data1.isna().sum()

data2 = data1.drop(['salary'],axis=1)
data2.head()

data2 = data2.iloc[:,1:]
data2.head()

def norm(i):
  x = (i - i.min())/(i.max()-i.min())
  return x

data2 = norm(data2)

data2.describe()

import seaborn as sns
sns.pairplot(data2.iloc[:,:5])

data2.columns

predictors = data2.iloc[:,:14]
predictors.head()

target = data2.iloc[:,14:]
target.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(predictors,target,test_size = 0.3,random_state = 0)

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

"""**Defineing learners**"""

learner1 = GaussianNB()
learner2 = KNeighborsClassifier(n_neighbors = 3)
learner3 = DecisionTreeClassifier(random_state=0)

"""# **Vooting Classifier**"""

from sklearn.ensemble import VotingClassifier

vooting = VotingClassifier([('NB', learner1), 
                            ('KNN', learner2), 
                            ('DT', learner3)])

vooting.fit(x_train,y_train)

Placement_prediction = vooting.predict(x_test)
Placement_prediction

print('Placement_prediction:', accuracy_score(y_test, Placement_prediction))

Placement_prediction_train = vooting.predict(x_train)
Placement_prediction_train

print('Placement_prediction:', accuracy_score(y_train, Placement_prediction_train)*100,"%")

"""# **Bagging without Grid search CV and with Decision Tree**"""

from sklearn.ensemble import BaggingClassifier
from sklearn import tree
clf = tree.DecisionTreeClassifier()

help(BaggingClassifier)

bag  = BaggingClassifier(base_estimator=clf, n_estimators=600, max_samples=0.50,bootstrap=True)

bag.fit(x_train,y_train)

from sklearn.metrics import accuracy_score,confusion_matrix

bag_placement_predict_test = bag.predict(x_test)
bag_placement_predict_test

bag_accuracy_test = accuracy_score(y_test,bag_placement_predict_test)
bag_accuracy_test

bag_placement_predict_train = bag.predict(x_train)
bag_placement_predict_train

bag_accuracy_train = accuracy_score(y_train,bag_placement_predict_train)
bag_accuracy_train

"""**Bagging with Grid Search CV**"""

from sklearn.model_selection import GridSearchCV

bagg_clf = BaggingClassifier(n_jobs=-1, random_state=42)

param = { "base_estimator": [DecisionTreeClassifier(), KNeighborsClassifier(),GaussianNB()],
          "n_estimators": [70,50,30],
          "max_samples": [0.5,1.0],
          "bootstrap":  [True,False],
          "bootstrap_features": [True,False]}

bag_grid  = GridSearchCV(estimator=bagg_clf,param_grid=param,cv = 3, n_jobs = -1, verbose = 1)

bag_grid.fit(x_train,y_train)

bag_grid.best_params_

"""**Creating Model on the basis of Best Parameters got from grid search CV**"""

clf_final = BaggingClassifier(base_estimator=clf,n_estimators = 70, bootstrap=True,bootstrap_features=False,max_samples=0.5)

clf_final.fit(x_train,y_train)

clf_final_predict_test = clf_final.predict(x_test)

bag_final_acc_test = accuracy_score(y_test,bag_placement_predict_test)
bag_final_acc_test

clf_final_predict_train = clf_final.predict(x_train)

bag_final_acc_train = accuracy_score(y_train,bag_placement_predict_train)
bag_final_acc_train

"""# **Ada Boost**"""

from sklearn.ensemble import AdaBoostClassifier

abc = AdaBoostClassifier()

abc.fit(x_train,y_train)

confusion_matrix(y_test, abc.predict(x_test))

accuracy_score(y_test, abc.predict(x_test))

accuracy_score(y_train, abc.predict(x_train))

"""**Adaboost with grid search CV**"""

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier as knn

abc1 = AdaBoostClassifier()

# for adaboost we define param_grid in this below format only other format is not supportable
grid = dict()
grid['base_estimator'] = [DecisionTreeClassifier(),knn(),GaussianNB()]
grid['n_estimators'] = [10, 50, 100, 500]
grid['learning_rate'] = [0.0001, 0.001, 0.01, 0.1, 1.0]
grid['algorithm'] = ['SAMME', 'SAMME.R']

grid_adaboost = GridSearchCV(estimator = AdaBoostClassifier(), param_grid = grid,n_jobs=-1, cv = 10, scoring='accuracy')

grid_result= grid_adaboost.fit(x_train,y_train)

grid_result.best_params_

"""**making model on the basis of adaboost grid search**"""

ada2 = AdaBoostClassifier(base_estimator=GaussianNB(),algorithm='SAMME.R',learning_rate=0.01,n_estimators=500)

ada2.fit(x_train,y_train)

accuracy_score(y_test, ada2.predict(x_test))

"""# **Gradient Boosting**"""

from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()

gbc.fit(x_train,y_train)

accuracy_score(y_test, gbc.predict(x_test))

"""**`Gradient Boosting With Grid search CV`**"""

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import GradientBoostingClassifier
gbc1 = GradientBoostingClassifier()

param = {
    "loss" : ['deviance', 'exponential'],
    "learning_rate" : [0.1,0.2,0.4,1.0,0.01],
    "criterion" : ['friedman_mse','mse','mae']
}

grid_gbc = GridSearchCV(estimator= gbc1,param_grid=param,n_jobs=-1,cv = 10)

grid_result = grid_gbc.fit(x_train,y_train)

grid_result.best_params_

gbc2 = GradientBoostingClassifier(loss='exponential',learning_rate=0.1,criterion= 'friedman_mse')

gbc2.fit(x_train,y_train)

accuracy_score(y_test, gbc2.predict(x_test))

"""# **XG Boost aka Xtreme Gradient boosting**"""

import xgboost as xgb

xgb_clf = xgb.XGBClassifier(max_depths = 5, n_estimators = 10000, learning_rate = 0.3, n_jobs = -1)

xgb_clf.fit(x_train, y_train)

accuracy_score(y_test, xgb_clf.predict(x_test))

xgb.plot_importance(xgb_clf)

"""**xgboosting with grid search**"""

xgb1 = xgb.XGBClassifier(n_estimators = 500, learning_rate = 0.1, random_state = 42)

param = {
    'max_depth': range(3,10,2),
    'gamma': [0.1, 0.2, 0.3],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0,9],
    'rag_alpha': [1e-2, 0.1, 1]
}

grid_search = GridSearchCV(xgb1, param, n_jobs = -1, cv = 5, scoring = 'accuracy')

result_grid = grid_search.fit(x_train,y_train)

grid_search.best_params_

xgb2 = xgb.XGBClassifier(n_estimators = 500, learning_rate = 0.1, random_state = 42,colsample_bytree= 0.8, gamma = 0.1, max_depth = 3, rag_alpha = 0.01, subsample = 0.9)

xgb2.fit(x_train,y_train)

accuracy_score(y_test, xgb2.predict(x_test))
