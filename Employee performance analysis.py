## Import the data from cloud
# pip install pymongo[srv]
import pymongo
import pandas as pd
## Connecting to cloud using the pymongo package
client = pymongo.MongoClient("mongodb+srv://ds_project:ds_project@cluster0.h8ffl.mongodb.net/Project2?retryWrites=true&w=majority")

db = client.Project2        ## Connecting to the database
print(db)

collection = db.Emp_perform       ## Extracting the data from database

## Finding the memory location
data = collection.find()
print(data)

## Converting the data into dataframe
data_WAC = pd.DataFrame(list(data))


# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pylab

#Loading Dataset
## data_WAC = pd.read_excel("C:/Users/Dell/Desktop/Assighnments/Project/Performance__Analysis.xlsx")
data_WAC.shape ## Shape of the dataset
data_WAC.isnull().sum() # zero Null values
data_WAC.isna().sum() # zero nan values
data_WAC.dtypes # int and Float
desc=pd.DataFrame(data_WAC.describe())
data_WAC.columns

## Removing unimp columns
data_WAC = data_WAC.drop(['Employee Name'], axis = 1)
data_WAC = data_WAC.drop(['Employee ID'], axis = 1)
data_WAC = data_WAC.drop(['Sum'], axis = 1)

## Reanaming the dataset
data_WAC.columns='Designation','Salary_Hike','Training_Hours', 'wlb', 'Tenure','Salary','Total_Work_Exp','Tenure_After_Last_Promotion','Education','Adaptability','Attendane','Leadership_Skills','Quality_Of_Deliverables', 'Comm_Skills','Decision_Making','Team_Work_Skills','Productivity','Cust_Relation_Skills','Job_Knowledge','Dependability','Rating'


## Countplot can be done on Categorical data to pull insights
## Value counts and countplots for catagarical data
data_WAC.Education.value_counts()
sns.countplot(data_WAC.Education)

data_WAC.Designation.value_counts()
sns.countplot(data_WAC.Designation)

data_WAC.wlb.value_counts()
sns.countplot(data_WAC.wlb)


## Manual EDA
## Mean, Median and Mode
data_WAC.mean()
data_WAC.median()
a = data_WAC.mode()

## Skew and Kurt
data_WAC.skew()
data_WAC.kurt()

## Catplots for x="Designation",y="Training_Hours"
import seaborn as sns
g=sns.catplot(x="Designation",y="Training_Hours",data=data_WAC,kind="point")
g.set_xticklabels(rotation=90)
## Catplots for x="Designation",y="Salary"
g=sns.catplot(x="Designation",y="Salary",data=data_WAC,kind="point")
g.set_xticklabels(rotation=90)
## Catplots for x="wlb",y="Salary"
g=sns.catplot(x="wlb",y="Salary",data=data_WAC,kind="point")
g.set_xticklabels(rotation=90)

## kdeplotplot,  violinplot, boxenplot, distplot
sns.kdeplot(data_WAC.Training_Hours)
sns.violinplot(data_WAC.Salary)
sns.boxenplot(data_WAC.Tenure)
sns.boxenplot(x="wlb",y="Salary",hue="Rating",data=data_WAC)
sns.distplot(data_WAC.Total_Work_Exp)


#W# Manual Labele encoding Worklife bal
value = {'Bad': 0,'Good': 1 , 'Best' : 2}
data_WAC.wlb = [value[item] for item in data_WAC.wlb]
## Manual Labele encoding Education
data_WAC['Education'].unique() #['Graduate', 'Post Graduate', 'Doctorate']
label2 = {'Graduate' : 1, 'Post Graduate' : 2, 'Doctorate' : 3}
data_WAC['Education']=[label2[item] for item in data_WAC['Education']]
## Manual Labele encoding Performance RAting
data_WAC['Rating'].unique() # ['A', 'B', 'C', 'D', 'E']
label3 = {'A' : 5, 'B' : 4, 'C' : 3, 'D' : 2, 'E' : 1}
data_WAC['Rating']=[label3[item] for item in data_WAC['Rating']]
## Label encoding the for Designatio
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
data_WAC['Designation'].unique()
data_WAC['Designation']=le.fit_transform(data_WAC['Designation'])

#Histograms
data_WAC.hist()
plt.hist(data_WAC.Salary) #Data is right skewed
plt.hist(data_WAC.Training_Hours) #Seems data is in normalized format
plt.hist(data_WAC.Total_Work_Exp) #Data is right skewed

#BoxPlots
data_WAC.boxplot()
plt.boxplot(data_WAC['Salary']),plt.ylabel("Salary") 
plt.boxplot(data_WAC['Training_Hours']),plt.ylabel("Training Hours")
plt.boxplot(data_WAC['Total_Work_Exp']),plt.ylabel("Total Work Exp") 

## Auto EDA
## pip install sweetviz
## importing sweetviz
import sweetviz as sv
advert_report = sv.analyze(data_WAC)
advert_report.show_html('Employee Performance EDA.html')

## Splitting the data
X = data_WAC.drop(['Rating'], axis = 1)
Y = data_WAC.Rating
## Feature selaction
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
ordered_rank_features = SelectKBest(score_func=chi2,k=20)
ordered_feature = ordered_rank_features.fit(X, Y)
dfscores=pd.DataFrame(ordered_feature.scores_,columns=["Score"])
dfscores
dfscores.nlargest(21, 'Score')
## Education, Total work exp, Desgnition, Tenure

## Heat mapa and correlation part
cor=pd.DataFrame(data_WAC.corr())
import seaborn as sns
sns.heatmap(data_WAC.corr(), annot = True, fmt='.0%') #(Salary and Total work exp),(Total Work Exp and Tenure),(Salary and Tenure) have high postive correlation,


# Import library for VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
def calc_vif(X):
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return(vif)
calc_vif(X)
## Getting "Total years of exp" and "Tenure"has high VIF vlues.


## Robust Scaler
## Quantile Transformer Scaler
## Log Transformation
## Power Transformer Scaler

##Power Transformer Scaler
## parameters: method = 'yeo-johnson', 'box-cox'
from sklearn.preprocessing import PowerTransformer
scaler = PowerTransformer(method = 'yeo-johnson')
col_names = ["Salary_Hike", "Salary", "Tenure", "Total_Work_Exp"]
features = data_WAC[col_names]
X[col_names] = scaler.fit_transform(features.values)
X

# Import library for VIF after feture transmission
from statsmodels.stats.outliers_influence import variance_inflation_factor
def calc_vif(X):
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return(vif)
calc_vif(X)
## After power transmission we got all variable has below 10 VIF value.

## Concating the X, Y
df = pd.concat([X, Y], join = 'outer', axis = 1)
#Exploratory Data Analysisq
cor=pd.DataFrame(data_WAC.corr())
import seaborn as sns
sns.heatmap(df.corr(), annot = True, fmt='.0%') #(Salary and Total work exp),(Total Work Exp and Tenure),(Salary and Tenure) have high postive correlation,
## Total work life balance has 0% relation with output

## value counts and countplots
Y.value_counts()
sns.countplot(Y)
## Balancing the imbalance data(Feature Engineering)
from imblearn.over_sampling import SMOTE
smote = SMOTE('minority')
X3, Y3 = smote.fit_resample(X, Y)
X4, Y4 = smote.fit_resample(X3, Y3)
X5, Y5 = smote.fit_resample(X4, Y4)
X6, Y6 = smote.fit_resample(X5, Y5)

sns.countplot(Y3) 
sns.countplot(Y4) 
sns.countplot(Y5) 
sns.countplot(Y6) 

## Splitting thedata into train and test data
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X6, Y6, test_size = 0.30, random_state = 0)

## Model buielding packages
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.model_selection import GridSearchCV
import warnings
from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from sklearn.svm import SVC
#######
## Decission_tree
from sklearn.tree import DecisionTreeClassifier as DT
DT = DT(criterion = 'entropy')
## Model fit on train data
DT.fit(train_x, train_y)
## Model predict on test data
pred_test = DT.predict(test_x)
## Confusion matrics
from sklearn.metrics import accuracy_score, confusion_matrix
cm_test = confusion_matrix(pred_test, test_y)
cm_test
## Accuracy for test data
acc_test_D = np.mean(pred_test == test_y)
acc_test_D

## Sencitivity(True positive rate)
Sensitivity = cm_test[0,0]/(cm_test[0,0] + cm_test[0,1])
print('sensitivity:', Sensitivity)
## Specificity(True nagative)
specitivity = cm_test[1,1]/(cm_test[1,0] + cm_test[1,1])
print('Specificity:', specitivity)

## Plot the Decition tree
## conda install python-graphviz
# from sklearn.externals.six import StringIO 
# from sklearn.tree import export_graphviz
# conda install graphviz
# pip install graphviz
from IPython.display import Image
from sklearn import tree
from six import StringIO 
from IPython.display import Image  
import pydotplus
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn import model_selection
from graphviz import Digraph
import pydotplus
dot_data = StringIO()
export_graphviz(DT, out_file=dot_data, filled=True, rounded=True,special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
graph.write_png("decisiontree.png")


## Model predict on train data
pred_train = DT.predict(train_x)
## Confusion matrics
from sklearn.metrics import accuracy_score, confusion_matrix
cm_train = confusion_matrix(pred_train, train_y)
cm_train
## Accuracy for train data
acc_train_D = np.mean(pred_train == train_y)
acc_train_D

## Sencityvity(True positive rate)
sensitivity_train = cm_train[0,0]/(cm_train[0,0] + cm_train[0,1])
print('sensitivity_train:', sensitivity_train)
## Specityvity(True Nagetive)
Specificity_train = cm_train[1,1]/cm_train[1,1] + cm_train[1,0]
print('Specificity_train:', Specificity_train)

## Tree plot
dot_data = StringIO()
export_graphviz(DT, out_file=dot_data, filled=True, rounded=True,special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
graph.write_png("decisiontree.png")

#### pruning \/\//\\/\/\/\/\//\\//\\//\/\/\\/\/\/\/\//\\\\\/\\//\/\\/\/\/\//
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 110, num = 11)]
max_depth.append(None)
max_leaf_nodes = [2, 5, 10, 20,30, 40]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 5, 7]

## Prouning
from sklearn import tree
dtree = tree.DecisionTreeClassifier(criterion = "gini", splitter = 'best', max_leaf_nodes = 10, min_samples_leaf = 3, max_depth= 5)
dtree.fit(train_x, train_y)
## Model predict on test data
pred_test = dtree.predict(test_x)
## Confusion matrics
pd.crosstab(pred_test, test_y)
## Accuracy for test sata
accuracy_test_pr = np.mean(pred_test == test_y)
accuracy_test_pr

## Model predict on the train data
pred_train = dtree.predict(train_x)
## Confusion matrics
pd.crosstab(pred_train, train_y)
## Accuracy for train data
acc_train_pr = np.mean(pred_train == train_y)
acc_train_pr


## #### Grid search on decision tre
#### Grid search on decision tree #####\//\\//\\//\/\/\\/\/\/\/\//\\\\\/\\//\/\\/\/\/\//
from sklearn.model_selection import GridSearchCV
criterion = ['gini', 'entropy']
max_depth = [1,3,5,7,9, 11,None]
splitter = ['best', 'splitter']
grid = GridSearchCV(estimator = DT , param_grid = dict(criterion = criterion, max_depth = max_depth, splitter = splitter))                                                              
grid.fit(train_x, train_y)
print (grid.best_score_)
print (grid.best_params_)
#building the model based on the optimised parameters
model = DT(criterion = 'gini', max_depth = 5, splitter = 'best' )
## Model fit on train data
model.fit(train_x, train_y)
## Model predict on test data
pred_test = model.predict(test_x)
## Confusion matrics
pd.crosstab(pred_test, test_y)
## Accuracy for test sata
accuracy_test_gr = np.mean(pred_test == test_y)
accuracy_test_gr 
## Plot the Decition tree
from sklearn import tree
dt_dia = tree.DecisionTreeClassifier(random_state = 0)
dt_dia_t = dt_dia.fit(test_x, test_y)
tree.plot_tree(dt_dia_t)

## Model predict on the train data
pred_train = model.predict(train_x)
## Confusion matrics
pd.crosstab(pred_train, train_y)
## Accuracy for train data
acc_train_gr = np.mean(pred_train == train_y)
acc_train_gr

## Plot the Decition tree
from sklearn import tree
dt_dia = tree.DecisionTreeClassifier(random_state = 0)
dt_dia_tr = dt_dia.fit(train_x, train_y)
tree.plot_tree(dt_dia_t)

###################################################################

## Building the model using RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=5, n_estimators=500, criterion="entropy")
## model fitting on the train data
rf.fit(train_x, train_y)
## Predicting on the test data
pred_test_rf = rf.predict(test_x)
## Confusion matrics
from sklearn.metrics import accuracy_score, confusion_matrix
cm_test = confusion_matrix(pred_test_rf, test_y)
## Test acc
from sklearn.metrics import accuracy_score
RF_test_Accuracy = accuracy_score(test_y,pred_test_rf)
RF_test_Accuracy


pred_train_rf = rf.predict(train_x)
## Confusion matrics
cm_test = confusion_matrix(pred_train_rf, train_y)
## Accuracy
RF_train_Accuracy = accuracy_score(train_y,pred_train_rf)
RF_train_Accuracy

# selecting feature importance
feature_imp = pd.Series(rf.feature_importances_, index = X.columns).sort_values(ascending=False)
feature_imp
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()

#### Grid search ################

from sklearn.model_selection import GridSearchCV
n_estimators = [100, 300, 500, 800, 1200,1500,2000]
max_depth = [5, 8, 15, 25, 30]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10] 
hyperF = dict(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf) 
gridF = GridSearchCV(rf, hyperF, cv = 3, verbose = 1, n_jobs = -1)
bestF = gridF.fit(train_x, train_y)
print (gridF.best_score_)
print (gridF.best_params_)

rf_withBestParam = RandomForestClassifier(max_depth=15, n_estimators=1500, criterion="entropy", min_samples_leaf=1, min_samples_split=2)
## model fitting on the train data
rf_withBestParam.fit(train_x, train_y)
## Predicting on the test data
pred_test_rf1 = rf_withBestParam.predict(test_x)
## Confusion matrics
from sklearn.metrics import accuracy_score, confusion_matrix
cm_test_rf = confusion_matrix(pred_test_rf1, test_y)
from sklearn.metrics import accuracy_score
RF_TestAccuracy_BestParam = accuracy_score(test_y,pred_test_rf1)
RF_TestAccuracy_BestParam


## Model predict on train data
pred_train_rf1 = rf_withBestParam.predict(train_x)
## Confusion matrics
from sklearn.metrics import accuracy_score, confusion_matrix
cm_train_rf = confusion_matrix(pred_train_rf1, train_y)
## Accuracy for train data
RF_TrainAccuracy_BestParam = accuracy_score(train_y,pred_train_rf1)
RF_TrainAccuracy_BestParam

###################################################

####\/\/\//\\\//\//\/\\\\\\//\/\\/\//\\/\/\/\/\//\\//\\//\/\/\\/\/\/\/\//\\\\\/\\//\/\\/\/\/\//\\//\/\\/\\\/\/\

# AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(learning_rate = 0.2, n_estimators = 5000)
## Fitting on train data
ada_clf.fit(train_x, train_y)
## Predicting on test data
test_pred_ada = ada_clf.predict(test_x)  
## Cross table for test data
pd.crosstab(test_y, test_pred_ada)
## Test Accuracy
test_acc_ada = np.mean(test_y == test_pred_ada)
test_acc_ada

## Predicting on train data
train_pred_ada = ada_clf.predict(train_x)
## Predicting on train data
pd.crosstab(train_y, train_pred_ada)
## Train Accuracy
train_acc_ada = np.mean(train_y == train_pred_ada)
train_acc_ada

####\/\/\//\\\//\//\/\\\\\\//\/\\/\//\\/\/\/\/\//\\//\\

## eXtreme Gradient Boosting
## Model buielding
## pip install xgboost
import xgboost as xgb
model_xg = xgb.XGBClassifier(max_depths = 5, n_estimators = 10000, learning_rate = 0.3, n_jobs = -1)
## FItting the model to train data
model_xg.fit(train_x, train_y)
## Model evoluting on test data
pred_xg = model_xg.predict(test_x)
## Confusion matrics for test data
from sklearn.metrics import accuracy_score, confusion_matrix
confusion_matrix(test_y, pred_xg)
## Accuracy for test data
test_Acc_XG = accuracy_score(test_y, pred_xg)
test_Acc_XG

## Model avolutiong on train data
pred_trn_xg = model_xg.predict(train_x)
## Confusion matrics
confusion_matrix(pred_trn_xg, train_y)
## Accuracy for train data
train_Acc_XG = accuracy_score(train_y, pred_trn_xg)
train_Acc_XG

####\/\/\//\\\//\//\/\\\\\\//\/\\/\//\\/\/\/\/\//\\//\\
## SVM
## Buielding the model
from sklearn.svm import SVC
svm = SVC(kernel = "linear")
## Fitting on the train data
svm.fit(train_x, train_y)
## Predict on test data
pred_svm = svm.predict(test_x)
## confussion matrics
con_svm = pd.crosstab(pred_svm, test_y)
con_svm
## Accuracy for test data
acc_test_svm = np.mean(pred_svm == test_y)
acc_test_svm

## predict on train
pred_svm_train = svm.predict(train_x)
## confusion matrics for train data
comn_test = pd.crosstab(pred_svm_train, train_y)
comn_test
## Accuracy for train data
acc_train_svm = np. mean(pred_svm_train == train_y)
acc_train_svm

# Which is the best Model ?
results = pd.DataFrame({
    'Model': ['Decision Tree', 'prouning_DT', 'grid search alg', 'Random forest', 'Ada boosting', 'EXG boosting', 'SVM'],
    'train accuracy': [acc_train_D, acc_train_pr, acc_train_gr , RF_train_Accuracy, train_acc_ada, train_Acc_XG, acc_train_svm]})

result_df = results.sort_values(by='train accuracy', ascending=False)
result_df_train = result_df.set_index('train accuracy')
result_df_train

results = pd.DataFrame({
    'Model': ['Decision Tree', 'prouning_DT', 'grid search alg', 'Random forest', 'Ada boosting', "EXG boosting", 'SVM'],
    'test accuracy': [acc_test_D, accuracy_test_pr, accuracy_test_gr, RF_test_Accuracy, test_acc_ada, test_Acc_XG, acc_test_svm]})

result_df = results.sort_values(by='test accuracy', ascending=False)
result_df_test = result_df.set_index('test accuracy')
result_df_test






