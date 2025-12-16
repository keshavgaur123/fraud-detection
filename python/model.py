# Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

from mlxtend.plotting import plot_learning_curves
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, matthews_corrcoef

import warnings
warnings.filterwarnings("ignore")















# Read Data into a Dataframe
df = pd.read_csv('../data/creditcard.csv')

# Display the first few rows of the dataframe to check the data
print(df.head())

# Describe Data
df.describe()

# Display columns to inspect the dataset structure
print(df.columns)

# Check for missing values in the dataset
print(df.isna().sum())

def countplot_data(data, feature):
    '''
        Method to compute countplot of given dataframe
        Parameters:
            data(pd.DataFrame): Input Dataframe
            feature(str): Feature in Dataframe
    '''
    plt.figure(figsize=(10,10))
    sns.countplot(x=feature, data=data)
    plt.show()
def pairplot_data_grid(data, feature1, feature2, target):
    '''
        Method to construct pairplot of the given feature wrt data
        Parameters:
            data(pd.DataFrame): Input Dataframe
            feature1(str): First Feature for Pair Plot
            feature2(str): Second Feature for Pair Plot
            target: Target or Label (y)
    '''

    sns.FacetGrid(data, hue=target, height=6).map(plt.scatter, feature1, feature2).add_legend()
    plt.show()

# Visualizations
countplot_data(df, df.Class)
pairplot_data_grid(df, "Time", "Amount", "Class")
pairplot_data_grid(df, "Amount", "Time", "Class")










# Refine the data based on 'Amount' feature (Example use case)
amount_more = 0
amount_less = 0
for i in range(df.shape[0]):
    if(df.iloc[i]["Amount"] < 2500):
        amount_less += 1
    else:
        amount_more += 1
print(amount_more)
print(amount_less)

percentage_less = (amount_less/df.shape[0])*100
print(f"Percentage of transactions with Amount < 2500: {percentage_less}%")

fraud = 0
legitimate = 1
for i in range(df.shape[0]):
    if(df.iloc[i]["Amount"]<2500):
        if(df.iloc[i]["Class"] == 0):
            legitimate += 1
        else:
            fraud += 1
print(f"Fraudulent transactions with Amount < 2500: {fraud}")
print(f"Legitimate transactions with Amount < 2500: {legitimate}")

# Further data analysis with refined data
df_refine = df[["Time", "Amount", "Class"]]
sns.pairplot(df_refine, hue="Class", height=6)
plt.show()













# Visualizing the distribution of the target variable
df.Class.value_counts()

sns.FacetGrid(df_refine, hue="Class", height=6).map(sns.histplot, "Time").add_legend()
plt.show()

# Correlation matrix of the dataset
plt.figure(figsize=(20,20))
df_corr = df.corr()
sns.heatmap(df_corr)

# Split dataset into features (X) and target (y)
X = df.drop(labels='Class', axis=1) # Features
y = df.loc[:,'Class']               # Target Variable

# Split data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Apply SMOTE to handle class imbalance
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

# Feature selection using Mutual Information
from sklearn.feature_selection import mutual_info_classif
mutual_infos = pd.Series(data=mutual_info_classif(X_res, y_res, random_state=1), index=X_train.columns)

# Sorting and displaying mutual information values
print(mutual_infos.sort_values(ascending=False))

# Plot count of the resampled target variable
sns.countplot(y_res)



















# Define evaluation functions for grid search classifiers
def grid_eval(grid_clf):
    """
        Method to compute the best score and parameters computed by grid search
        Parameter:
            grid_clf: The Grid Search Classifier
    """
    print("Best Score", grid_clf.best_score_)
    print("Best Parameter", grid_clf.best_params_)
    
def evaluation(y_test, grid_clf, X_test):
    """
        Method to compute the following:
            1. Classification Report
            2. F1-score
            3. AUC-ROC score
            4. Accuracy
        Parameters:
            y_test: The target variable test set
            grid_clf: Grid classifier selected
            X_test: Input Feature Test Set
    """
    y_pred = grid_clf.predict(X_test)
    print('CLASSIFICATION REPORT')
    print(classification_report(y_test, y_pred))
    
    print('AUC-ROC')
    print(roc_auc_score(y_test, y_pred))
      
    print('F1-Score')
    print(f1_score(y_test, y_pred))
    
    print('Accuracy')
    print(accuracy_score(y_test, y_pred))

# Add classifier pipelines and grid search as needed (for example, SGDClassifier, RandomForest, etc.)

# Example: Running SGDClassifier pipeline
from sklearn.linear_model import SGDClassifier
pipeline_sgd = Pipeline([
    ('scaler', StandardScaler(copy=False)),
    ('model', SGDClassifier(max_iter=1000, tol=1e-3, random_state=1, warm_start=True))
])

param_grid_sgd = [{
    'model__loss': ['log'],
    'model__penalty': ['l1', 'l2'],
    'model__alpha': np.logspace(start=-3, stop=3, num=20)
}, {
    'model__loss': ['hinge'],
    'model__alpha': np.logspace(start=-3, stop=3, num=20),
    'model__class_weight': [None, 'balanced']
}]

# GridSearchCV for SGDClassifier
grid_sgd = GridSearchCV(estimator=pipeline_sgd, param_grid=param_grid_sgd, scoring='accuracy', n_jobs=-1, cv=5, verbose=1, return_train_score=False)
grid_sgd.fit(X_res, y_res)

grid_eval(grid_sgd)
evaluation(y_test, grid_sgd, X_test)

# now try and test new thing
# contact me if you need future update
# update code 10:18 pm wed 10/12/25 by k*****_ga** (after defining evaluation functions)

# Import additional libraries for models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from imblearn.pipeline import Pipeline as ImbPipeline  # For SMOTE + scaler + model



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from imblearn.pipeline import Pipeline as ImbPipeline  # For SMOTE + scaler + model
from sklearn.linear_model import SGDClassifier
import numpy as np
import warnings
import pandas as pd
import seaborn as sns


import matplotlib.pyplot as plt

# Define models and their hyperparameters
models = {
    'SGDClassifier': {
        'model': SGDClassifier(max_iter=1000, tol=1e-3, random_state=1, warm_start=True),
        'params': [{
            'model__loss': ['log'],
            'model__penalty': ['l1', 'l2'],
            'model__alpha': np.logspace(-3, 3, 10)
        }, {
            'model__loss': ['hinge'],
            'model__alpha': np.logspace(-3, 3, 10),
            'model__class_weight': [None, 'balanced']
        }]
    },
    'LogisticRegression': {
        'model': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'params': {
            'model__C': np.logspace(-3, 3, 5),
            'model__penalty': ['l2', 'none'],
            'model__solver': ['lbfgs', 'saga']
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42, n_jobs=-1),
        'params': {
            'model__n_estimators': [100, 200],
            'model__max_depth': [None, 10, 20],
            'model__class_weight': ['balanced', 'balanced_subsample']
        }
    },
    'XGBoost': {
        'model': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1),
        'params': {
            'model__n_estimators': [100, 200],
            'model__max_depth': [3, 6, 10],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__scale_pos_weight': [1, y_train.value_counts()[0]/y_train.value_counts()[1]]
        }
    },
    'MLP': {
        'model': MLPClassifier(max_iter=500, random_state=42),
        'params': {
            'model__hidden_layer_sizes': [(50,), (100,)],
            'model__activation': ['relu', 'tanh'],
            'model__alpha': [0.0001, 0.001]
        }
    }
}

# Loop through models, create pipeline, and run GridSearchCV
for name, mp in models.items():
    print(f"\n===== Training {name} =====")

    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('scaler', StandardScaler()),
        ('model', mp['model'])
    ])

    grid = GridSearchCV(estimator=pipeline,
                        param_grid=mp['params'],
                        scoring='roc_auc',  # ROC-AUC is better for imbalanced datasets
                        cv=5,
                        n_jobs=-1,
                        verbose=1)

    grid.fit(X_train, y_train)

    print(f"Best ROC-AUC Score for {name}: {grid.best_score_:.4f}")
    print(f"Best Parameters for {name}: {grid.best_params_}")

    # Evaluate on test set
    print(f"\nTest Set Evaluation for {name}:")
    evaluation(y_test, grid, X_test)
    print('hello')


#update next>--------------------------------------------------------Unified pipeline:


#SMOTE → StandardScaler → Model______________________GridSearchCV uses ROC-AUC as scoring (better for imbalanced classes)------------->Can handle all #models in one loop for easy comparison----------------------->Automatically prints best hyperparameters and test evaluation

