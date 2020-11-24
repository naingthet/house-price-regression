
![png](figures/housesbanner.png)

# Introduction
In this project, we will tackle the Kaggle House Prices: Advanced Regression Techniques competition. The goal of this project is to develop a robust and powerful regression model to predict house sale prices given 79 features describing homes in Ames, Iowa. The objective is to achieve the highest validation accuracy possible, without peeking at the test data or allowing the test data to influence our methodology. 

## Framework
1. Acquire the data
2. Manually clean the data using knowledge of dataset
3. Preprocess the data
4. Select a base model
5. Select optimal feature set
6. Tune model hyperparameters
7. Export and submit for scoring!

# Setup

## Libraries


```python
# Data loading
import urllib

# Computation time
import time

# Essential data science libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

# Data processing and feature selection
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE, RFECV

# Dimensionality Reduction
from sklearn.decomposition import PCA

# Models
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor,\
ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

# Model selection
from sklearn import metrics
from sklearn.model_selection import cross_validate, ShuffleSplit, cross_val_score, GridSearchCV 

# Hyperparameter tuning and optimization
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.pyll import scope
```

## Graphing 


```python
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')
```

## Shortcut Functions

Throughout this project we will perform cross validation frequently to assess our models. These functions will help us to reproduce our cross validation techniques quickly and accurately. 


```python
# Shuffle data and return cross validation scores

def cv_results(model, X_train, y_train):
  cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
  results = cross_val_score(model, X_train, y_train, cv=cv, n_jobs=-1, scoring='neg_mean_squared_error')
  rmse = np.sqrt(-results).mean()
  return rmse

# Function for printing results as well as model name and parameters

def cv_results_print(model, X_train, y_train):
  start = time.time()
  cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
  results = cross_val_score(model, X_train, y_train, cv=cv, n_jobs=-1, scoring='neg_mean_squared_error')
  rmse = np.sqrt(-results).mean()
  
  model_name = model.__class__.__name__
  model_params = model.get_params()
  print('Model: {}'.format(model_name))
  print('Model Parameters: {}'.format(model_params))
  print('5 Fold CV RMSE: {:.4f}'.format(rmse))
  print('Computation Time: {:.2f}'.format(time.time() - start))

def cv_results_10_print(model, X_train, y_train):
  start = time.time()
  cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
  results = cross_val_score(model, X_train, y_train, cv=cv, n_jobs=-1, scoring='neg_mean_squared_error')
  rmse = np.sqrt(-results).mean()
  
  model_name = model.__class__.__name__
  model_params = model.get_params()
  print('Model: {}'.format(model_name))
  print('Model Parameters: {}'.format(model_params))
  print('10 Fold CV RMSE: {:.4f}'.format(rmse))
  print('Computation Time: {:.2f}'.format(time.time() - start))
```

## Loading Data

The data for this project is provided by Kaggle and available through the Kaggle API. To make this notebook easily reproducible, I have uploaded the data to this project's GitHub repository, from which we will download the data. 


```python
# Data description
descr = urllib.request.urlopen('https://raw.githubusercontent.com/naingthet/house-price-regression/gh-pages/data/data_description.txt')
# Uncomment the code below to print the data description
# for line in descr:
#   decoded_line = line.decode("utf-8")
#   print(decoded_line)

# Training data
train = pd.read_csv('https://raw.githubusercontent.com/naingthet/house-price-regression/gh-pages/data/train.csv')
# Test data
test = pd.read_csv('https://raw.githubusercontent.com/naingthet/house-price-regression/gh-pages/data/test.csv')
```

# Data Cleaning and Visualization

By previewing the dataset it is clear that we have a very large number of columns to work with, and many of these columns appear to have a small number of unique values.


```python
print(train.shape, test.shape)
train.head()
```

    (1460, 81) (1459, 80)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>




```python
# Finding the columns with the highest proportion of missing values
train_null = train.isnull().mean().sort_values(ascending=False).reset_index()[:10]

#Let's visualize this with a barplot
g= sns.catplot(data=train_null, y='index', x=0, kind='bar', orient='h', height=5, aspect=2.5)
g.set_axis_labels('Proportion of Null Values', 'Column')


plt.show()
```


![png](figures/output_16_0.png)


We can see that a few columns have very large proportions of their values missing. However, this may be misleading as some null values actually represent information. 

## Null Values Representing 0 or None
According to the data description, some of our features have null values in place of 0 or "None". For example, a null value in the "Alley" column means that the particular home does not have alley access and does not mean the value is missing. Before moving forward, we will fill in these null values to avoid losing information. 

### Null values representing "None"


```python
train['Alley'].value_counts()
```




    Grvl    50
    Pave    41
    Name: Alley, dtype: int64




```python
cols_none = ['Alley', 'BsmtCond', 
            'BsmtExposure', 'BsmtFinType1', 
            'BsmtFinType2', 'BsmtQual', 'FireplaceQu', 'GarageFinish', 
            'GarageQual', 'GarageType', 'GarageCond', 'PoolQC', 
            'Fence', 'MiscFeature', 'MasVnrType']

for col in cols_none:
  train[col] = train[col].fillna('None')
  test[col] = test[col].fillna('None')
```


```python
train['Alley'].value_counts()
```




    None    1369
    Grvl      50
    Pave      41
    Name: Alley, dtype: int64



### Null values representing 0


```python
cols_zero = ['GarageYrBlt', 'GarageArea', 
            'GarageCars', 'BsmtFinSF1', 'BsmtUnfSF', 'BsmtFinSF2', 
             'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']

for col in cols_zero:
  train[col] = train[col].fillna(0)
  test[col] = test[col].fillna(0)
```

## Adjusting data types
We can see that the MSubClass variable is encoded numerically, but is actually categorical. Thus, we will transform this column into a categorical variable by changing the datatype to string. We will do this with a few columns that are actually categorical. 


```python
cols_to_str = ['MSSubClass', 'OverallCond', 'YrSold', 'MoSold']

for col in cols_to_str:
  train[col] = train[col].apply(str)
  test[col] = test[col].apply(str)

train[cols_to_str].dtypes
```




    MSSubClass     object
    OverallCond    object
    YrSold         object
    MoSold         object
    dtype: object



## Removing Non-informative Features
We will now be removing columns that we expect will not be informative (i.e. those with many missing values. To make our lives easier, we will first drop these columns from the dataframes.

### Missing Values

At first glance, it appears that a few columns may almost entirely consist of missing values. While in some cases it would be useful to fill missing values,  it may actually harm our predictive models if we impute missing values for a substantial portion of a column.


```python
# Finding the columns with the highest proportion of missing values
train_null = train.isnull().mean().sort_values(ascending=False).reset_index()[:10]

#Let's visualize this with a barplot
g= sns.catplot(data=train_null, y='index', x=0, kind='bar', orient='h', height=5, aspect=2.5)
g.set_axis_labels('Column', 'Proportion of Null Values')


plt.show()
```


![png](figures/output_30_0.png)


Since we have already filled in many null values, none of the columns have a very large proportion of missing values. However, the 'LotFrontage' column still has some missing values that we can try to fill in. In this particular case, we can assume that the LotFrontage, or front yard space, of a particular home is generally similar to that of the surrounding homes, so we can impute the missing values using the median of the homes in the same neighborhood.



```python
# Here, we are grouping the data by neighborhood and using this grouping to transform the LotFrontage column
train['LotFrontage'] = train.groupby('Neighborhood').LotFrontage.transform(lambda row: row.fillna(row.median()))
test['LotFrontage'] = test.groupby('Neighborhood').LotFrontage.transform(lambda row: row.fillna(row.median()))
```

We will check the missing values one last time. 


```python
# Finding the columns with the highest proportion of missing values
train_null = train.isnull().mean().sort_values(ascending=False).reset_index()[:10]

#Let's visualize this with a barplot
g= sns.catplot(data=train_null, y='index', x=0, kind='bar', orient='h', height=5, aspect=2.5)
g.set_axis_labels('Column', 'Proportion of Null Values')


plt.show()
```


![png](figures/output_34_0.png)


Now we only have missing values in the 'Electrical' column, but this represents a very small proportion of the overall values. We will impute these values in the next step, when we use a Scikit-learn pipeline to transform our data.

Let's preview our dataframe one last time before we move on.


```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>None</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>None</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>None</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>None</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>None</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>



## Correlated Features
Now that we have imputed many of our missing values, we will now use a correlation matrix to see how the variables are related.


```python
fig, ax = plt.subplots(figsize = (18,15))
sns.heatmap(train.corr(), vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 60)
plt.show()
```


![png](figures/output_39_0.png)


Here, the bottom row and the right column of the heatmap tell us how each of the features is correlated with the SalePrice output. We can see, for example, that LotArea is highly correlated with sale price, and this makes intuitive sense. However, as we move forward, we must consider that our features may have nonlinear relationships with the output variable (or one another), which this heatmap will not show. 

Next, we can go one step further and map relationships that have a correlation above a certain value (in this case, we will use an absolute value of 0.7). This will make it easier for us to visualize the strong linear associations. 


```python
fig, ax = plt.subplots(figsize = (18,15))
train_corr = abs(train.corr())
sns.heatmap(train_corr > 0.7, cmap=plt.cm.Reds)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 60)
plt.show()
```


![png](figures/output_42_0.png)


Interestingly, we can see that only 2 of the input variables have correlations of at least 0.7 with the output variable. We also see that there are a few input features that are highly correlated. 

At this point, we could remove one input feature from each pair of highly correlated input features. Doing so would avoid providing redundant data to our model. 

While in many cases it would be useful to drop redundant features, this technique presents a challenge. When dropping one redundant feature from a pair, we would have to manually select the feature to drop. Additionally, we would drop the feature based on linear summary statistics, which do not tell the full story. With this in mind, we will keep all of these variables in our data, especially since we will be working with models that have feature selection capabilities (i.e. ensemble methods and tree methods) or regularization (i.e. Lasso and Ridge Regression) built in. 

First we will take a look at the distribution of the output variables, as it may be useful to transform this variable. 

## Separating X and Y Values


```python
fig, ax = plt.subplots(figsize=(10,6))
ax.hist(train['SalePrice'])
#sns.distplot(data=train, x='SalePrice')
ax.set_title('Distribution of Sale Price')
plt.show()
```


![png](figures/output_46_0.png)


We can see that the sale price has a bit of right skew and the data does not form a normal distribution. With this in mind, we will log transform the sale price values. Conveniently, Kaggle's scoring for this competition also uses a log-transformed output variable, so this will help us to estimate our model prediction scores as well. 


```python
# Splitting train data into X and y
X_train = train.drop('SalePrice', axis=1)
y_train = np.log1p(train['SalePrice'])
# Log transforming the y values since this is how Kaggle scores the competition


# As we were not provided the y values for the test set, we will simply make a deep copy of the data
X_test = test.copy(deep=True)

print(X_train.shape, y_train.shape, X_test.shape)
```

    (1460, 80) (1460,) (1459, 80)



```python
fig, ax = plt.subplots(figsize=(10,6))
ax.hist(y_train)
#sns.histplot(data=y_train)
ax.set_title('Distribution of log-transformed Sale Price')
plt.show()
```


![png](figures/output_49_0.png)


As we can see, the output now resembles a normal distribution, which will help our models to predict outcomes. 

## Preventing Data Leakage
We have now split the training data into X and Y datasets, but there are still a couple of changes we can make to make our data more robust. 

First, we will drop the Id column because it does not provide any valuable information. 

Next, we will drop the SaleType and SaleCondition columns to avoid data leakage. These variables were generated at the time of sale, meaning they provide information about the output variable that would not typically be available at the time of prediction. This is called data leakage. 

It is important to consider the objective of this project. Our goal is to predict house prices based on the qualities of the house and neighborhood, not to guess the price of a house that has just been sold. While these variables would likely help us to achieve higher scores on the train set, the test set, and the Kaggle competition, we will exclude them because our goal is to build an accurate and robust model that can generalize to new, unseen test cases.


```python
# Dropping the ID column
# Dropping columns that may cause data leakage (e.g. those that hint to the outcome)
drop_cols = ['Id', 'SaleType', 'SaleCondition']
X_train = X_train.drop(drop_cols, axis=1)
X_test = X_test.drop(drop_cols, axis=1)
print(X_train.shape, y_train.shape, X_test.shape)
```

    (1460, 77) (1460,) (1459, 77)


## Data Transformation Pipeline

Now that we have finished our initial data cleaning, we must preprocess it for our models. We now face two challenges: 


1.   The numerical variables have very different ranges and scales
2.   Categorical variables are encoded using strings

We will address each of these challenges using Scikit-learn pipelines, which allow us to arrange a series of transformations into a Pipeline class, then apply the transformations to the data. 

The added benefit of the Pipeline is that it will allow us to fit the transformations on the training set and to apply the transformations to the unseen test set. This will help us to avoid data leakage and develop models with greater potential for generalization. For example, when imputing missing values in the test set, we will use the median value of the corresponding column in the train set. 

Again, we do so because our ultimate goal is not simply to achieve the best test score, but to build a robust model that can generalize to new cases. 


```python
# Determining which columns are numeric and categorical

# Finding the names of the numerical columns
num_cols = X_train.dtypes[X_train.dtypes != object].index.to_list()

# Finding the names of the categorical columns
X_train_cat = X_train.drop(num_cols, axis=1)
cat_cols = X_train_cat.columns.to_list()
```

First, we will build our pipeline for numerical data. We will first impute any missing values with the median of the feature, then normalize the values. We will use RobustScaler here because it works well with outliers. StandardScaler, which normalizes all values from min to max, is very sensitive to outliers. 


```python
num_pipeline = Pipeline([
                         ('imputer', SimpleImputer(strategy='median')),
                         ('robust_scaler', RobustScaler())
])
```

Next, we will build our categorical pipeline. We will impute missing values with the most frequent value and use one hot encode our categorical values. We use one hot encoding because the categorical values are currently encoded as strings, and the OneHotEncoder will create a new column for each category, allowing our models to take advantage of the categorical information.


```python
cat_pipeline = Pipeline([
                         ('imputer', SimpleImputer(strategy='most_frequent')),
                         ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'))
                         # Ignore unknowns in case the train set contains categories not found in the test set
])
```

Lastly, we put it all together to create our full pipeline. This will ensure the transformations are applied to the appropriate columns.


```python
full_pipeline = ColumnTransformer([
                                   ('num', num_pipeline, num_cols),
                                   ('cat', cat_pipeline, cat_cols)
])
```

Now we can transform our data using the pipelines. Note that we are fitting to the training data and transforming the training and test data to avoid data leakage.


```python
X_train = full_pipeline.fit_transform(X_train)
X_test = full_pipeline.transform(X_test)
# Output is a SciPy sparse array, since most values are 0 due to one hot encoding
print('X_train shape: {}'.format(X_train.shape))
print('X_test shape: {}'.format(X_test.shape))
full_pipeline_params = full_pipeline.get_params() # Storing the pipeline parameters
# Column names won't be that useful to us now, since there are ~300 columns to look through
```

    X_train shape: (1460, 324)
    X_test shape: (1459, 324)


Our data processing is finished! We now have datasets with 301 features, due to one hot encoding. Fortunately, the data is stored in SciPy sparse matrices, which store the locations of nonzero values, which will save space and help us to minimize computation time. 

# Selecting a Base Model
Now that we have reduced the dimensionality of our dataset, we will train a set of base regression models using default parameters. The goal here is to select the most promising model and to focus on maximizing the predictive power of that singular model.


```python
# List of models to evaluate
models = [
          # Linear regression
          LinearRegression(),
          Ridge(),
          SGDRegressor(),
          Lasso(),

          # Ensemble methods
          RandomForestRegressor(),
          AdaBoostRegressor(),
          ExtraTreesRegressor(),

          # KNN
          KNeighborsRegressor(),

          # Decision Trees
          DecisionTreeRegressor(),

          #XGBoost
          XGBRegressor()
]

# DataFrame to compile results
model_columns = ['model_name', 'parameters', 'rmse', 'time']
base_models = pd.DataFrame(columns=model_columns)

# Populate dataframe with results of each base model
model_index = 0

for model in models:
  start = time.time()
  # Saving model name and paramters
  model_name = model.__class__.__name__
  base_models.loc[model_index, 'model_name'] = model_name
  base_models.loc[model_index, 'parameters'] = str(model.get_params())

  # Cross validation score
  results = cv_results(model, X_train, y_train)
  base_models.loc[model_index, 'rmse'] = results

  #Computation time
  base_models.loc[model_index, 'time'] = time.time() - start

  model_index += 1

base_models = base_models.sort_values(by='rmse', ascending=True)
base_models
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model_name</th>
      <th>parameters</th>
      <th>rmse</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>XGBRegressor</td>
      <td>{'objective': 'reg:squarederror', 'base_score'...</td>
      <td>0.14415</td>
      <td>1.27799</td>
    </tr>
    <tr>
      <th>4</th>
      <td>RandomForestRegressor</td>
      <td>{'bootstrap': True, 'ccp_alpha': 0.0, 'criteri...</td>
      <td>0.145845</td>
      <td>9.46269</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ExtraTreesRegressor</td>
      <td>{'bootstrap': False, 'ccp_alpha': 0.0, 'criter...</td>
      <td>0.148672</td>
      <td>12.5669</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ridge</td>
      <td>{'alpha': 1.0, 'copy_X': True, 'fit_intercept'...</td>
      <td>0.15806</td>
      <td>1.10746</td>
    </tr>
    <tr>
      <th>5</th>
      <td>AdaBoostRegressor</td>
      <td>{'base_estimator': None, 'learning_rate': 1.0,...</td>
      <td>0.172334</td>
      <td>0.711068</td>
    </tr>
    <tr>
      <th>0</th>
      <td>LinearRegression</td>
      <td>{'copy_X': True, 'fit_intercept': True, 'n_job...</td>
      <td>0.195078</td>
      <td>1.95247</td>
    </tr>
    <tr>
      <th>8</th>
      <td>DecisionTreeRegressor</td>
      <td>{'ccp_alpha': 0.0, 'criterion': 'mse', 'max_de...</td>
      <td>0.207736</td>
      <td>0.167726</td>
    </tr>
    <tr>
      <th>7</th>
      <td>KNeighborsRegressor</td>
      <td>{'algorithm': 'auto', 'leaf_size': 30, 'metric...</td>
      <td>0.245247</td>
      <td>0.134309</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Lasso</td>
      <td>{'alpha': 1.0, 'copy_X': True, 'fit_intercept'...</td>
      <td>0.395037</td>
      <td>1.04008</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SGDRegressor</td>
      <td>{'alpha': 0.0001, 'average': False, 'early_sto...</td>
      <td>7.60419e+14</td>
      <td>1.06799</td>
    </tr>
  </tbody>
</table>
</div>



We see that the XGBoost regressor had the lowest RMSE of all the base models we tested. XGBoost is an optimized implementation of gradient boosting. The model's performance on our dataset is unsurprising, as it has consistently performed well in notable machine learning competitions.

# Feature Selection with RFECV

Now that we have selected our model, we will tune it to maximize its predictive power (i.e. minimize RMSE). At this point, our datasets contain 301 features, many of which may not contain useful information. As XGBoost has a built in feature importance metric, we can use recursive feature elimination (RFE) alongside cross validation. This functionality is conveniently provided by sklearn. 

RFE works by first tuning the model (in this case, XGBoost) on all of the features in the dataset. The model's feature importance metric will then be assessed to identify the least important feature. This feature is removed from the feature set and the model is retuned on the entire feature set minus the dropped feature. RFECV with Scikit-learn will apply this for us using cross validation to find the optimal feature set that will maximize our performance based on the objective function (in this case, minimize RMSE). 


```python
start = time.time()
xgb_reg = XGBRegressor(objective='reg:squarederror')
cv_split = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
rfecv = RFECV(estimator=xgb_reg, step=1, cv=cv_split, scoring='neg_mean_squared_error')
rfecv = rfecv.fit(X_train, y_train)
print('Time Elapsed: {}'.format(time.time()-start))
```

    Time Elapsed: 457.01400446891785


As we have done with other transformations, we fit the transformation algorithm on the training data and subsequently apply the transformation to both the training and testing set. 


```python
X_train_rfecv = rfecv.transform(X_train)
X_test_rfecv = rfecv.transform(X_test)
```


```python
X_train_rfecv.shape, X_test_rfecv.shape
```




    ((1460, 57), (1459, 57))




```python
print('5-fold CV RMSE of XGBoost after RFECV: {:.4f}'.format(cv_results(xgb_reg, X_train_rfecv, y_train)))
```

    5-fold CV RMSE of XGBoost after RFECV: 0.1398


# Hyperparameter Tuning
We have now selected our base model and used RFECV to select the optimal feature set. Thus far, our optimizations have focused on transforming the datasets to maximize predictive power, but we have yet to adjust the hyperparameters of the XGBoost model itself. In this next step, we will use Grid Search and Hyperopt with cross validation to identify the optimal hyperparameters for our XGBRegressor. 

## Grid Search CV
Grid search is a hyperparameter tuning algorithm that, provided a dictionary of possible hyperparameter values, exhaustively trains the predictive model on each and every hyperparameter combination. Grid Search is incredibly useful because it can identify an optimal set of hyperparameters as long as the optimal values are contained in the grid. However, as Grid Search trains and cross validates the selected model on every single combination, it has very high computational complexity and will often result in long search times. 

As Grid Search can take a very long time to run, we must do our best to minimize the combinations of hyperparameters provided to the algorithm.


```python
# Defining the parameter grid through which grid search will iterate
param_grid = [
              {
               'booster': ['gbtree'],
               'n_estimators' : list(range(100, 501, 100)),
               'objective' : ['reg:squarederror'],
               'max_depth' : list(range(3, 11)),
               'gamma' : [0.01, 0.1, 0.5, 1.0]
              }
]

# Implement grid search using cross validation
cv_split = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
grid_search = GridSearchCV(estimator=XGBRegressor(), param_grid=param_grid,
                           scoring = 'neg_mean_squared_error', 
                           cv = cv_split)
```


```python
# Fit grid search
start = time.time()
grid_search.fit(X_train_rfecv, y_train)
print('Time Elapsed: {}'.format(time.time()-start))
```

    Time Elapsed: 261.41600584983826



```python
# Retrieve the best estimator from the grid search and save the model
xgb_grid = grid_search.best_estimator_
cv_results_print(xgb_grid, X_train_rfecv, y_train)
```

    Model: XGBRegressor
    Model Parameters: {'objective': 'reg:squarederror', 'base_score': 0.5, 'booster': 'gbtree', 'colsample_bylevel': 1, 'colsample_bynode': 1, 'colsample_bytree': 1, 'gamma': 0.01, 'gpu_id': -1, 'importance_type': 'gain', 'interaction_constraints': '', 'learning_rate': 0.300000012, 'max_delta_step': 0, 'max_depth': 3, 'min_child_weight': 1, 'missing': nan, 'monotone_constraints': '()', 'n_estimators': 200, 'n_jobs': 0, 'num_parallel_tree': 1, 'random_state': 0, 'reg_alpha': 0, 'reg_lambda': 1, 'scale_pos_weight': 1, 'subsample': 1, 'tree_method': 'exact', 'validate_parameters': 1, 'verbosity': None}
    5 Fold CV RMSE: 0.1361
    Computation Time: 2.10


## Hyperparameter tuning with Hyperopt

Grid Search is a very simple algorithm that fits our model using every combination of input parameters. However, the algorithm does not infer any details from each combination. Hyperopt seeks to combat this issue that persists Grid Search and other "greedy" search algorithms. 

Hyperopt is a Python library for the optimization of model hyperparameters that leverages Bayesian concepts of prior probability. Rather than exhaustively or randomly searching a parameter space, hyperopt takes advantage of probability distributions to quickly converge on an optimal set of hyperparameter values. 

We will implement hyperopt using the Tree of Parzen Estimators (TPE) algorithm, which uses prior probabilities to approximate expected improvement in model performance. TPE uses expected improvement, rather than actual model score or accuracy, to converge on optimal hyperparameters, which saves substantial time. 

### Parameter space
First we must define a parameter space for hyperopt using probability distributions. Note that we can also ask hyperopt to simply select from a list of possible hyperparameters, as we are doing with a few variables here. 


```python
param_hyperopt = {
    'n_estimators' : scope.int(hp.quniform('n_estimators', 100, 1000, 1)),
    'max_depth': scope.int(hp.quniform('max_depth', 2, 20, 1)),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
    'subsample': hp.uniform('subsample', 0.8, 1.0), 
    'colsample_bytree' : hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
    'alpha' : hp.choice('alpha', np.arange(0.0, 1.1, 0.1)), 
    'objective':  hp.choice('objective', ['reg:squarederror']),
    'booster': hp.choice('booster', ['gbtree'])
}
```

Next, we will define a function that will time, implement, and record the hyperopt TPE algorithm on our dataset and provide the best performing hyperparameter set. 


```python
def hyperopt(param_space, X_train_data, y_train_data, num_eval):
    
    # Timing
    start = time.time()
    
    # Creating an objective function that will output rmse loss given a set of model parameters
    # Hyperopt will seek to minimize the loss returned by this function
    def objective_function(params):
        clf = XGBRegressor(**params)
        score = cross_val_score(clf, X_train_data, y_train_data, cv=5, scoring='neg_mean_squared_error')
        return {'loss':np.sqrt(-score).mean(), 'status':STATUS_OK}

    # Trials object will store information for each trial, allowing us to see under the hood
    trials = Trials()

    # The fmin function will carry out the hyperopt optimization for us
    # Using TPE and given the objective function, the algorithm will find the hyperparameters that minimize loss as defined by our objective function
    best_param = fmin(objective_function, 
                      param_space, 
                      algo=tpe.suggest, 
                      max_evals=num_eval, 
                      trials=trials,
                      rstate= np.random.RandomState(0))
    
    # Loss for each trial
    loss = [x['result']['loss'] for x in trials.trials]
    
    # Extract our best parameter values into a list
    best_param_values = [x for x in best_param.values()]

    # Fit a new model based on the best parameters, which are stored in the list alphabetically    
    clf_best = XGBRegressor(
        alpha = best_param_values[0],
        booster = 'gbtree',
        colsample_bytree = best_param_values[2],
        learning_rate = best_param_values[3],
        max_depth = int(best_param_values[4]),
        n_estimators = int(best_param_values[5]),
        objective = 'reg:squarederror',
        subsample = best_param_values[7]
                          )
                     
    clf_best.fit(X_train_data, y_train_data)
    
    print("")
    print("##### Results")
    print("Score best parameters: ", min(loss))
    print("Best parameters: ", best_param)
    print("Time elapsed: ", time.time() - start)
    print("Parameter combinations evaluated: ", num_eval)
    
    return trials, clf_best
```


```python
# Running the optimization function and storing the model and trial log
hyperopt_trials, xgb_hyperopt = hyperopt(
    param_space=param_hyperopt, 
    X_train_data=X_train_rfecv, 
    y_train_data=y_train, 
    num_eval=30)
```

    100%|██████████| 30/30 [01:06<00:00,  2.23s/trial, best loss: 0.12048773117575351]
    
    ##### Results
    Score best parameters:  0.12048773117575351
    Best parameters:  {'alpha': 1, 'booster': 0, 'colsample_bytree': 1, 'learning_rate': 0.045439499742861544, 'max_depth': 3.0, 'n_estimators': 586.0, 'objective': 0, 'subsample': 0.9089066803820676}
    Time elapsed:  67.22251677513123
    Parameter combinations evaluated:  30



```python
cv_results_print(xgb_hyperopt, X_train_rfecv, y_train)
```

    Model: XGBRegressor
    Model Parameters: {'objective': 'reg:squarederror', 'base_score': 0.5, 'booster': 'gbtree', 'colsample_bylevel': 1, 'colsample_bynode': 1, 'colsample_bytree': 1, 'gamma': 0, 'gpu_id': -1, 'importance_type': 'gain', 'interaction_constraints': '', 'learning_rate': 0.045439499742861544, 'max_delta_step': 0, 'max_depth': 3, 'min_child_weight': 1, 'missing': nan, 'monotone_constraints': '()', 'n_estimators': 586, 'n_jobs': 0, 'num_parallel_tree': 1, 'random_state': 0, 'reg_alpha': 1, 'reg_lambda': 1, 'scale_pos_weight': 1, 'subsample': 0.9089066803820676, 'tree_method': 'exact', 'validate_parameters': 1, 'verbosity': None, 'alpha': 1}
    5 Fold CV RMSE: 0.1292
    Computation Time: 1.38


We can now see that the hyperopt algorithm has not only found a superior set of hyperparameters, but it has done so while saving us significant time compared to Grid Search!

### Visualizing Hyperopt Score

We can use the trial log to visualize how the minimum error decreases over time


```python
hyperopt_scores = [x['result']['loss'] for x in hyperopt_trials.trials]
```


```python
min_tracker = []
trials_tracker = []

for i in hyperopt_scores:
  trials_tracker.append(i)
  min_so_far = min(trials_tracker)
  min_tracker.append(min_so_far)

fig, ax = plt.subplots()
ax.plot(range(len(min_tracker)), min_tracker)
ax.set_title('Minimum RMSE by Hyperopt Iteration')
ax.set_ylabel('Minimum RMSE')
ax.set_xlabel('Trial')

plt.show()
```


![png](figures/output_93_0.png)


Interestingly, this graph shows us that the best error reduces in large steps, rather than with each iteration. This is because of how hyperopt works--rather than minimizing the objective function at every iteration (i.e. fitting the model and determining the error), hyperopt will use expected improvement to drive iterations. This is also the same reason why the minimum error presented by hyperopt was different from our actual CV result.

Most notably, the hyperopt algorithm is able to converge with very few trials, especially considering that it was given a wide parameter space. It is for this reason that, even if hyperopt does not perform better than grid search, it may be the optimal choice, as it is able to converge faster and provides a strong result. 

# Comparing the Models
In this last step, we will compare the Grid Search and Hyperopt optimizations using 10 fold cross validation. This is to help us be more confident that one method is superior to the other for our use case. We will also consider the base model for reference. 


```python
# Base Model
cv_results_10_print(xgb_reg, X_train_rfecv, y_train)
```

    Model: XGBRegressor
    Model Parameters: {'objective': 'reg:squarederror', 'base_score': None, 'booster': None, 'colsample_bylevel': None, 'colsample_bynode': None, 'colsample_bytree': None, 'gamma': None, 'gpu_id': None, 'importance_type': 'gain', 'interaction_constraints': None, 'learning_rate': None, 'max_delta_step': None, 'max_depth': None, 'min_child_weight': None, 'missing': nan, 'monotone_constraints': None, 'n_estimators': 100, 'n_jobs': None, 'num_parallel_tree': None, 'random_state': None, 'reg_alpha': None, 'reg_lambda': None, 'scale_pos_weight': None, 'subsample': None, 'tree_method': None, 'validate_parameters': None, 'verbosity': None}
    10 Fold CV RMSE: 0.1417
    Computation Time: 2.54



```python
# Grid Search CV
cv_results_10_print(xgb_grid, X_train_rfecv, y_train)
```

    Model: XGBRegressor
    Model Parameters: {'objective': 'reg:squarederror', 'base_score': 0.5, 'booster': 'gbtree', 'colsample_bylevel': 1, 'colsample_bynode': 1, 'colsample_bytree': 1, 'gamma': 0.01, 'gpu_id': -1, 'importance_type': 'gain', 'interaction_constraints': '', 'learning_rate': 0.300000012, 'max_delta_step': 0, 'max_depth': 3, 'min_child_weight': 1, 'missing': nan, 'monotone_constraints': '()', 'n_estimators': 200, 'n_jobs': 0, 'num_parallel_tree': 1, 'random_state': 0, 'reg_alpha': 0, 'reg_lambda': 1, 'scale_pos_weight': 1, 'subsample': 1, 'tree_method': 'exact', 'validate_parameters': 1, 'verbosity': None}
    10 Fold CV RMSE: 0.1373
    Computation Time: 1.99



```python
# Hyperopt
cv_results_10_print(xgb_hyperopt, X_train_rfecv, y_train)
```

    Model: XGBRegressor
    Model Parameters: {'objective': 'reg:squarederror', 'base_score': 0.5, 'booster': 'gbtree', 'colsample_bylevel': 1, 'colsample_bynode': 1, 'colsample_bytree': 1, 'gamma': 0, 'gpu_id': -1, 'importance_type': 'gain', 'interaction_constraints': '', 'learning_rate': 0.045439499742861544, 'max_delta_step': 0, 'max_depth': 3, 'min_child_weight': 1, 'missing': nan, 'monotone_constraints': '()', 'n_estimators': 586, 'n_jobs': 0, 'num_parallel_tree': 1, 'random_state': 0, 'reg_alpha': 1, 'reg_lambda': 1, 'scale_pos_weight': 1, 'subsample': 0.9089066803820676, 'tree_method': 'exact', 'validate_parameters': 1, 'verbosity': None, 'alpha': 1}
    10 Fold CV RMSE: 0.1304
    Computation Time: 1.31


There you have it! We found that the hyperopt algorithm provided the best algorithm for our use case, although all three of the models (hyperopt, grid search, and base) had similar performances. 

# Conclusion
Considering that the XGBoost model showed little improvement despite extensive efforts to optimize the model, it may be worthwhile to consider other models, as they may be able to achieve superior results once optimized. Despite this, we were able to develop a model with high predictive power. 

In this project, we cleaned and preprocessed our data, selected a base model (XGBoost), selected features using RFECV, and optimized the model using hyperopt's TPE algorithm. At each step of the way, we were able to enhance the predictive power of our model, all the while avoiding data leakage in hopes of developing a model that would not only perform well on the Kaggle competition, but also perform well when generalizing to new data. 

## Exporting Results


```python
# Making predictions on the test set with our favorite model
y_pred = xgb_hyperopt.predict(X_test_rfecv)
# We log-transformed our y values, so we need to reverse the transformation
y_pred = np.expm1(y_pred)
y_pred
```




    array([124487.766, 158638.4  , 178700.73 , ..., 184126.64 , 114805.01 ,
           229285.73 ], dtype=float32)




```python
results = pd.DataFrame(data={'Id':test['Id'], 'SalePrice': y_pred})
results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1461</td>
      <td>124487.765625</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1462</td>
      <td>158638.406250</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1463</td>
      <td>178700.734375</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1464</td>
      <td>191255.046875</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1465</td>
      <td>186095.890625</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1454</th>
      <td>2915</td>
      <td>78624.429688</td>
    </tr>
    <tr>
      <th>1455</th>
      <td>2916</td>
      <td>88798.343750</td>
    </tr>
    <tr>
      <th>1456</th>
      <td>2917</td>
      <td>184126.640625</td>
    </tr>
    <tr>
      <th>1457</th>
      <td>2918</td>
      <td>114805.007812</td>
    </tr>
    <tr>
      <th>1458</th>
      <td>2919</td>
      <td>229285.734375</td>
    </tr>
  </tbody>
</table>
<p>1459 rows × 2 columns</p>
</div>




```python
results.to_csv('house_price_sub.csv', header=True, index=False)
```
