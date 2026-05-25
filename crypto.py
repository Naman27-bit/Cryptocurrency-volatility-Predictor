# Generated from: crypto.ipynb
# Converted at: 2026-05-25T11:52:35.509Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv("dataset.csv")
df


df.dtypes

df.info()
df

df.describe()

df.shape

df.isnull().sum()

df.duplicated().sum()

df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df

#EDA(EXPLORATORY DATA ANALYSIS)

# Visualizing missing values
sns.countplot(df.isnull())

df = df.drop(columns=['Unnamed: 0'], errors='ignore')
df

df.shape

#numerical features
plt.figure(figsize=(15,10))
plt.suptitle('Numerical Features Distribution', fontsize=15)  
numerical_columns=df.select_dtypes(include=['int64','float64']).columns

for i in range(0, len(numerical_columns)):
    plt.subplot(4,4,i+1)
    sns.histplot(df[numerical_columns[i]], kde=True)
plt.tight_layout()


#univariate analysis(histogram)
plt.hist(df['open'], bins=30, color='blue')
plt.title('Distribution of Opening Prices')

#bivariate analysis(scatter plot)
plt.scatter(df['open'], df['close'], alpha=0.5)
plt.title('Open vs Close Prices')
plt.xlabel('Opening Price')
plt.ylabel('Closing Price')

sns.pairplot(df[['open', 'close', 'high', 'low']])


#define numerical and categorical columns
#let's drop the ID column for analysis
numerical_columns=[column for column in df.columns if df[column].dtype!='O']
categorical_columns=[column for column in df.columns if df[column].dtype=='O']

#print columns
print('we have {} numerical columns : {}'.format(len(numerical_columns), numerical_columns))
print('we have {} categorical columns : {}'.format(len(categorical_columns), categorical_columns))

#proportion of count data on categorical columns
for column in categorical_columns:
    print(df[column].value_counts(normalize=True)*100)
    print('\n')

#proportion of count data on numerical columns
for column in numerical_columns:
    print(df[column].value_counts(normalize=True)*100)
    print('\n')

#dropping unnecessary columns

df = df.drop(columns=['Unnamed: 0'], errors='ignore')
df

#CATEGORICAL FEATURES

for column in categorical_columns:
    plt.figure(figsize=(8,4))
    sns.countplot(x=df[column])
    plt.title('Count Plot of {}'.format(column))
    plt.show()

df.info()

#FEATURE EXTRACTION
df['date'] = pd.to_datetime(df['date'], format='%y-%m-%d')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek   
df

# feature engineering log_return
# rolling_volatility_7
# rolling_volatility_14
# moving_avg_7
# moving_avg_14
# high_low_range = (High - Low) / Close
# volume_marketcap_ratio = Volume / MarketCap
# 


#feature engineering

import pandas as pd
import numpy as np

df['log_return'] = np.log(df['close'] / df['close'].shift(1))
df['volatility_7'] = df['log_return'].rolling(7).std()
df['volatility_14'] = df['log_return'].rolling(14).std()

df['ma_7'] = df['close'].rolling(7).mean()
df['ma_14'] = df['close'].rolling(14).mean()

df['hl_range'] = (df['high'] - df['low']) / df['close']
# support both 'market_cap' and 'marketCap' column names
market_col = 'market_cap' if 'market_cap' in df.columns else ('marketCap' if 'marketCap' in df.columns else None)
if market_col is None:
    raise KeyError("Neither 'market_cap' nor 'marketCap' found in df columns")
df['volume_mcap_ratio'] = df['volume'] / df[market_col]


df.dropna(inplace=True)
df

#FEATURE AND TARGET SPLIT
x=df[['volatility_7','ma_7', 'ma_14', 'hl_range', 'volume_mcap_ratio']]#independent variables
y=df['volatility_14']#dependent variable

#train-test split

#TRAIN-TEST SPLIT
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# IN THIS CODE, WE TRAINED 80% OF DATA AND 20% FOR TESTING.


#train XGboost model
from xgboost import XGBRegressor
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    missing=np.inf
)
model.fit(x_train, y_train)

# IN THIS MODEL, WE TOOK LESS LEARNING RATE AND HIGH TREES IN ORDER TO MAKE MODEL BETTER.



#MODEL EVALUATION

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pred = model.predict(x_test)

print("MAE:", mean_absolute_error(y_test, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
print("R2 Score:", r2_score(y_test, pred))


df.head()

# FROM THIS METRICES, MODEL EXPLAIN APPROX 60% OF VARIATION AND FROM MAE, THERE IS LESS AVERAGE ERROR.


plt.hist(df['volatility_14'])


# AS A REUSLT, WE CLEARLY OBSERVED THAT INITALLY CRYPTOCURRENCY PRICE WILL INCREASE AND AFTER THAT IT WILL DECREASE AFTER 14 DAYS.


# linear regression model
from sklearn.linear_model import LinearRegression
import numpy as np

# Remove rows with infinity or NaN values in x_train and y_train

tanu= np.isfinite(x_train).all(axis=1) & np.isfinite(y_train)
x_train = x_train[tanu]
y_train = y_train[tanu]

lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
plt.hist(df['volatility_14'])


# IN THIS MODEL ALSO, AN OUTPUT IS SAME AS XGBOOST MODEL,SO WE CAN USE ANY OF THIS MODELS.