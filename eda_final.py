import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('data/churn_train.csv')

'''
Features
========

Feature                 |   Type    |   Pred. Exp   |   Findings
-------------------------------------------------------------------
avg_dist                |   float   |   Low         |
avg_rating_by_driver    |   float   |   Low         |
avg_rating_of_driver    |   float   |   High        |
avg_surge               |   float   |   Med         |
city                    |   cat     |   Med         |   Dummify
phone                   |   cat     |   Med         |   Dummify
signup_date             |   date    |   Med         |   Conv. date
surge_pct               |   float   |   High        |
trips_in_first_30_days  |   int     |   Low         |
luxury_car_user         |   bool    |   Low         |
weekday_pct             |   float   |   Low         |


City: Astapor, Winterfell, King's Landing
Phone: Android, iPhone, NaN

Columns with NaNs:
avg_rating_of_driver, avg_rating_by_driver, phone (Non-null counts = rate_by - 39,838 : rate_of - 33,472 : phone - 39,681)

'''

# Adding churn as target variable
current_date = '2014-07-01'
df['churn'] = df['last_trip_date'] < '2014-06-01' #Company definition of churn
df = df.drop('last_trip_date', axis=1) # Cannot use last_trip_date as feature given churn based on last_trip_date

def churn_rates(df, feature):
    vals = set(df[feature].values)
    churns = []
    for val in vals:
        val_churn_count = len(df[(df[feature] == val) & (df['churn'] == 1)])
        val_count = len(df[df[feature] == val])
        churns.append((val, float(val_churn_count)/val_count))
    return churns

#Correlation matrix
df.corr() #Churn seems to have a weak negative correlation with trips_in_first_30_days and luxury_car_user.

#City analysis
df[['city','churn','avg_dist']].groupby(['city','churn']).count()
df_astapor = df[df['city'] == 'Astapor']
df_kings = df[df['city'] == "King's Landing"]
df_winter = df[df['city'] == 'Winterfell']
df_astapor.corr()
df_kings.corr()
df_winter.corr()

#Phone analysis
df[['phone','churn','avg_dist']].groupby(['phone','churn']).count()
df_iphone = df[df['phone'] == 'iPhone']
df_android = df[df['phone'] == "Android"]
df_iphone.corr()
df_android.corr()

#Churn breakout
churn_n = df[df['churn'] == 0]
churn_y = df[df['churn'] == 1]
churn_n.describe()
churn_y.describe()
churn_rates(df,'city') #Churn rates by city: KL = 0.37, Winter = 0.65, Astapor = 0.75

phone_df = df.dropna(subset=['phone'])
churn_rates(phone_df,'phone') #Churn rates by phone: Android = 0.79, iPhone = 0.55

churn_rates(df,'luxury_car_user') #Churn rates by car type: Luxury = 0.50, Non-lux = 0.70

#Checking histograms for potentially correlated features:
plt.hist(df.trips_in_first_30_days, bins = 50)
plt.hist(df.avg_dist, bins = 50, range = (0, 50))
plt.show()

#Full correlation matrix
pd.plotting.scatter_matrix(df, alpha = 0.3, figsize = (14,8), diagonal = 'kde')
plt.show()

#Churn correlation matrix
def scatter(df, y):
    #drop objs
    df = df.drop(['city', 'phone', 'signup_date'], axis = 1)
    features = df.columns
    fix, axs = plt.subplots(4, 3, figsize=(8,8))
    for feature, ax in zip(features, axs.flatten()):
        ax.scatter(df[feature], y)
        ax.set_title(feature)
    plt.legend()
    plt.tight_layout()
    plt.show()
scatter(df, df['churn'])
