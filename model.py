import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score


class ModelPrediction:

    def __init__(self, df, model='logistic'):

        self.model_choices = {
        'logistic': {'model': LogisticRegression(),'params': {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}},
        'rforest' : {'model': RandomForestClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'), 'params': {}}
                              }
        self.model = self.model_choices[model]['model']
        self._preprocess_features(df)

    def _preprocess_features(self, df):
        current_date = '2014-07-01' #Date specified to use in competition

        class_columns = ['city', 'phone']
        ordinal_columns = ['avg_dist', 'trips_in_first_30_days', 'luxury_car_user']
        X_columns = class_columns + ordinal_columns
        y_column = 'churn'
        df[y_column] = df['last_trip_date'] < '2014-06-01'
        X = df[X_columns]

        self.y = df[y_column]
        self.X = pd.get_dummies(X, columns=class_columns, drop_first=True, dummy_na=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.25)

        return self

    def cross_validate(self):
        return cross_val_score(self.model, self.X, self.y)

    def grid_search(self):
        models = [x['model'] for x in self.model_choices.values()]
        params = [x['params'] for x in self.model_choices.values()]

        grid = {}
        for i in range(len(models)):
            gs = GridSearchCV(models[i], params[i]).fit(self.X, self.y)
            grid[self.model_choices.keys()[i]] = gs

        return grid

if __name__ == "__main__":
    df = pd.read_csv('data/churn_train.csv', low_memory=False)
    model = ModelPrediction(df)

    gs = model.grid_search()
    for m, g in gs.iteritems():
        print 'The best params for {} are {}'.format(m, g.best_params_)
        print '{} gives a CV score of {}'.format(m, g.best_score_)

    #print model.cross_validate()
    model.model.fit(model.X_train, model.y_train)
    model.model.score(model.X_test, model.y_test)
