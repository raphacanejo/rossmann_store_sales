import joblib
import inflection
import pandas as pd
import numpy as np
import math
import datetime

# Classe com as Limpezas, Transformações e Encoding
class Rossmann(object):
    def __init__(self):
        self.competition_distance_scaler = joblib.load('webapp/parameter/competition_distance_scaler.joblib')
        self.competition_time_month_scaler = joblib.load('webapp/parameter/competition_time_month_scaler.joblib')
        self.promo_time_week_scaler = joblib.load('webapp/parameter/promo_time_week_scaler.joblib')
        self.store_type_scaler = joblib.load('webapp/parameter/store_type_scaler.joblib')
        self.year_scaler = joblib.load('webapp/parameter/year_scaler.joblib')
        

    def data_cleaning(self, df_1):

        ## 1.1 - Renomear colunas
        cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 'StoreType',
                    'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
                    'Promo2SinceYear', 'PromoInterval']
        
        snake_case = lambda x: inflection.underscore(x)

        cols_new = list(map(snake_case, cols_old))

        # Rename
        df_1.columns = cols_new

        # Data Types
        df_1['date'] = pd.to_datetime(df_1['date'])

        # Fillout NA
        df_1['competition_distance'] = df_1['competition_distance'].apply(lambda x: 200000.0 if math.isnan(x) else x)

        df_1['competition_open_since_month'] = df_1.apply(lambda x: x['date'].month if math.isnan(x['competition_open_since_month']) else x['competition_open_since_month'], axis=1)

        df_1['competition_open_since_year'] = df_1.apply(lambda x: x['date'].year if math.isnan(x['competition_open_since_year']) else x['competition_open_since_year'], axis=1)
        
        df_1['promo2_since_week'] = df_1.apply(lambda x: x['date'].week if math.isnan(x['promo2_since_week']) else x['promo2_since_week'], axis=1)
        
        df_1['promo2_since_year'] = df_1.apply(lambda x: x['date'].year if math.isnan(x['promo2_since_year']) else x['promo2_since_year'], axis=1)

        df_1['promo_interval'].fillna(0, inplace=True)

        month_map = {
            1:'Jan',
            2:'Feb',
            3:'Mar',
            4:'Apr',
            5:'May',
            6:'Jun',
            7:'Jul',
            8:'Aug',
            9:'Sept',
            10:'Oct',
            11:'Nov',
            12:'Dec'
        }

        df_1['month_map'] = df_1['date'].dt.month.map(month_map)

        df_1['is_promo'] = df_1[['promo_interval', 'month_map']].apply(lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split(',') else 0, axis=1)

        # Change Data Type
        df_1['competition_open_since_month'] = df_1['competition_open_since_month'].astype('int64')
        df_1['competition_open_since_year'] = df_1['competition_open_since_year'].astype('int64')

        df_1['promo2_since_week'] = df_1['promo2_since_week'].astype('int64')
        df_1['promo2_since_year'] = df_1['promo2_since_year'].astype('int64')

        return df_1
    

    def feature_engineering(self, df_2):
        # year
        df_2['year']= df_2['date'].dt.year

        # mounth
        df_2['month']= df_2['date'].dt.month

        # day
        df_2['day']= df_2['date'].dt.day

        # week of year
        df_2['week_of_year']= df_2['date'].dt.isocalendar().week

        # year week
        df_2['year_week']= df_2['date'].dt.strftime('%Y-%W')

        df_2['competition_open_since_year'] = df_2['competition_open_since_year'].astype(int)
        df_2['competition_open_since_month'] = df_2['competition_open_since_month'].astype(int)

        df_2['competition_since'] = df_2.apply(lambda x: datetime.datetime(year=x['competition_open_since_year'], month=x['competition_open_since_month'], day=1), axis=1)
        df_2['competition_time_month'] = ((df_2['date'] - df_2['competition_since'])/30).apply(lambda x: x.days).astype(int)

        df_2['promo_since'] = df_2['promo2_since_year'].astype(str) + '-' + df_2['promo2_since_week'].astype(str)
        df_2['promo_since'] = df_2['promo_since'].apply(lambda x: datetime.datetime.strptime(x + '-1', '%Y-%W-%w') - pd.Timedelta(days=7))

        df_2['promo_time_week'] = ((df_2['date'] - df_2['promo_since'])/7).apply(lambda x: x.days).astype(int)

        df_2['assortment'] = df_2['assortment'].apply(lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended')

        df_2['state_holiday'] = df_2['state_holiday'].apply(lambda x: 'public_holiday' if x == 'a' else  'easter_holiday' if x == 'b' else 'christmas' if x == 'c' else 'regular_day')

        # Filtragem de linhas
        df_2 = df_2[df_2['open'] != 0]

        # Seleção de colunas
        cols_drop = ['open', 'promo_interval', 'month_map']
        df_2 = df_2.drop(cols_drop, axis=1)

        return df_2
    

    def data_preparation(self, df_5):
        df_5['competition_time_month'] = df_5['competition_time_month'].astype('int64')
        df_5['promo_time_week'] = df_5['promo_time_week'].astype('int64')
        df_5['week_of_year'] = df_5['week_of_year'].astype('int64')
        df_5['year'] = df_5['year'].astype('int64')
        df_5['month'] = df_5['month'].astype('int64')
        df_5['day'] = df_5['day'].astype('int64')

        # promo_time_week
        df_5['promo_time_week'] = self.promo_time_week_scaler.fit_transform(df_5[['promo_time_week']].values)
        # year
        df_5['year'] = self.year_scaler.fit_transform(df_5[['year']].values)
        # competition_distance
        df_5['competition_distance'] = self.competition_distance_scaler.fit_transform(df_5[['competition_distance']].values)
        # competition_time_month
        df_5['competition_time_month'] = self.competition_time_month_scaler.fit_transform(df_5[['competition_time_month']].values)

        # state_holiday - One Hot Encoding
        df_5 = pd.get_dummies(df_5, prefix=['state_holiday'], columns=['state_holiday'])

        # store_type - Label Encoding
        df_5['store_type'] = self.store_type_scaler.fit_transform(df_5['store_type'])

        # assortment
        # Nessa feature existe um tipo de ordem basic < extended < extra
        assortment_dict = {'basic': 1, 'extra': 2, 'extended': 3}
        df_5['assortment'] = df_5['assortment'].map(assortment_dict)

        # month
        df_5['month_sin'] = df_5['month'].apply(lambda x: np.sin(x * (2. * np.pi/12)))
        df_5['month_cos'] = df_5['month'].apply(lambda x: np.cos(x * (2. * np.pi/12)))

        # day
        df_5['day_sin'] = df_5['day'].apply(lambda x: np.sin(x * (2. * np.pi/30)))
        df_5['day_cos'] = df_5['day'].apply(lambda x: np.cos(x * (2. * np.pi/30)))

        # week_of_year
        df_5['week_of_year_sin'] = df_5['week_of_year'].apply(lambda x: np.sin(x * (2. * np.pi/52)))
        df_5['week_of_year_cos'] = df_5['week_of_year'].apply(lambda x: np.cos(x * (2. * np.pi/52)))

        # day_of_week
        df_5['day_of_week_sin'] = df_5['day_of_week'].apply(lambda x: np.sin(x * (2. * np.pi/7)))
        df_5['day_of_week_cos'] = df_5['day_of_week'].apply(lambda x: np.cos(x * (2. * np.pi/7)))

        cols_selected = ['store', 'promo', 'store_type', 'assortment', 'competition_distance', 'competition_open_since_month',
                        'competition_open_since_year', 'promo2', 'promo2_since_week', 'promo2_since_year', 'competition_time_month', 'promo_time_week',
                        'month_cos', 'month_sin', 'day_cos', 'day_sin', 'week_of_year_cos', 'week_of_year_sin', 'day_of_week_sin', 'day_of_week_cos']
        
        return df_5[cols_selected]
    
    
    def get_prediction(self, model, original_data, test_data):
        # prediction
        pred = model.predict(test_data)
        
        # join pred into the original data
        original_data['prediction'] = np.expm1(pred)
        
        return original_data.to_json(orient='records', date_format='iso')


