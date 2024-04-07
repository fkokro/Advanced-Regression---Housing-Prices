import sys
from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from src.exception import CustomException
from src.logger import logging
import os
import numpy as np
from src.utils import save_object


@dataclass
class DataTransfomationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', 'preporcessor.pkl')

class FillNaN(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype
        
    def fit(self, X, y=None):
        """Seperates dataframe by data type.
        input dtype[str]: Categorical|Numerical
        """
        if self.dtype == 'Categorical':
            self.dataframe = X.select_dtypes(include="object")
            return self
        else:
            self.dataframe = X.select_dtypes(exclude="object")
            return self
    
    def transform(self,X):
        """Fills missing values in dataframe; 'NONE' for categorical dtype, 0 for
        numerical column data type."""
        if self.dtype == 'Categorical':
            self.dataframe.fillna('NONE', inplace=True)
            return self.dataframe
        else:
            self.dataframe.fillna(0, inplace=True)
            return self.dataframe
   
class DataTransfomation:
    def __init__(self):
        self.data_transformation_config=DataTransfomationConfig()
           
    def get_data_transformer_object(self):
        "This function is responsible for data transformation"

        try:
            numerical_columns = ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

            categorical_columns = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

            
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', FillNaN('Numerical')),
                    ('scalar', StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', FillNaN('Categorical')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            
            logging.info('Numerical columns encoding completed')
            logging.info('Caterorical columns encoding completed')
            
            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
    
        except Exception as e:
            raise CustomException(e, sys)

       
    def initiate_data_transformation(self, train_path, test_path):
        
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info('Read train and test data complete.')

            logging.info('Obtaining preprocessing object')
            preprocessor_obj=self.get_data_transformer_object()
            
            target_column_name = 'SalePrice'
            
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

 
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
            logging.info("Preprocessing complete.")
            # train_arr = np.c_[
            #     input_feature_train_arr, np.array(target_feature_train_df)
            # ]
            # test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # logging.info(f"Saved preprocessing object.")

            # save_object(

            #     file_path=self.data_transformation_config.preprocessor_obj_file_path,
            #     obj=preprocessor_obj

            # )

            # return (
            #     train_arr,
            #     test_arr,
            #     self.data_transformation_config.preprocessor_obj_file_path,
            # )

        except Exception as e:
            raise CustomException(e,sys)