import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def train_val_test(df, target, seed=42):
    
    '''
    This function is used to split the data into train, validate, and test variables
    '''
    
    train, val_test = train_test_split(df, train_size=0.7, random_state=seed, stratify=df[target])
    
    val, test = train_test_split(val_test, train_size=0.5, random_state=seed, stratify=val_test[target])
    
    return train, val, test




def prep_titanic(titanic):
    
    '''
    This function is used to drop unnecessary columns for the titanic data 
    and create dummies for the sec and embark_town columns
    '''
    
    titanic.drop(columns=['class','embarked', 'passenger_id', 'deck', 'age', 'Unnamed: 0'], inplace=True)
    
    titanic_dummies = pd.get_dummies(titanic[['sex', 'embark_town']], drop_first=True)
    titanic = pd.concat([titanic, titanic_dummies], axis=1)
    
    return titanic



def prep_telco(telco):
    
    '''
    This function is used to drop unnecessary columns, convert the total_charges column to a float
    and create dummies for the object columns for better data manipulation later
    '''
    
    telco.drop(columns=['Unnamed: 0', 'payment_type_id', 'contract_type_id', 
                        'internet_service_type_id', 'customer_id'], inplace=True)
    
    telco['total_charges'] = (telco['total_charges'] + '0').astype('float')

    
    telco_dummies = pd.get_dummies(telco[['gender', 'partner', 'dependents', 
                                      'phone_service', 'multiple_lines', 
                                      'online_security', 'online_backup', 
                                      'device_protection', 'tech_support', 
                                      'streaming_tv', 'streaming_movies', 
                                      'paperless_billing', 'churn', 'internet_service_type', 
                                      'contract_type', 'payment_type']], drop_first=True)
    
    telco = pd.concat([telco, telco_dummies], axis=1)
    
    return telco




def prep_iris(iris):
    
    '''
    This function is used to drop unecessary columns, rename the species_name column
    and created dummies for the species column
    '''
    
    iris.drop(columns=['species_id', 'measurement_id', 'Unnamed: 0'], inplace=True)
    
    iris.rename(columns={'species_name':'species'}, inplace=True)
    
    iris_dummies = pd.get_dummies(iris[['species']], drop_first=True)
    iris = pd.concat([iris, iris_dummies], axis=1)
    
    return iris