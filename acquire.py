from env import get_connection
import pandas as pd
import os

def get_titanic_data():
    
    '''
    This function is used to get titanic data from sql database.
    '''
    
    if os.path.isfile('titanic.csv'):
        
        return pd.read_csv('titanic.csv')
    
    else:
        
        url = get_connection('titanic_db')
        query = '''SELECT * FROM passengers'''
        df = pd.read_sql(query, url)
        df.to_csv('titanic.csv')
        return df

def get_iris_data(get_connection):
    
    '''
    This function is used to get iris data from sql database.
    '''
    
    if os.path.isfile('iris.csv'):
        
        return pd.read_csv('iris.csv')
    
    else:
        
        url = get_connection('iris_db')
        query = '''
                SELECT * FROM measurements 
                JOIN species USING(species_id)
                '''
        df = pd.read_sql(query, url)
        df.to_csv('iris.csv')
        return df

def get_telco_data(get_connection):
    
    '''
    This function is used to get titanic data from sql database.
    '''
    
    if os.path.isfile('telco.csv'):
        
        return pd.read_csv('telco.csv')
    
    else:
        
        url = get_connection('telco_churn')
        query = '''SELECT * FROM customers
                    JOIN internet_service_types USING(internet_service_type_id)
                    JOIN contract_types USING(contract_type_id)
                    JOIN payment_types USING(payment_type_id)
                    '''
        df = pd.read_sql(query, url)
        df.to_csv('telco.csv')
        return df