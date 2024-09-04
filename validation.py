import os
import pandas as pd

class Validation:
    
    @staticmethod
    def validate_str_alnum(value:str):
        return isinstance(value,str) and value.replace(' ','').isalnum()


    @staticmethod        
    def read_in_int_value_0_1(message: str):
        user_input = None
        while user_input not in [0, 1]:
            user_input = input(message)
            if user_input.isnumeric() and int(user_input) in [0, 1]:
                return int(user_input)

    
    @staticmethod
    def validate_csv_filename(filename):
        if filename.endswith('.csv'):
            return True
        else:
            print("Invalid filename")

    @staticmethod
    def read_in_value(validation_function,message:str):
        while True:
            user_input = input(message)
            if validation_function(user_input):
                return user_input

    @staticmethod
    def validate_df(df):
        '''Checks DF for missing values and d-type object, 
            if not found Return True'''
        if df.isnull().values.any():
            raise ValueError("[i] Data frame have missing values")      
        if any(df.dtypes == "object"):
            raise ValueError("[i] Data frame contains d-types object") 
        return True
    
    @staticmethod
    def validate_bool(value:bool):
        '''Check if bool, Returns True if so'''
        return isinstance(value,bool)

    @staticmethod
    def validate_tuple(value:tuple):
        '''Checks Tuple, if positive numbers are integers,
            and negative numbers are in range of -1 and 0, Returns True'''
        for i in value:
            if i > 0 and isinstance(int):
                return isinstance(value,tuple)
            else:
                return False
        for i in value:
            if -1 < i < 0 and isinstance(float):
                return isinstance(value,tuple)
            else:
                return False

    @staticmethod
    def validate_int(value:int):
        '''Validate value is Integer, Returns True if so'''
        return isinstance(value,int)
    
    @staticmethod
    def validate_str(value:str):
        '''Validate value is String, Returns True if so'''
        return isinstance(value,str)

    #@staticmethod
    #def validate_csv_filename(filename):
    #    '''Checks if filepath is existing and ends with ".csv",
    #        Returns True if so'''
    #    if isinstance(filename, str) and filename.endswith('.csv'):
    #        if os.path.isfile(filename):
    #            return True
    #        else:
    #            print("File does not exist.")
    #            return False
    #    else:
    #        print("Invalid CSV filename")
    #        return False

