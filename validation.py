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
