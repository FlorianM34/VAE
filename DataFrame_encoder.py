import os
import pandas as pd

class DataFrame_encoder :

    def __init__(self, file_path):
        self.file_reader = open(file_path)
        self.file_path = file_path
        self.encoded = {}
        self.new_file_encoded = ""
        self.new_file_encoded_path = ""



    def list_decoder(self, char_encoder, argmax_prediction):

        char_decoder = {v: k for k, v in char_encoder.items()}
        
        decoded_list = [char_decoder[idx] for idx in argmax_prediction if idx != 0]
        
        return decoded_list

    def list_decoder(self, char_encoder, argmax_prediction):

        char_decoder = {v: k for k, v in char_encoder.items()}
        
        decoded_list = [char_decoder[idx] for idx in argmax_prediction if idx != 0]
        
        return decoded_list

        
    
    def _feature_to_binary(self, feature, positive_feature, negative_feature):

        if feature == positive_feature:
            return 0
        elif feature == negative_feature:
            return 1
        else :
            raise ValueError(f"Invalid string: {feature}")
            
            
    def clean_file(self, input_file_path, output_file_path):
        with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
            for line in infile:
                if len(line.split()) > 1:
                    outfile.write(line)

    def txt_to_csv(self, csv_path, txt_path = "__default__", index = False, sep = ","):

        if txt_path == "__default__":
            txt_path = self.file_path
        
        read_file = pd.read_csv(txt_path)
        read_file.to_csv(csv_path, index = index)

        self.file_reader = pd.read_csv(csv_path, sep = sep)
        self.file_path = csv_path
        print("CSV file save to ", csv_path)

    
    def set_columns(self,columns_name_list):
        try :
            self._is_csv()
            self.file_reader.columns = columns_name_list
            self.file_reader.to_csv(self.file_path, index=False)
            print("Columns set")
            print(self.get_head())

        except ValueError as e :
            print(r"Error : {e}")


    def _remove_extension(self, url): 
    
        i = 0   
        domain_name = ""
        while url[i] != ".": 
            domain_name += url[i]
            i += 1
        return domain_name

    def remove_extension(self, column):
        column_without_extension =  self.file_reader[column].apply(self._remove_extension)

        self.file_reader[column] = column_without_extension

    def remove_empty_line(self, csv_path):
        df = pd.read_csv(csv_path)

        if 'encoded_url' not in df.columns:
            raise KeyError("La colonne 'encoded_url' n'existe pas dans le fichier CSV.")
        
        df_cleaned = df.dropna(subset=['encoded_url'])
        df_cleaned = df_cleaned[df_cleaned['encoded_url'].str.strip() != '']
    
        df_cleaned.to_csv(csv_path, index=False)

    def generate_char_encoder(self, columns_not_to_encoded):
        char_encoded = {}
        counter = 0
        columns_to_encoded = self.file_reader.drop(columns=columns_not_to_encoded)   

        for column_name in columns_to_encoded :
            column = self.file_reader[column_name]
            for line in column :
                for char in line : 
                    if char not in char_encoded:
                        counter += 1
                        char_encoded[char] = counter

        self.encoded = char_encoded
        return char_encoded


    def convert_result_data_to_binary(self, column_name, positive_feature_name, negative_feature_name):
        print("checkpoint 1")
        print(column_name)
        self.get_head()
        results_column = self.file_reader[column_name]

        print("checkpoint 2")
        counter = 0

        positive_value = positive_feature_name
        negative_value = negative_feature_name
        
        transformed_results = []

        for feature in results_column:
            print("checkpoint 3")
            try:
                if feature == positive_value:
                    transformed_results.append(0)
                elif feature == negative_value:
                    transformed_results.append(1)
                else:
                    raise ValueError(f"Invalid string: {feature}")
                
                counter += 1
                    
                print("checkpoint 4")
            except ValueError as e:
                print(f"Error in convert_result_dat_to_binary : {e}")
                return False
        
        print("checkpoint 5")
        self.file_reader[column_name] = transformed_results

        print("checkpoint 6")
        print(f'Column {column_name} encoded Sucessfully')

        return True
                
        
    
    def _is_csv(self):
        _, file_extension = os.path.splitext(self.file_path)
        if file_extension == ".csv":
            return True
        else :
            raise ValueError(f"Wrong extension: {file_extension}. Excepted .csv. \n Maybe try txt_to_csv.") 
        
    def _is_txt(self):
        _, file_extension = os.path.splitext(self.file_path)
        if file_extension == ".txt":
            return True
        else :
            raise ValueError(f"Wrong extension: {file_extension}. Excepted .txt. \n The default file_path is wrong try to pass it in argument.") 
        
    
    def _has_encoder(self):
        if self.encoded == {}:
            raise ValueError(f"You had to call generate_char_encoder with the right arguments before using this method")
        else : 
            return True

    
    def _create_train_file_encoded(self, columns_not_to_encoded):

        try :
            self._is_csv()
            self._has_encoder()


            columns_to_encoded = self.file_reader.drop(columns=columns_not_to_encoded)   

            for column_name in columns_to_encoded:
                column = self.file_reader[column_name]

                new_column = []
                for i in range(len(column)) :

                    new_word = ""

                    for char in column[i] : 
            
                        new_word += str(self.encoded[char]) + " "
                    new_word = new_word.split()
                    new_word_int = [int(e) for e in new_word]
                    new_column.append( new_word_int )
                self.file_reader[column_name] = new_column
            print("File encoded successfuly ! ")
            return True
    
        except ValueError as e:
            
            print(f"Error {e}")
            return False


    def get_column(self,column_name):
        try :
            self._is_csv
            return self.file_reader[str(column_name)]
        except ValueError as e :
            print(e)
        

    def get_columns(self, columns_name):
        try :
            self._is_csv

            if len(columns_name) == 1:
                return self.file_reader[columns_name[0]]
            return self.file_reader[columns_name]
        except ValueError as e:
            print(e)

    
    def encoded_data(self, columns_not_to_change):
        
                    
        if self._create_train_file_encoded( columns_not_to_change) :
            
            return True
            
        else : 
            
            print("An Error occured when encoding the file")
            
            return False
        
    def get_value(self, returned_columns, check_column , feature_to_check):
        try :
            self._is_csv()

            new_columns = []

            returned_columns = self.get_columns(returned_columns)

            check_column = self.get_column(check_column)

            for i in range(len(check_column)) :
                if str(check_column[i]) == feature_to_check :
                    new_columns.append(returned_columns[i])
            
            return new_columns
        

        except ValueError as e:
            print(e)

    
    def get_df_info(df):
        
        print("\n\003[1mShape of DataFrame;\033[0m ", df.shape)
        print("\n\033[1mColumns in DataFrame:\033[0m ", df.columns.to_list())
        print("\n\033[1mData types of columns:\033[0m\n", df.dtypes)
        
        print("\n\033[1mInformation about DataFrame:\033[0m")
        df.info()
        
        print("\n\033[1mNumber of unique values in each column:\033[0m")
        
        print("\n\033[1mNumber of null values in each column:\033[0m\n", df.isnull().sum())

    
    def get_head(self):
        try:
            self._is_csv
            print(self.file_reader.head())
        except ValueError as e:
            print(f"Error : {e}")
            return False
    

    def get_path(self):
        return self.file_encoded_path
    
    def to_csv(self):
        return self.file_reader
