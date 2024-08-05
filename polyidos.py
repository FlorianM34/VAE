import pandas as pd
from sklearn.svm import SVC
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
import argparse
import sys
import importlib
import numpy as np

sys.path.append("/Users/florianmauger/Documents/IGH_Stage/Machine_Learning_Model_Test/PYTHON-SCRIPT")
import DataFrame_encoder as DFE
importlib.reload(DFE)

def check_args(args):
    if not args.txt_path or not args.csv_path :
        raise ValueError("Missing two positional arguments --txt_path --csv_path")
def without_zero(li):
    li_without_zero = []
    for item in li:
        if item == 0:
            continue
        else:
            li_without_zero.append(item)
    return li_without_zero

def zero_counter(li):
    zero_counter = 0
    for nb in li:
        if nb == 0:
            zero_counter += 1
    return zero_counter

def main(args):

    df = DFE.DataFrame_encoder(args.txt_path)
    

    df.txt_to_csv(csv_path = args.csv_path, txt_path = args.txt_path, sep=" ")

    df.set_columns(["class", "encoded_url"])

    df.get_head()

    df.convert_result_data_to_binary("class","_label_positive_", "_label_negative_")


    char_encoder = df.generate_char_encoder(["class"])

    df.encoded_data(["class"])

    df.get_head()

    df_csv = df.to_csv()

    df_csv.head()

    # padded the sequence into smtg like (None, 63)

    X = pad_sequences(df_csv["encoded_url"])
    y = df_csv["class"]

    print(type(X))
    print(X.shape)
    model = load_model('polyidos.keras')

    decoded_data = []
    
    X_filtered = []


    for list_encoded in X:
        if zero_counter(list_encoded) == len(list_encoded) or zero_counter(list_encoded) < 12:
            pass
        else:
            X_decoded = df.list_decoder(char_encoder, list_encoded)
            X_decoded_joined = "".join(X_decoded)
            decoded_data.append(X_decoded_joined)
            X_filtered.append(list_encoded)

    X_filtered = np.array(X_filtered)
    y_filtered = np.array([0 for _ in range(X_filtered.shape[0])])
    print(y_filtered.shape)

    print(X_filtered.shape)


    train_predictions = model.predict(X_filtered).flatten()
    
    train_results = pd.DataFrame(data={"Train Prediction": train_predictions, "Actuals": y_filtered, "decoded_data" : decoded_data,  })
    output_file = args.csv_path
    train_results.to_csv(output_file, index=False)

    y_test_pred_rounded = (train_predictions > 0.5).astype(int)

    test_accuracy = accuracy_score(y_filtered, y_test_pred_rounded)
    print(f'Test Accuracy: {test_accuracy}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--txt_path", type = str)
    parser.add_argument("--csv_path", type = str)

    args = parser.parse_args()

    try :
        check_args(args)
        main(args)
    except ValueError as e:
        print(f"Error {e}")
