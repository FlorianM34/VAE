import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import LeaveOneOut, KFold, cross_val_score, ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Bidirectional, LSTM
from tensorflow.keras.models import load_model
import argparse

import sys
import importlib
sys.path.append("/Users/florianmauger/Documents/IGH_Stage/Machine_Learning_Model_Test/PYTHON-SCRIPT")
import DataFrame_encoder as DFE
importlib.reload(DFE)

def check_args(args):
    if not args.txt_path or not args.csv_path:
        raise ValueError("Missing two positional arguments --txt_path and --csv_path")


def main(args):

    df = DFE.DataFrame_encoder(args.txt_path)

    df.txt_to_csv(csv_path = args.csv_path, txt_path = args.txt_path, sep=" ")

    df.set_columns(["class", "encoded_url"])

    df.convert_result_data_to_binary("class","_label_positive_", "_label_negative_")

    df.remove_extension("encoded_url")

    df.generate_char_encoder(["class"])

    df.encoded_data(["class"])

    df.get_head()

    df = df.to_csv()

    df.head()

    ### ALL THE IMPORTS ###

    X = pad_sequences(df["encoded_url"])
    y = df["class"]

    X.shape

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.25, random_state = 42)

    X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape


    np.max(X_test) + 1

    X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape

    np.max(X_train) + 1

    X_test[1]

    model = Sequential()
    model.add(Input(shape=(63,)))  
    model.add(Embedding(input_dim=38, output_dim=64))  
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))

    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))


    # FAIRE DE L'INFÉRANCE QUAND LE MODELE SERA BIEN 

    cp = ModelCheckpoint('model.keras', save_best_only = True)

    model.compile(loss = MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])


    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, callbacks=[cp])

    model = load_model('model22.keras')

    train_predictions = model.predict(X_test).flatten()

    train_results = pd.DataFrame(data={"Train Prediction": train_predictions, "Actuals": y_test, })
    output_file = "train_results.csv"
    train_results.to_csv(output_file, index=False)

    y_test_pred_rounded = (train_predictions > 0.5).astype(int)

    test_accuracy = accuracy_score(y_test, y_test_pred_rounded)
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
