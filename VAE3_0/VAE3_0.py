#!/usr/bin/env python
# coding: utf-8

# In[33]:

# VAE_3_0.py

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import csv
from keras import ops
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras import backend as K
import importlib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope, to_categorical
import argparse
from colorama import Fore, Style
import matplotlib.pyplot as plt
import datetime
#tf.config.run_functions_eagerly(True)
# In[35]:



# In[45]:




# In[49]:


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, RepeatVector
from tensorflow.keras.models import Model


    # Glove embedding
def load_glove_embeddings(file_path):

    embeddings_index = {}
    with open(file_path, encoding="utf-8") as file_reader:
        for line in file_reader:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    return embeddings_index

print(f"Found {len(embeddings_index)} word vectors.")

# Encodeur
class Encoder_1(Model):
    def __init__(self, seq_lenght, latent_dim, embedding_dim, input_dim):
        super(Encoder_1, self).__init__()

        self.embedding = Embedding(input_dim = input_dim, output_dim=embedding_dim)
        self.lstm1 = Bidirectional(LSTM(128, return_sequences=True))
        self.dropout1 = Dropout(0.2)
        self.lstm2 = Bidirectional(LSTM(64, return_sequences=False))
        self.dropout2 = Dropout(0.2)
        self.dense_mu = layers.Dense(latent_dim)
        self.dense_logvar = layers.Dense(latent_dim)

        self.seq_lenght = seq_lenght
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim

    def call(self, x):
        
        x = self.embedding(x)
        x = self.lstm1(x)
        x = self.dropout1(x)
        x = self.lstm2(x)
        x = self.dropout2(x)

        mu = self.dense_mu(x)
        logvar = self.dense_logvar(x)
        return mu, logvar


    def to_string(self):
        print("\n")
        print("EncoderA Modele")
        print("    Embedding(input_dim = input_dim, output_dim=embedding_dim)")
        print("    Bidirectional(LSTM(128, return_sequences=True))")
        print("    Dropout(0.2)")
        print("    Bidirectional(LSTM(64, return_sequences=False))")
        print("    Dropout(0.2)")
        print("\n")


class Encoder_2(Model):
    def __init__(self, seq_lenght, latent_dim, embedding_dim, input_dim):
        super(Encoder_2, self).__init__()

        self.embedding = Embedding(input_dim = input_dim, output_dim = embedding_dim)
        self.bilstm = Bidirectional(LSTM(128, return_sequences = False))
        self.dropout = Dropout(0.2)
        self.dense_mu = Dense(latent_dim)
        self.dense_logvar = Dense(latent_dim)
        self.act = ELU()

        self.seq_lenght = seq_lenght
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim

    def call(self, x):
        x = self.embedding(x)
        x = self.bilstm(x)
        x = self.dropout(x)
        x = self.act(x)

        mu = self.dense_mu(x)
        logvar = self.dense_logvar(x)
        return mu, logvar

    def to_string(self):
        print("\n")
        print("EncoderB Modele")
        print("    Embedding(input_dim = input_dim, output_dim = embedding_dim)")
        print("    Bidirectoinal(LSTM(128, return_sequences = False))")
        print("    Dropout(0.2)")
        print("    self.act = ELU()")
        print("\n")


class Encoder_3(Model):
    def __init(self, seq_lenght, latent_dim, embedding_dim, input_dim, embedding_matrix):
        super(Encoder_3, self).__init__()

        self.embedding = Embedding( input_dim = input_dim,
                                    output_dim = embedding_dim,
                                    weights=[embedding_matrix],
                                    trainable = False)

        self.bilstm = Bidirectional(LSTM(128, return_sequences = False))
        self.dropout = Dropout(0.2)
        self.dense_mu = Dense(latent_dim)
        self.dense_logvar = Dense(latent_dim)
        self.act = ELU()

    def call(self, x):

        x = self.embedding(x)
        x = self.bilstm(x)
        x = self.dropout(x)
        x = self.act(x)

        mu = self.dense_mu(x)
        logvar = self.dense_logvar(x)
        return mu, logvar

    def to_string(self):
        print("\n")
        print("EncoderC Modele")
        print("self.embedding = Embedding( input_dim = input_dim,\n
                                    output_dim = embedding_dim,\n
                                    weights=[embedding_matrix],\n
                                    trainable = False")
        print("    Bidirectoinal(LSTM(128, return_sequences = False))")
        print("    Dropout(0.2)")
        print("    self.act = ELU()")
        print("\n")

# Décodeur
class Decoder_1(Model):
    def __init__(self, seq_lenght, latent_dim, embedding_dim, input_dim):
        super(Decoder_1, self).__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.seq_lenght = seq_lenght
        self.input_dim = input_dim
       
        self.decoder_input = Input(shape = (latent_dim, )) #8
        self.repeat = RepeatVector(seq_lenght) #osf
        self.lstm_dropout = LSTM(96, return_sequences = True, recurrent_dropout = 0.2)
        self.dense_output = Dense(input_dim, activation='softmax')

    def call(self, x):
        x = self.repeat(x)
        x = self.lstm_dropout(x)
        x = self.dense_output(x)
        return x

    def to_string(self):
        print("\n")
        print("DecoderA modele : ") 
        print("     Input(shape = (latent_dim, ))")
        print("     RepeatVector(seq_length)")
        print("     LSTM(96, return_sequences = True, recurrent_dropout = 0.2)")
        print("     Dense(input_dim, activation='softmax')")
        print("\n")





class Decoder_2(Model):
    def __init__(self, seq_lenght, latent_dim, embedding_dim, input_dim):
        super(Decoder_2, self).__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.seq_lenght = seq_lenght
        self.input_dim = input_dim

        self.repeat = RepeatVector(seq_lenght) #osf
        self.lstm1 = LSTM(96, return_sequences=True, input_shape=(seq_lenght, embedding_dim))
        self.dropout1 = Dropout(0.2)
        self.lstm2 = LSTM(64, return_sequences=True)
        self.dropout2 = Dropout(0.2)
        self.dense_output = Dense(input_dim, activation='softmax')

    def call(self, x):
        x = self.repeat(x)
        x = self.lstm1(x)
        x = self.dropout1(x)
        x = self.lstm2(x)
        x = self.dropout2(x)
        x = self.dense_output(x)
        return x

    def to_string(self):
        print("\n")
        print("DecoderB modele : ")
        print("    Input(shape = (latent_dim, ))")
        print("    LSTM(96, return_sequences=True, input_shape=(seq_length, embedding_dim))")
        print("    Dropout(0.2)")
        print("    LSTM(64, return_sequences=True)")
        print("    Dropout(0.2) Dense(input_dim, activation='softmax')")
        print("\n")

# VAE
class VAE(Model):

    def __init__(self, encoder, decoder, kl_weight):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = kl_weight
        self.total_loss_list = []
        self.reconstruction_loss_list = []
        self.kl_loss_list = []


    def call(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed


    def reparameterize(self, mu, logvar):
        eps = tf.random.normal(shape=tf.shape(mu))
        return eps * tf.exp(logvar * 0.5) + mu

    #@tf.function change nothing...

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            mu, logvar = self.encoder(data)
            z = self.reparameterize(mu, logvar)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(data, reconstruction))
            kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))
            
            # Adjust the weight for the KL loss

            total_loss = reconstruction_loss + self.kl_weight * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {"loss": total_loss, "reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss}


    def get_losses(self):
        return self.kl_loss_list, self.reconstruction_loss_list, self.total_loss_list

"OK"


# In[53]:

def format_list_of_numbers(li):
    formatted_list = []
    for item in li:
        formatted_number = f"{item:.5f}"
        formatted_list.append(formatted_number)
    return formatted_list

def without_zero(li):
    li_without_zero = []
    for item in li:
        if item == 0:
            continue
        else:
            li_without_zero.append(item)
    return li_without_zero

def write_to_csv(file_reader, date, result_name, output_name, data_size, epochs, kl_weight, encoder, decoder, kl_loss, reconstruction_loss, total_loss):
    file_reader.seek(0, 2)  
    if file_reader.tell() == 0:  
        writer = csv.DictWriter(file_reader, fieldnames=[
           "date", "result_name", "output_name", "Data_size", "Epochs", "kl_weight", "Encoder", "Decoder", "kl loss", "reconstruction loss", "Total loss"])
        writer.writeheader()
    else:
        writer = csv.DictWriter(file_reader, fieldnames=[
            "date", "result_name", "output_name", "Data_size", "Epochs", "kl_weight", "Encoder", "Decoder", "kl loss", "reconstruction loss", "Total loss"])

    writer.writerow({
        "date": date,
        "result_name": result_name,
        "output_name": output_name,
        "Data_size": data_size,
        "Epochs": epochs,
        "kl_weight": kl_weight,
        "Encoder": encoder,
        "Decoder": decoder,
        "kl loss": kl_loss,
        "reconstruction loss": reconstruction_loss,
        "Total loss": total_loss
    })

def main(args):

    
###DATA TRANSFORMATION ###

    import DataFrame_encoder as DFE
    importlib.reload(DFE)
    df = DFE.DataFrame_encoder("train.txt")
    df.txt_to_csv(txt_path = "train.txt", csv_path = "VAE_training.csv", sep = " ")

    df.set_columns(["class", "url"])
    df.convert_result_data_to_binary("class", "_label_positive_", "_label_negative_")
    df.remove_extension("url")
    char_encoder = df.generate_char_encoder(["class"])
    #print(char_encoder)
    df.encoded_data(["class"])
    new_columns =df.get_value(["url"], "class", "1")

    padded_sequence = pad_sequences(new_columns)

    #################
    
    #GloVe Embedding
    glove_path = "glove.6B.50d.txt"
    embeddings_index = load_glove_embeddings(glove_path)
    embedding_matrix = np.zeros((input_dim, embedding_dim))
    
    for char, i in char_encoder.items():
        embedding_vector = embeddings_index.get(char)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


    seq_lenght = 63
    input_dim = 38

    file_reader = open(args.training_result_file_name, "w")

    Encoder1 = Encoder_1(seq_lenght, args.latent_dim, args.embedding_dim, input_dim)
    Encoder2 = Encoder_2(seq_lenght, args.latent_dim, args.embedding_dim, input_dim)
    Encoder3 = Encoder_3(seq_lenght, args.latent_dim, args.embedding_dim, input_dim, embedding_matrix)

    Decoder1 = Decoder_1(seq_lenght, args.latent_dim, args.embedding_dim, input_dim)
    Decoder2 = Decoder_2(seq_lenght, args.latent_dim, args.embedding_dim, input_dim)

    encoders = { 
        "Encoder 1": Encoder1,
        "Encoder 2": Encoder2,
        "Encoder 3": Encoder3,
    }   
    decoders = { 
        "Decoder 1": Decoder1,
        "Decoder 2": Decoder2,
    }
    selected_encoder_class = list(encoders.values())[args.encoder]
    selected_decoder_class = list(decoders.values())[args.decoder]

    encoder = selected_encoder_class
    decoder = selected_decoder_class 

    vae = VAE(encoder, decoder, args.kl_weight)

    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=lambda y_true, y_pred: 0.0)
    vae.fit(
        padded_sequence[:args.data_size], 
        padded_sequence[:args.data_size], 
        epochs = args.epochs, 
        batch_size = args.batch_size,  
        validation_split = 0.2
    )


    kl_loss_list, reconstruction_loss_list, total_loss_list = vae.get_losses() #faire un pyplot des list

    # Plot the losses
    '''
    plt.figure(figsize=(10, 5))

    plt.plot(reconstruction_loss_list, label='Reconstruction Loss')
    plt.plot(kl_loss_list, label='KL Loss')
    plt.plot(total_loss_list, label='Total Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.title('Training Losses Over Epochs')
    plt.savefig(f"graphe_loss{data_size}{kl_weight}.png")
    plt.show()
    '''
    prediction = vae.predict(padded_sequence[:args.data_size_prediction])

    for i in range(args.data_size_prediction):
            
        actuals_decoded_list = df.list_decoder(char_encoder, padded_sequence[i] )
            
        actuals_decoded_word = "".join(actuals_decoded_list)

        file_reader.write(f"{i} - Actuals : {actuals_decoded_word}\n\n")

        argmax_prediction = tf.argmax(prediction[i], axis = -1).numpy()

        prediction_decoded_list = df.list_decoder(char_encoder, argmax_prediction )

        prediction_decoded_word = "".join(prediction_decoded_list)

        file_reader.write(f"{i} - Prediction : {prediction_decoded_word}\n\n\n")
            
        
    encoder.save("encoder_model3_0.keras")
    decoder.save("decoder_model3.keras")
    print(f"\nyou can see your training result at " + Fore.CYAN + f">>> cat {args.training_result_file_name}\n"+ Style.RESET_ALL)
    print("models saved successfully\n")
     
    with open("vae_performance_metrics.csv", "a", newline = "") as csv_file_reader:

        x = datetime.datetime.now()
        date = x.strftime("%c")

        write_to_csv(csv_file_reader,date, args.training_result_file_name, args.output_file_name, args.data_size, args.epochs, args.kl_weight, args.encoder, args.decoder, "kl_loss_list[len(kl_loss_list)-1]", "reconstruction_loss_list[len(reconstruction_loss_list) - 1]","total_loss_list[len(total_loss_list) -1 ]")  

    print("The performance has been saved to the file " + Fore.CYAN + "vae_performance_metrics.csv\n" + Style.RESET_ALL)
    print("You can load the CSV performance metrics in your terminal with this command " + Fore.CYAN + ">>> bash read_csv.bash " +Style.RESET_ALL + "or this command" +Fore.CYAN+" >>> column -s, -t < vae_performance_metrics.csv | less -#2 -N -S" + Style.RESET_ALL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Train a VAE with LSTM on sequence data.")
    parser.add_argument("--data_size", type = int, default = 10_000, help="Size of the data size for training, default is 10_000")
    parser.add_argument("--data_size_prediction", type = int, default = 10, help="Size of the data size for prediction, default is 10")
    parser.add_argument("--epochs", type = int, default = 1, help = "Number of epoch for the Model, default is 1")
    parser.add_argument("--batch_size", type = int, default = 32, help = "Batch size, default is 32")
    parser.add_argument("--embedding_dim", type = int, default = 50, help = "Embedding dim, default is 50")
    parser.add_argument("--latent_dim", type = int, default = 2, help = "latent dim, default is 8")
    parser.add_argument("--kl_weight", type = float, default = 0.1, help = "Coefficient for kl_loss to minimize the normal standard law correction, default is 0.1")
    parser.add_argument("--encoder", type = int, default = 1)
    parser.add_argument("--decoder", type = int, default = 1)

    parser.add_argument("--training_result_file_name",type = str, default = "training_result_default.txt")
    parser.add_argument("--output_file_name",type = str, default = "output_default.txt")
    args = parser.parse_args()

    main(args)
