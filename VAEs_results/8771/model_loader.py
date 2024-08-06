import sys
import csv
import numpy as np
sys.path.append("../../")
from VAE2_0 import Encoder_1, Encoder_2, Encoder_3, Encoder_4, Encoder_5, Decoder_1, Decoder_2, VAE
import DataFrame_encoder as DFE
import os    
import argparse
import tensorflow as tf

def get_modele_data_by_fid(file_reader, file_id):
    with open(file_reader, mode='r', newline='') as fichier:
        lecteur = csv.DictReader(fichier)
        for ligne in lecteur:
            if ligne['File_id'] == file_id:
                return ligne

    print(f"{file_id} not found")
    return None

def get_encoder(encoder_id, seq_lenght, input_dim, latent_dim, embedding_dim):

    match encoder_id:
        case "1":
            return Encoder_1(seq_lenght, latent_dim, embedding_dim, input_dim)
        case "2":
            return Encoder_2(seq_lenght, latent_dim, embedding_dim, input_dim)
        case "3":
            return Encoder_3(seq_lenght, latent_dim, embedding_dim, input_dim)
        case "4":
            return Encoder_4(seq_lenght, latent_dim, embedding_dim, input_dim)
        case "5":
            return Encoder_5(seq_lenght, latent_dim, embedding_dim, input_dim)

def get_decoder(decoder_id, seq_lenght, input_dim, latent_dim, embedding_dim):

    match decoder_id:
        case "1":
            return Decoder_1(seq_lenght, latent_dim, embedding_dim, input_dim)
        case "2":
            return Decoder_2(seq_lenght, latent_dim, embedding_dim, input_dim)

def generate_new_sequences(decoder, num_samples=10, latent_dim=8):
    random_latent_vectors = np.random.normal(size=(num_samples, latent_dim))

    generated_sequences = decoder.predict(random_latent_vectors)

    return generated_sequences

def main(fid, args):
    modele_data = get_modele_data_by_fid("../../vae_performance_metrics.csv", fid)
    predict_size = args.predict_size

    seq_lenght = 63
    input_dim = 38
    embedding_dim = int(modele_data["Embedding_dim"])
    latent_dim = int(modele_data["Latent_dim"])
    kl_weight = float(modele_data["kl_weight"])

    encoder_id = modele_data["Encoder"]
    decoder_id = modele_data["Decoder"]
    
    encoder = get_encoder(encoder_id, seq_lenght, latent_dim, embedding_dim, input_dim)
    decoder = get_decoder(decoder_id, seq_lenght, latent_dim, embedding_dim, input_dim)

    encoder.load_weights('encoder_weight.weights.h5')
    decoder.load_weights('decoder_weight.weights.h5')

    generated_sequences = generate_new_sequences(decoder, num_samples = predict_size)


    file_reader = open("predictions.txt", "w")

    df = DFE.DataFrame_encoder("../../train.txt")

    df.txt_to_csv(txt_path = "../../train.txt", csv_path = "../../VAE_training.csv", sep = " ")

    df.set_columns(["class", "url"])

    df.convert_result_data_to_binary("class", "_label_positive_", "_label_negative_")

    df.remove_extension("url")

    char_encoder = df.generate_char_encoder(["class"])

    for i in range(predict_size):

        argmax_prediction = tf.argmax(generated_sequences[i], axis = -1).numpy()

        prediction_decoded_list = df.list_decoder(char_encoder, argmax_prediction )

        prediction_decoded_word = "".join(prediction_decoded_list)

        file_reader.write(f"_label_negative_ {prediction_decoded_word}\n\n")

    print("Perdictions have been written succefully")

if __name__ == "__main__":
    absolute_path = os.path.abspath(__file__)

    current_path = os.path.dirname(absolute_path)

    splited_path = current_path.split("/")

    fid = splited_path[len(splited_path)-1]
    print(fid)

    parser = argparse.ArgumentParser(description = "Train a VAE with LSTM on sequence data.")

    parser.add_argument("--predict_size", type = int, default = 1000)

    args = parser.parse_args()
    

    main(fid, args)
