import argparse
import os
from VAE2_0 import Encoder_1, Encoder_2, Encoder_3, Decoder_1, Decoder_2
import random
from colorama import Fore, Style

def random_file_id():
    return random.randint(1000,9999)

def get_user_input(prompt, input_type, default_value=None):
    while True:
        user_input = input(prompt)
        if user_input == "":
            return default_value
        try:
            return input_type(user_input)
        except ValueError:
            print(f"Please enter a valid {input_type.__name__}.")

def get_user_choice(prompt, options):
    for i in range(len(options)):
        print(Fore.RED + f"{i + 1}." + Style.RESET_ALL) 
        print(Fore.CYAN)
        print(options[i].to_string())
    print(Style.RESET_ALL)
    while True:
        choice = input(prompt)
        try:
            choice = int(choice)
            if 1 <= choice <= len(options):
                return choice - 1
            else:
                print(f"Please enter a number between 1 and {len(options)}.")
        except ValueError:
            print("Please enter a valid number.")



def main():
    # Continue with VAE training and prediction using the selected encoder and decoder
    data_size = get_user_input("Enter the size of the data for training (default is 10_000 max is 950_000 ): ", int, default_value=10000)
    data_size_prediction = get_user_input("Enter the size of the data for prediction (default is 10): ", int, default_value=10)
    epochs = get_user_input("Enter the number of epochs for the model (default is 1): ", int, default_value=1)
    batch_size = get_user_input("Enter the batch size (default is 32): ", int, default_value=32)
    embedding_dim = get_user_input("Enter the embedding dimension (default is 2): ", int, default_value=2)
    latent_dim = get_user_input("Enter the latent dimension (default is 8): ", int, default_value=8)
    kl_weight = get_user_input("Enter the coefficient for KL loss (default is 0.1): ", float, default_value=0.1)
    random_id = random_file_id()

    print(f"Data size for training: {data_size}")
    print(f"Data size for prediction: {data_size_prediction}")
    print(f"Number of epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Latent dimension: {latent_dim}")
    print(f"KL loss weight: {kl_weight}")

    seq_lenght = 63
    input_dim = 38

    Encoder1 = Encoder_1(seq_lenght, latent_dim, embedding_dim, input_dim) 
    Encoder2 = Encoder_2(seq_lenght, latent_dim, embedding_dim, input_dim)
    Encoder3 = Encoder_3(seq_lenght, latent_dim, embedding_dim, input_dim)

    Decoder1 = Decoder_1(seq_lenght, latent_dim, embedding_dim, input_dim)
    Decoder2 = Decoder_2(seq_lenght, latent_dim, embedding_dim, input_dim)

    encoders = {
        "Encoder 1": Encoder1,
        "Encoder 2": Encoder2,
        "Encoder 3": Encoder3,
    }
    decoders = {
        "Decoder 1": Decoder1,
        "Decoder 2": Decoder2,
    }

    print(Fore.RED + "Available Encoders:\n" + Style.RESET_ALL)
    encoder_choice= int(get_user_choice("Select an encoder by number: ", list(encoders.values())))
    print("Encoder choosed:", encoder_choice)

    print(Fore.RED + "Available Decoders:\n" + Style.RESET_ALL)
    decoder_choice = int(get_user_choice("Select a decoder by number: ", list(decoders.values())))
    print("decoder choosed : ", decoder_choice)


    print(f"Selected Encoder: {list(encoders.keys())[encoder_choice]}")
    print(f"Selected Decoder: {list(decoders.keys())[decoder_choice]}")
    

    output_file_name = get_user_input("Enter the output file name this will contain all the errors if one occured and all the information durring the process ( default is VAE_output + random_id + .log : ", str, default_value = "VAE_output" +str(random_id) + ".log")
    training_result_file_name = get_user_input("Enter the training result filename  ( default is training_result + random_id + .txt : ", str, default_value = "training_result"+str(random_id) + ".txt")


    command = f"nohup python VAE3_0.py --data_size {data_size} --data_size_prediction {data_size_prediction} --epochs {epochs} --batch_size {batch_size} --embedding_dim {embedding_dim} --latent_dim {latent_dim} --kl_weight {kl_weight} --encoder {encoder_choice} --decoder {decoder_choice} --training_result_file_name {training_result_file_name} --output_file_name {output_file_name}> {output_file_name} 2>&1 &"

    try:
        os.system(command)
        print(Fore.CYAN + f"The training as been launch you can watch it with >>>tail -f {output_file_name}" + Style.RESET_ALL)

        
    except:
        print("An error occured please pass the rights arguments")


if __name__ == "__main__":
    main()

