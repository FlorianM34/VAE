import matplotlib.pyplot as plt
import argparse

def reconstuct_list(li):
    new_li = []
    for i in range(len(li)):
        if i % 2 == 1:
            new_li.append(li[i])
        print("damn")
    return new_li

def isolate_losses(line):
    splitted_line = line.split()

    kl_loss = 0
    reconstruction_loss = 0
    total_loss = 0

    for i in range(len(splitted_line)) : 
        word = splitted_line[i]
        if word == "kl_loss:":
            kl_loss = float(splitted_line[i+1])
        if word == "loss:":
            total_loss = float(splitted_line[i+1])
        if word == "reconstruction_loss:" and (splitted_line[i+1][-4:] != "here" or splitted_line[i+1][-3:] != "FID" ):
            print(splitted_line[i+1][-3:])
            reconstruction_loss = float(splitted_line[i+1][:4])

    print(kl_loss, reconstruction_loss, total_loss)
    return kl_loss, reconstruction_loss, total_loss


def get_losses(fid):

    reconstruction_loss_list,kl_loss_list,total_loss_list = [], [], []

    with open(f"../VAEs_results/{fid}/VAE_output.log") as f:
        lines = f.readlines()

    print(len(lines))        
    for line in lines:
        if "23750/23750" in line:

            kl_loss, reconstruction_loss, total_loss = isolate_losses(line)  
            kl_loss_list.append(kl_loss)
            reconstruction_loss_list.append(reconstruction_loss)
            total_loss_list.append(total_loss)
    print(len(reconstruction_loss_list), len(kl_loss_list), len(total_loss_list))
    return reconstuct_list(reconstruction_loss_list) ,reconstuct_list(kl_loss_list),reconstuct_list(total_loss_list)





def main(args):


    reconstruction_loss_list,kl_loss_list,total_loss_list = get_losses(args.fid)

    plt.figure(figsize=(10, 5))

    plt.plot(reconstruction_loss_list, label='Reconstruction Loss')
    plt.plot(kl_loss_list, label='KL Loss')
    plt.plot(total_loss_list, label='Total Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.title('Training Losses Over Epochs')
    plt.savefig(f"graphe_loss{args.fid}.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Train a VAE with LSTM on sequence data.")
    parser.add_argument("--fid", type = str)
    args = parser.parse_args()

    main(args)
