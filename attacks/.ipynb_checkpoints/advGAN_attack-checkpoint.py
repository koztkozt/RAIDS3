from sklearn.model_selection import train_test_split
import os, sys, importlib
import numpy as np

np.random.seed(0)
import torch

torch.manual_seed(0)
import pandas as pd
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attacks.advGAN.advGAN import AdvGAN_Attack
from attacks.advGAN.advGAN_Uni import AdvGAN_Uni_Attack

from stage1.model import stage1
from stage2.data import stage2_data
from stage2.model import stage2
from stage1.model import stage1


def advGAN_Attack(dataset_name, stage1, stage2, train_dataset, config, universal=False):
    image_nc = config.num_channels
    epochs = config.num_epoch
    batch_size = config.batch_size
    dataset_name = config.dataset_name
    image_size = (config.img_height, config.img_width)
    target = config.target
    
    BOX_MIN = 0
    BOX_MAX = 1
    # target = 0.2

    if not universal:
        advGAN = AdvGAN_Attack(device, stage1, stage2, dataset_name, target, image_nc, BOX_MIN, BOX_MAX, batch_size)
    else:
        advGAN = AdvGAN_Uni_Attack(
            device, stage1, stage2, dataset_name, image_size, target, image_nc, BOX_MIN, BOX_MAX, batch_size
        )

    advGAN.train(train_dataset, epochs)
    return advGAN


if __name__ == "__main__":
    configfile = importlib.import_module(sys.argv[1])
    config = configfile.advGANConfig()
    dirname = os.path.dirname(os.path.abspath(__file__))
    dirparent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    attack_name = "advGAN"
    dataset_name = config.dataset_name
    data_path = config.data_path
    
    stage1_path = os.path.join(dirparent, "stage1/stage1_" + dataset_name + ".pt")
    if sys.argv[2] == "none":
        stage2_path = os.path.join(dirparent, "stage2/stage2_" + dataset_name + "_" + "abrupt" + ".pt")
    else:
        stage2_path = os.path.join(dirparent, "stage2/stage2_" + dataset_name + "_" + sys.argv[2] + ".pt")

    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    stage1_model = stage1().to(device)
    stage1_model.load_state_dict(torch.load(stage1_path))
    stage1_model.eval()
    stage2_model = stage2().to(device)
    stage2_model.load_state_dict(torch.load(stage2_path))
    stage2_model.eval()

    print("Loading training data...")
    X = np.load(dirparent + "/" + data_path + "X_train.npy")
    Y = pd.read_csv(dirparent + "/" + data_path + "/Y_train_attack_" + sys.argv[2] + ".csv")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=56)

    print("Creating model...")
    train_dataset = stage2_data(X_train, Y_train)

    print("Start advGAN training")
    advGAN = advGAN_Attack(dataset_name, stage1_model, stage2_model, train_dataset, config)
    torch.save(advGAN.netG.state_dict(), "./models/" + dataset_name + "_" + attack_name + "_netG_epoch_32.pth")

    print("Start advGAN_uni training")
    advGAN_uni = advGAN_Attack(dataset_name, stage1_model, stage2_model, train_dataset, config, universal=True)
    advGAN_uni.save_noise_seed("./models/" + dataset_name + "_" + attack_name + "U_noise_seed.npy")
    torch.save(advGAN_uni.netG.state_dict(), "./models/" + dataset_name + "_" + attack_name + "U_netG_epoch_32.pth")

    print("Finish training")
