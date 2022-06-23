from cmath import e
import pandas as pd
import numpy as np
from pathlib import Path
import importlib
import sys
import os
import math
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attacks.attack_test import (
    fgsm_attack_test,
    optimized_attack_test,
    optimized_uni_test,
    advGAN_test,
    advGAN_uni_test,
)
from stage2.data import stage2_data
from stage2.model import stage2
from stage1.model import stage1
from attacks.advGAN_attack import advGAN_Attack
from attacks.optimization_universal_attack import generate_noise
from attacks.optimization_attack import optimized_attack
from attacks.fgsm_attack import fgsm_attack
from attacks.advGAN.models import Generator

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


def attacks(config):
    dataset_name = config.dataset_name
    data_path = config.data_path
    batch_size = config.batch_size
    dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dirparent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
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
    test_dataset = stage2_data(X_test, Y_test)

    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("Length of dataloader :", len(test_data_loader))

    # attack_names = ["fgsm", "opt", "optU", "advGAN", "advGANU"]
    attack_names = ["advGAN"]
    dist_all = pd.DataFrame()
    print("Attacking: RAIDS")

    for i in range(len(attack_names)):
        print("[+] Attacking using", attack_names[i])
        dist, y_true, y_pred, y_pred_adv = attack_test(
            test_data_loader,
            stage1_model,
            stage2_model,
            dataset_name,
            attack_names[i],
            config,
            device,
        )
        result_print(attack_names[i], dataset_name, y_true, y_pred, y_pred_adv)
        dist_all[attack_names[i]] = dist
        print(dist_all.describe())

    dist_all.to_csv("./results/" + dataset_name + "_dist_" + sys.argv[2] + ".csv")
    dist_all.describe().to_csv("./results/" + dataset_name + "_dist_describe_" + sys.argv[2] + ".csv")


def attack_test(
    test_dataset,
    stage1,
    stage2,
    dataset_name,
    attack_name,
    config,
    device,
):
    dataset_name = config.dataset_name
    batch_size = config.batch_size
    target = config.target
    image_nc = config.num_channels
    image_size = (config.img_height, config.img_width)

    total_dist = []
    y_pred, y_true, y_pred_attack = [], [], []

    inter = 0.0
    for i, sample in enumerate(test_dataset):
        if i / len(test_dataset) > inter:
            print(f"attacks completed: {(inter):.0%}")
            inter += 0.1

        plot_fig = True if i % 64 == 0 else False

#         image, angle, speed, target = sample
#         angle_speed = np.array([list(a) for a in zip(angle, speed)])
#         angle_speed = torch.from_numpy(angle_speed)

        image, angle, speed, target, curve = sample
        if not curve:
            continue
        image = image.type(torch.FloatTensor)
        angle = angle.type(torch.FloatTensor)
        speed = speed.type(torch.FloatTensor)
        target = target.type(torch.FloatTensor)
        image = image.to(device)
        target = target.to(device)
        angle_speed = np.array([list(a) for a in zip(angle, speed)])
        angle_speed = torch.from_numpy(angle_speed)
        angle_speed = angle_speed.to(device)

        if attack_name == "fgsm":
            (
                output,
                adv_output,
                pred_angle_speed,
                pred_angle_speed_attack,
                plt_,
                noise,
                perturbed_image,
            ) = fgsm_attack_test(stage1, stage2, image, angle_speed, target, device, image_size, plot_fig)

        elif attack_name == "opt":
            (
                output,
                adv_output,
                pred_angle_speed,
                pred_angle_speed_attack,
                plt_,
                noise,
                perturbed_image,
            ) = optimized_attack_test(stage1, stage2, image, angle_speed, target, device, image_size, plot_fig)

        elif attack_name == "optU":
            noise = np.load(dirparent + "/attacks/models/" + dataset_name + "_" + attack_name + "_noise_seed.npy")
            noise = np.tile(noise, (image.shape[0], 1, 1, 1))
            noise = torch.from_numpy(noise).type(torch.FloatTensor).to(device)
            output, adv_output, pred_angle_speed, pred_angle_speed_attack, plt_, perturbed_image = optimized_uni_test(
                stage1, stage2, image, angle_speed, target, device, noise, image_size, plot_fig
            )

        elif attack_name == "advGAN":
            advGAN_generator = Generator(image_nc, image_nc, attack_name).to(device)
            advGAN_generator.load_state_dict(
                torch.load(dirparent + "/attacks/models/" + dataset_name + "_" + attack_name + "_netG_epoch_32.pth")
            )
            advGAN_generator.eval()
            output, adv_output, pred_angle_speed, pred_angle_speed_attack, plt_, noise, perturbed_image = advGAN_test(
                stage1, stage2, image, angle_speed, target, advGAN_generator, device, image_size, plot_fig
            )

        elif attack_name == "advGANU":
            advGANU_generator = Generator(image_nc, image_nc, attack_name).to(device)
            advGANU_generator.load_state_dict(
                torch.load(dirparent + "/attacks/models/" + dataset_name + "_" + attack_name + "_netG_epoch_32.pth")
            )
            advGANU_generator.eval()
            noise_seed = np.load(dirparent + "/attacks/models/" + dataset_name + "_" + attack_name + "_noise_seed.npy")
            noise_seed = np.tile(noise_seed, (image.shape[0], 1, 1, 1))
            noise = advGANU_generator(torch.from_numpy(noise_seed).type(torch.FloatTensor).to(device))
            output, adv_output, pred_angle_speed, pred_angle_speed_attack, plt_, perturbed_image = advGAN_uni_test(
                stage1, stage2, image, angle_speed, target, device, noise, image_size, plot_fig
            )
            noise = noise.detach().cpu().numpy()

        output_pred = np.round(torch.sigmoid(output.detach().cpu()))
        adv_pred = np.round(torch.sigmoid(adv_output.detach().cpu()))

        y_true.extend(target.detach().cpu().data)
        y_pred.extend(output_pred.reshape(-1))
        y_pred_attack.extend(adv_pred.reshape(-1))

        if plot_fig:
            Path("results/images/" + dataset_name + "/" + sys.argv[2] + "/" + attack_name + "/").mkdir(
                parents=True, exist_ok=True
            )
            plt_.savefig(
                "results/images/" + dataset_name + "/" + sys.argv[2] + "/" + attack_name + "/" + str(i) + ".jpg",
                bbox_inches="tight",
            )
            plt_.close()

        euclidean_dist = math.sqrt(np.sum(noise * noise))

        total_dist.append(euclidean_dist)

    return total_dist, y_true, y_pred, y_pred_attack


def result(y_true, y_pred):

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred).flatten()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    # print("TN, FP, FN, TP :", cf_matrix)
    # print("accuracy :", accuracy)
    # print("accuracy :", precision)
    # print("accuracy :", recall)

    # Initialize a blank dataframe and keep adding
    df = pd.DataFrame(columns=["TN", "FP", "FN", "TP", "Accuracy", "Precision", "Recall"])
    df.loc[0] = cf_matrix.tolist() + [accuracy, precision, recall]
    df["Total_Actual_Neg"] = df["TN"] + df["FP"]
    df["Total_Actual_Pos"] = df["FN"] + df["TP"]
    df["Total_Pred_Neg"] = df["TN"] + df["FN"]
    df["Total_Pred_Pos"] = df["FP"] + df["TP"]
    df["TP_Rate"] = df["TP"] / df["Total_Actual_Pos"]  # Recall
    df["FP_Rate"] = df["FP"] / df["Total_Actual_Neg"]
    df["TN_Rate"] = df["TN"] / df["Total_Actual_Neg"]
    df["FN_Rate"] = df["FN"] / df["Total_Actual_Pos"]
    return df


def result_print(attack_name, dataset_name, y_true, y_pred, y_pred_attack):

    result_org = result(y_true, y_pred)
    print("RAIDS without attack:")
    print(result_org)

    result_advGAN = result(y_true, y_pred_attack)
    print(f"RAIDS under attack by {attack_name}:")
    print(result_advGAN)

    df_all = pd.concat([result_org, result_advGAN])
    df_all.to_csv("./results/" + dataset_name + "_" + attack_name + "_" + sys.argv[2] + ".csv")


if __name__ == "__main__":
    configfile = importlib.import_module(sys.argv[1])
    config = configfile.attacksconfig()
    dirname = os.path.dirname(os.path.abspath(__file__))
    dirparent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    attacks(config)
