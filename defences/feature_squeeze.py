from sklearn.model_selection import train_test_split
import importlib
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
import math

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attacks.advGAN.models import Generator
from attacks.advGAN_attack import advGAN_Attack
from attacks.attacking import advGAN_ex, advGANU_ex, fgsm_ex, opt_ex, opt_uni_ex
from attacks.fgsm_attack import fgsm_attack
from attacks.optimization_attack import optimized_attack
from attacks.optimization_universal_attack import generate_noise
from attacks.attack_test import (
    fgsm_attack_test,
    optimized_attack_test,
    optimized_uni_test,
    advGAN_test,
    advGAN_uni_test,
)

# from scipy import ndimage

# from scipy.misc import imread, imresize, imsave

from stage1.model import stage1
from stage2.model import stage2
from stage2.data import stage2_data


def reduce_bit(image, bit_size):
    image_int = np.rint(image * (math.pow(2, bit_size) - 1))
    image_float = image_int / (math.pow(2, bit_size) - 1)
    return image_float


def median_filter_np(x, width, height=-1):
    """
    Median smoothing by Scipy.
    :param x: a tensor of image(s)
    :param width: the width of the sliding window (number of pixels)
    :param height: the height of the window. The same as width by default.
    :return: a modified tensor with the same shape as x.
    """
    if height == -1:
        height = width
    return ndimage.median_filter(x, size=2, mode="reflect")


def attack_detection(
    stage1,
    stage2,
    test_data_loader,
    config,
    dirparent,
    attack_name,
    device,
    threshold=0.05,
):
    dataset_name = config.dataset_name
    batch_size = config.batch_size
    target = config.target
    image_nc = config.num_channels
    image_size = (config.img_height, config.img_width)

    orig_image_detected = 0
    pertubed_image_detected = 0
    inter = 0.0
    for i, sample in enumerate(test_data_loader):
        if i / len(test_data_loader) > inter:
            print(f"attacks completed: {(inter):.0%}")
            inter += 0.1

        image = sample[0].type(torch.FloatTensor)
        image = image.to(device)
        squeeze_image = reduce_bit(sample[0], 2)
        squeeze_image = median_filter_np(np.transpose(sample[0].squeeze(0).numpy(), (1, 2, 0)), 2)
        squeeze_image = torch.from_numpy(np.transpose(squeeze_image, (-1, 0, 1))).unsqueeze(0)
        squeeze_image = squeeze_image.type(torch.FloatTensor)
        squeeze_image = squeeze_image.to(device)
        pred_angle_squeezed = stage1(squeeze_image)

        plot_fig = False
        image2, angle, target = sample

        if attack_name == "fgsm":
            output, adv_output, pred_angle, pred_angle_attack, plt_, norm_noise, perturbed_image = fgsm_attack_test(
                stage1, stage2, image2, angle, target, device, image_size, plot_fig
            )

        elif attack_name == "opt":
            (
                output,
                adv_output,
                pred_angle,
                pred_angle_attack,
                plt_,
                norm_noise,
                perturbed_image,
            ) = optimized_attack_test(stage1, stage2, image2, angle, target, device, image_size, plot_fig)

        elif attack_name == "optU":
            noise = np.load(dirparent + "/attacks/models/" + dataset_name + "_" + attack_name + "_noise_seed.npy")
            noise = np.tile(noise, (batch_size, 1, 1, 1))
            noise = torch.from_numpy(noise).type(torch.FloatTensor).to(device)
            output, adv_output, pred_angle, pred_angle_attack, plt_, perturbed_image = optimized_uni_test(
                stage1, stage2, image2, angle, target, device, noise, image_size, plot_fig
            )

        elif attack_name == "advGAN":
            advGAN_generator = Generator(image_nc, image_nc, attack_name).to(device)
            advGAN_generator.load_state_dict(
                torch.load(dirparent + "/attacks/models/" + dataset_name + "_" + attack_name + "_netG_epoch_32.pth")
            )
            advGAN_generator.eval()
            output, adv_output, pred_angle, pred_angle_attack, plt_, noise, perturbed_image = advGAN_test(
                stage1, stage2, image, angle, target, advGAN_generator, device, image_size, plot_fig
            )

        elif attack_name == "advGANU":
            advGANU_generator = Generator(image_nc, image_nc, attack_name).to(device)
            advGANU_generator.load_state_dict(
                torch.load(dirparent + "/attacks/models/" + dataset_name + "_" + attack_name + "_netG_epoch_32.pth")
            )
            advGANU_generator.eval()
            noise_seed = np.load(dirparent + "/attacks/models/" + dataset_name + "_" + attack_name + "_noise_seed.npy")
            noise_seed = np.tile(noise_seed, (batch_size, 1, 1, 1))
            noise = advGANU_generator(torch.from_numpy(noise_seed).type(torch.FloatTensor).to(device))
            output, adv_output, pred_angle, pred_angle_attack, plt_, perturbed_image = advGAN_uni_test(
                stage1, stage2, image, angle, target, device, noise, image_size, plot_fig
            )

        squeeze_perturbed_image = reduce_bit(perturbed_image, 2)
        squeeze_perturbed_image = median_filter_np(perturbed_image, 2)
        squeeze_perturbed_image = torch.from_numpy(squeeze_perturbed_image)
        squeeze_perturbed_image = squeeze_perturbed_image.to(device)
        pred_angle_attack_squeezed = stage1(squeeze_perturbed_image)

        output, output_squeeze = stage2_pred(device, stage2, pred_angle, pred_angle_squeezed, sample)
        score = torch.sum(torch.abs(2 * (output - output_squeeze)), 1)
        orig_image_detected += sum(i > threshold for i in score)
        # print(pred_angle_attack[0], pred_angle_attack_squeezed[0])
        output_adv, output_squeeze_adv = stage2_pred(
            device, stage2, pred_angle_attack, pred_angle_attack_squeezed, sample
        )
        score_adv = torch.sum(torch.abs(2 * (output_adv - output_squeeze_adv)), 1)
        pertubed_image_detected += sum(i > threshold for i in score_adv)
        # print(score)
        # print(score_adv)
        # print(orig_image_detected.item(), pertubed_image_detected.item())

        if i % 512 == 0:
            Path("results/images/" + dataset_name + "/" + sys.argv[2] + "/" + attack_name + "/").mkdir(
                parents=True, exist_ok=True
            )
            plt_ = generate_image(image, squeeze_image, perturbed_image, squeeze_perturbed_image)
            plt_.savefig(
                "results/images/" + dataset_name + "/" + sys.argv[2] + "/" + attack_name + "/" + str(i) + ".jpg",
                bbox_inches="tight",
            )
            plt_.close()
        # break
    return orig_image_detected.item(), pertubed_image_detected.item()


def defences(config):
    dataset_name = config.dataset_name
    dirname = os.path.dirname(os.path.abspath(__file__))
    dirparent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = config.data_path
    batch_size = config.batch_size
    # threshold = config.threshold

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

    # root_dir = "../udacity-data"
    # attacks = ("FGSM", "Optimization", "Optimization Universal", "AdvGAN", "AdvGAN Universal")

    print("Loading training data...")
    X = np.load(data_path + "/X_train.npy")
    Y = pd.read_csv(data_path + "/Y_train_attack_" + sys.argv[2] + ".csv")
    # full_dataset = stage2_data(X, Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.01, random_state=56)
    # train_dataset = stage2_data(X_train, Y_train)
    test_dataset = stage2_data(X_test, Y_test)

    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    num_sample = len(test_dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # attack_names = ["fgsm", "opt", "optU", "advGAN", "advGANU"]
    attack_names = ["fgsm"]

    # for threshold in [0.01, 0.5, 1.0, 1.5, 1.99]:
    for threshold in [0.5]:
        df = pd.DataFrame(columns=["Attack", "Total", "orig_image_detected", "pertubed_image_detected"])
        print("threshold =", threshold)
        for i in range(len(attack_names)):
            print("[+] Attacking using", attack_names[i])
            orig_image_detected, pertubed_image_detected = attack_detection(
                stage1_model,
                stage2_model,
                test_data_loader,
                config,
                dirparent,
                attack_names[i],
                device,
                threshold,
            )
            df.loc[i] = [attack_names[i], num_sample, orig_image_detected, pertubed_image_detected]
            print(df)
        df["False Positive"] = df["orig_image_detected"] / df["Total"]
        df["Detection Rate"] = df["pertubed_image_detected"] / df["Total"]
        print("Detection results")
        print(df)
        df.to_csv("./results/" + dataset_name + "_" + sys.argv[2] + "_" + str(threshold) + ".csv")


def generate_image(image, squeeze_image, perturbed_image, squeeze_perturbed_image):
    ax1 = plt.subplot(2, 2, 1)
    ax1.title.set_text("original image")
    plt.imshow(image.detach().cpu().numpy()[0, 0, :, :], cmap="gray")
    ax2 = plt.subplot(2, 2, 2)
    ax2.title.set_text("original image squeezed")
    plt.imshow(squeeze_image[0, 0, :, :], cmap="gray")
    ax3 = plt.subplot(2, 2, 3)
    ax3.title.set_text("perturbed image")
    plt.imshow(perturbed_image[0, 0, :, :], cmap="gray")
    ax4 = plt.subplot(2, 2, 4)
    ax4.title.set_text("perturbed image squeezed")
    plt.imshow(squeeze_perturbed_image[0, 0, :, :], cmap="gray")
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)

    return plt


# def plot_figure():
#     fig = plt.figure(figsize=(12, 4))
#     df1 = pd.read_excel("feature_squeezing_epoch.xlsx", sheetname="Sheet1")
#     df2 = pd.read_excel("feature_squeezing_nvidia.xlsx", sheetname="Sheet1")
#     df3 = pd.read_excel("feature_squeezing_vgg16.xlsx", sheetname="Sheet1")
#     ax1 = fig.add_subplot(1, 3, 1)
#     df1.plot(
#         ax=ax1,
#         x="Threshold",
#         y=["IT-FGSM", "Opt", "Opt_uni", "AdvGAN", "AdvGAN_uni", "Original(False)"],
#         title="Detection rate on Epoch",
#     )
#     plt.xticks([0.01, 0.05, 0.1, 0.15])
#     plt.ylabel("Detection rate")
#     ax2 = fig.add_subplot(1, 3, 2)
#     df2.plot(
#         ax=ax2,
#         x="Threshold",
#         y=["IT-FGSM", "Opt", "Opt_uni", "AdvGAN", "AdvGAN_uni", "Original(False)"],
#         title="Detection rate on Nvidia",
#     )
#     plt.xticks([0.01, 0.05, 0.1, 0.15])
#     # plt.ylabel("Detection rate")
#     ax3 = fig.add_subplot(1, 3, 3)
#     df3.plot(
#         ax=ax3,
#         x="Threshold",
#         y=["IT-FGSM", "Opt", "Opt_uni", "AdvGAN", "AdvGAN_uni", "Original(False)"],
#         title="Detection rate on VGG16",
#     )
#     plt.xticks([0.01, 0.05, 0.1, 0.15])
#     # plt.ylabel("Detection rate")
#     ax1.legend_.remove()
#     ax2.legend_.remove()
#     ax3.legend_.remove()
#     # plt.ylabel("Detection rate")
#     # ax1.legend(loc=2, bbox_to_anchor=(-1.0,1.0),borderaxespad = 0.)
#     # plt.xticks([0.01, 0.05, 0.1, 0.15])
#     # plt.ylabel("Detection rate")
#     fig.tight_layout()
#     # fig.show()
#     plt.show()


def stage2_pred(device, stage2, pred_angle, pred_angle_squeeze, sample):
    batch_x, angle, target = sample
    batch_x = batch_x.type(torch.FloatTensor)
    angle = angle.type(torch.FloatTensor)
    target = target.type(torch.FloatTensor)
    batch_x = batch_x.to(device)
    angle = angle.to(device)
    target = target.to(device)

    output = stage2(torch.abs(torch.sub(angle.unsqueeze(-1), pred_angle)))
    output_squeeze = stage2(torch.abs(torch.sub(angle.unsqueeze(-1), pred_angle_squeeze)))

    return torch.sigmoid(output.detach()), torch.sigmoid(output_squeeze.detach())


if __name__ == "__main__":
    configfile = importlib.import_module(sys.argv[1])
    config = configfile.defencesconfig()
    defences(config)
