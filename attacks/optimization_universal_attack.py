from sklearn.model_selection import train_test_split
import os, sys, importlib
import numpy as np
import pandas as pd

np.random.seed(0)

import torch

torch.manual_seed(0)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attacks.optimization_attack import optimized_attack
from stage1.model import stage1
from stage2.data import stage2_data
from stage2.model import stage2
from stage1.model import stage1


def proj_lp(v, xi, p):
    # Project on the lp ball centered at 0 and of radius xi
    v_ = v.detach().cpu().numpy()
    # SUPPORTS only p = 2 and p = Inf for now
    if p == 2:
        v_ = v_ * min(1, xi / np.linalg.norm(v_.flatten(1)))
        # v = v / np.linalg.norm(v.flatten(1)) * xi
    elif p == np.inf:
        v_ = np.sign(v_) * np.minimum(abs(v_), xi)
    else:
        raise ValueError("Values of p different from 2 and Inf are currently not supported...")

    return torch.from_numpy(v_)


def universal_attack(dataset, stage1, stage2, device, delta=0.3, max_iters=5, xi=10, p=np.inf, max_iter_lbfgs=30):
    v = 0
    fooling_rate = 0.0
    num_images = len(dataset)

    itr = 0
    while fooling_rate < 1 - delta and itr < max_iters:
        # np.random.shuffle(dataset)
        print("Starting pass number: ", itr)

        inter = 0.0
        for i, sample in enumerate(dataset):
            if i / len(dataset) > inter:
                inter += 0.01
                print(f"training completed: {(inter):.0%}")

            batch_x, angle, speed, target = sample
            batch_x = batch_x.type(torch.FloatTensor)
            angle = angle.type(torch.FloatTensor)
            speed = speed.type(torch.FloatTensor)
            target = target.type(torch.FloatTensor)
            inv_target = 1 - target
            batch_x = batch_x.to(device)
            angle = angle.to(device)
            speed = speed.to(device)
            inv_target = inv_target.to(device)
            angle_speed = np.array([list(a) for a in zip(angle, speed)])
            angle_speed = torch.from_numpy(angle_speed)

            perturbed_image = batch_x + v
            perturbed_image = torch.clamp(perturbed_image, 0, 1)
            perturbed_image = perturbed_image.to(device)
            temp = is_adversary(stage1, stage2, batch_x, perturbed_image, angle_speed, inv_target)
            if not temp[0]:
                # print('>> k = ', k, ', pass #', itr)
                _, d_noise, _, _, _, _ = optimized_attack(stage1, stage2, perturbed_image, angle_speed, target, device)
                v = v + d_noise
                v = proj_lp(v, xi, p)
                v = v.to(device)
        itr += 1

        count = 0
        for i, sample in enumerate(dataset):
            batch_x, angle, speed, target = sample
            batch_x = batch_x.type(torch.FloatTensor)
            angle = angle.type(torch.FloatTensor)
            speed = speed.type(torch.FloatTensor)
            target = target.type(torch.FloatTensor)
            inv_target = 1 - target
            batch_x = batch_x.to(device)
            angle = angle.to(device)
            speed = speed.to(device)
            inv_target = inv_target.to(device)
            angle_speed = np.array([list(a) for a in zip(angle, speed)])
            angle_speed = torch.from_numpy(angle_speed)

            perturbed_image = batch_x + v
            perturbed_image = torch.clamp(perturbed_image, 0, 1)
            perturbed_image = perturbed_image.to(device)
            if is_adversary(stage1, stage2, batch_x, perturbed_image, angle_speed, inv_target)[0]:
                count += 1
        fooling_rate = count / num_images

        print("Fooling rate: ", fooling_rate)
    # demension of v : (1, 3, image_size)
    return v


def is_adversary(stage1, stage2, x, x_adv, angle_speed, target):
    # print(target)
    adv_predicted_angle_speed = stage1(x_adv)
    final_vars = torch.abs(torch.sub(angle_speed, adv_predicted_angle_speed))
    y_pred_adv = stage2(final_vars)
    adv_pred = np.round(torch.sigmoid(y_pred_adv.detach()))
    if adv_pred == target:
        return [True]
    else:
        return [False]


def generate_noise(dataset, stage1, stage2, device):
    perturbation = universal_attack(dataset, stage1, stage2, device, target)
    perturbation = perturbation.detach().cpu().numpy()
    return perturbation


if __name__ == "__main__":
    configfile = importlib.import_module(sys.argv[1])
    config = configfile.optiUConfig()
    dirname = os.path.dirname(os.path.abspath(__file__))
    dirparent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    attack_name = "optU"
    dataset_name = config.dataset_name
    data_path = config.data_path
    target = config.target

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
    X = np.load(data_path + "/X_train.npy")
    Y = pd.read_csv(data_path + "/Y_train_attack_" + sys.argv[2] + ".csv")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=56)

    print("Creating model...")
    train_dataset = stage2_data(X_train, Y_train)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    num_sample = len(train_dataset)

    print("Start optiU training")
    perturbation = generate_noise(train_dataloader, stage1_model, stage2_model, device)
    np.save("./models/" + dataset_name + "_" + attack_name + "_noise_seed.npy", perturbation)
    print("Finish optiU training.")
