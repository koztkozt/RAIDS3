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

import math

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stage1.model import stage1
from stage2.model import stage2
from stage2.data import stage2_data


def defences(config):
    dataset_name = config.dataset_name
    dirname = os.path.dirname(os.path.abspath(__file__))
    dirparent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ch = config.num_channels
    num_epoch = config.num_epoch
    batch_size = config.batch_size
    data_path = config.data_path
    dataset_name = config.dataset_name
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

    print("Loading training data...")
    X = np.load(data_path + "/X_train.npy")
    Y = pd.read_csv(data_path + "/Y_train_attack_" + sys.argv[2] + ".csv")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=56)
    train_dataset = stage2_data(X_train, Y_train)
    train_generator = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)
    train_steps_per_epoch = int(len(train_dataset) / batch_size)
    test_dataset = stage2_data(X_test, Y_test)
    test_generator = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)
    test_steps_per_epoch = int(len(test_dataset) / batch_size)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    num_sample = len(train_dataset)

    T = [5, 10, 15, 20, 25]
    # T = [20]
    for t in T:
        net = stage2()
        net = net.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.01)

        # best_vloss = 1000000

        print("Length of dataloader :", len(train_data_loader))
        for epoch in range(num_epoch):
            net.train(True)
            running_loss = 0
            inter = 0.0
            for i_batch, sample in enumerate(train_data_loader):  # for each training i_batch
                if i_batch / len(train_data_loader) > inter:
                    print(f"epoch: {epoch} completed: {(inter):.0%}")
                    inter += 0.10

                batch_x, angle, target = sample
                batch_x = batch_x.type(torch.FloatTensor)
                angle = angle.type(torch.FloatTensor)
                # target = target.type(torch.FloatTensor)
                batch_x = batch_x.to(device)
                angle = angle.to(device)
                # target = target.to(device)

                predicted_angle = stage1_model(batch_x)

                final_vars = torch.abs(torch.sub(angle.unsqueeze(-1), predicted_angle))
                # softlabel
                softlabel = stage2_model(final_vars)
                # print(softlabel)
                target = torch.sigmoid(softlabel / t)
                # print(target)
                output = net(final_vars)
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # pred = np.round(torch.sigmoid(output.detach()))
                # print(pred.reshape(-1))
                # print(target)
                # break
            avg_loss = running_loss / train_steps_per_epoch
            print(f"Epoch {epoch} loss: {(avg_loss):4f}")
        torch.save(net.state_dict(), "adv_training_models/" + dataset_name + "_distillation_" + str(t) + ".pt")


if __name__ == "__main__":
    configfile = importlib.import_module(sys.argv[1])
    config = configfile.distillconfig()
    defences(config)
