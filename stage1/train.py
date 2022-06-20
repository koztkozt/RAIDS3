from sklearn.model_selection import train_test_split
import os, sys, importlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch
import math
import csv
from os import path
import numpy as np
import pandas as pd
import time
from torchvision import transforms, utils
from torchinfo import summary
from torch.utils.data import DataLoader
import cv2
from model import stage1, weight_init
from data import stage1_data

matplotlib.use("Agg")

if __name__ == "__main__":
    configfile = importlib.import_module(sys.argv[1])
    config = configfile.TrainConfig1()
    dirparent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    ch = config.num_channels
    num_epoch = config.num_epoch
    batch_size = config.batch_size
    data_path = config.data_path
    dataset_name = config.dataset_name
    img_height = config.img_height
    img_width = config.img_width
    num_channels = config.num_channels

    print("Loading training data...")
    X = np.load(dirparent + "/" + data_path + "X_train.npy")
    Y = pd.read_csv(dirparent + "/" + data_path + "/Y_train.csv")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=56)

    # print("Computing training set mean...")
    # X_train_mean = np.mean(X_train, axis=0, keepdims=True)
    # print("Saving training set mean...")
    # np.save(config.X_train_mean_path, X_train_mean)

    print("Creating model...")
    train_dataset = stage1_data(X_train, Y_train)
    train_generator = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)
    train_steps_per_epoch = int(len(train_dataset) / batch_size)
    test_dataset = stage1_data(X_test, Y_test)
    test_generator = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)
    test_steps_per_epoch = int(len(test_dataset) / batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = stage1()
    net = net.to(device)
    summary(net, input_size=(batch_size, num_channels, img_height, img_width))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    best_vloss = 1000000
    for epoch in range(num_epoch):
        net.train(True)
        running_loss = 0
        inter = 0.0
        for i_batch, sample in enumerate(train_generator):  # for each training i_batch
            if i_batch / len(train_generator) > inter:
                print(f"epoch: {epoch+1} completed: {(inter):.0%}")
                inter += 0.1

            batch_x, angle, speed = sample
            batch_x = batch_x.type(torch.FloatTensor)
            angle = angle.type(torch.FloatTensor)
            speed = speed.type(torch.FloatTensor)
            batch_x = batch_x.to(device)
            target = np.array([list(a) for a in zip(angle, speed)])
            target = torch.from_numpy(target)
            target = target.to(device)
            
            outputs = net(batch_x)
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        avg_loss = running_loss / train_steps_per_epoch
        print(f"Epoch {epoch+1} RMSE loss: {(avg_loss):4f}")

        print("########################### TESTING ##########################")
        net.train(False)
        running_vloss = 0.0
        yhat = []
        test_y = []
        for i, sample in enumerate(test_generator):
            batch_x, angle, speed = sample
            batch_x = batch_x.type(torch.FloatTensor)
            angle = angle.type(torch.FloatTensor)
            speed = speed.type(torch.FloatTensor)
            batch_x = batch_x.to(device)
            target = np.array([list(a) for a in zip(angle, speed)])
            target = torch.from_numpy(target)
            target = target.to(device)

            outputs = net(batch_x)
            loss = criterion(outputs, target)
            running_vloss += loss.item()

            # yhat.append(outputs.tolist())
            # test_y.append(angle.tolist())

        avg_vloss = running_vloss / test_steps_per_epoch
        print("LOSS train {} valid {}".format(avg_loss, avg_vloss))
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            torch.save(net.state_dict(), "stage1_" + dataset_name + ".pt")

            # yhat = np.concatenate(yhat).ravel()
            # test_y = np.concatenate(test_y).ravel()
            # print(yhat)
            # rmse = np.sqrt(np.mean((yhat - test_y) ** 2)) / (max(test_y) - min(test_y))
            # plt.figure(figsize=(32, 8))
            # plt.plot(test_y, "r.-", label="target")
            # plt.plot(yhat, "b^-", label="predict")
            # plt.legend(loc="best")
            # plt.title("RMSE: %.2f" % rmse)
            # plt.show()
            # model_fullname = "%s_%d.png" % (dataset_name, int(time.time()))
            # plt.savefig(model_fullname)
