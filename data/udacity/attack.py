import numpy as np
import pandas as pd
import csv
import random
from random import randint
import os, sys, importlib


def launch_attack_abrupt(csv_input):
    # Randomly select 30% images and for an image, add or subtract random value in [0.2, 0.9] to its corresponding angle
    # https://stackoverflow.com/questions/46450260/how-can-i-randomly-change-the-values-of-some-rows-in-a-pandas-dataframe
    dfupdate = csv_input[2:-2].sample(frac=0.3, random_state=5566)
    dfupdate["Anomaly"] = "1"
    dfupdate["random_rad"] = [random.uniform(0.1, 0.9) for row in dfupdate.index]
    dfupdate["add_subt"] = [random.choice([-1, 1]) for row in dfupdate.index]
    dfupdate["angle_new"] = csv_input["angle_new"] + (dfupdate["add_subt"] * dfupdate["random_rad"])
    dfupdate["speed_new"] = csv_input["speed_new"] + (dfupdate["add_subt"] * 20 * dfupdate["random_rad"])

    csv_input.update(dfupdate)
    return csv_input


def launch_attack_directed(csv_input):
    # Select the largest 15% and smallest 15% angles. Flip the sign of a selected angle if its absolute value is larger than 0.3.
    # https://stackoverflow.com/questions/46450260/how-can-i-randomly-change-the-values-of-some-rows-in-a-pandas-dataframe
    top15 = csv_input.nlargest(int(csv_input.shape[0] * 0.15), "angle_new")
    btm15 = csv_input.nsmallest(int(csv_input.shape[0] * 0.15), "angle_new")
    dfupdate = pd.concat([top15, btm15])
    dfupdate["Anomaly"] = "1"
    # If not, add or subtract a random value in [0.5, 1].
    dfupdate["random_rad"] = [random.uniform(0.5, 0.1) if value <= 0.3 else 0 for value in dfupdate["angle_new"]]
    dfupdate["add_subt"] = [random.choice([-1, 1]) if value <= 0.3 else 0 for value in dfupdate["angle_new"]]
    # Flip the sign of a selected angle if its absolute value is larger than 0.3.
    dfupdate["flip"] = [-1 if value > 0.3 else 1 for value in dfupdate["angle_new"]]

    dfupdate["angle_new"] = dfupdate["angle_new"] * dfupdate["flip"] + (dfupdate["add_subt"] * dfupdate["random_rad"])
    csv_input.update(dfupdate)
    return csv_input


if __name__ == "__main__":
    csv_input = pd.read_csv("Y_train.csv")

    # create new column Anomaly
    csv_input["Anomaly"] = "0"
    csv_input["random_rad"] = np.nan
    csv_input["add_subt"] = np.nan

    # copy original angle in rad to new angle for attacks later
    csv_input["angle_new"] = csv_input["angle_convert_org"]
    csv_input["speed_new"] = csv_input["speed"]

    # save a copy without any attacks
    csv_input.to_csv("Y_train_attack_none.csv", index=False)

    # csv_input = launch_attack_abrupt(csv_input)
    # csv_input.to_csv("Y_train_attack_abrupt.csv", index=False)

    # csv_input = launch_attack_directed(csv_input)
    # csv_input.to_csv("Y_train_attack_directed.csv", index=False)
