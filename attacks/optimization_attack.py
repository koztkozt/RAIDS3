import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np


def optimized_attack(stage1, stage2, image, angle_speed, target, device):
    predicted_angle_speed = stage1(image)
    final_vars = torch.abs(torch.sub(angle_speed, predicted_angle_speed))
    y_pred = stage2(final_vars)
    y_adv = y_pred
    inv_target = 1 - target

    perturb = torch.zeros_like(image)
    perturb.requires_grad = True
    perturb = perturb.to(device)
    optimizer = optim.Adam(params=[perturb], lr=0.005)
    diff = 0
    criterion = nn.BCEWithLogitsLoss()

    # while abs(diff) < abs(target):
    for i in range(100):
        perturbed_image = image + perturb
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        adv_predicted_angle_speed = stage1(perturbed_image)
        final_vars = torch.abs(torch.sub(angle_speed, adv_predicted_angle_speed))
        y_adv = stage2(final_vars)

        adv_pred = np.round(torch.sigmoid(y_adv.detach()))
        if np.array_equal(inv_target, adv_pred):
            # print("success")
            break

        optimizer.zero_grad()
        loss_y = criterion(y_adv, inv_target.unsqueeze(-1))
        loss_n = torch.mean(torch.pow(perturb, 2))
        loss_adv = loss_y + loss_n
        loss_adv.backward(retain_graph=True)
        optimizer.step()

        # print(diff, target)

    return perturbed_image, perturb, y_pred, y_adv, predicted_angle_speed, adv_predicted_angle_speed
