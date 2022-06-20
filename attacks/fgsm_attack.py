import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np


def fgsm_attack_fun(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def fgsm_attack(stage1, stage2, image, angle_speed, target, device, image_size, epsilon=0.01):
    correct = 0
    predicted_angle_speed = stage1(image)
    perturbed_image = image.clone()
    inv_target = 1 - target
    inv_target = inv_target.to(device)
    image.requires_grad = True
    # output = stage1(image)
    predicted_angle_speed = stage1(image)
    final_vars = torch.abs(torch.sub(angle_speed, predicted_angle_speed))
    output = stage2(final_vars)
    criterion = nn.BCEWithLogitsLoss()
    diff = 0
    # while abs(diff) < abs(target):
    for i in range(5):
        loss = criterion(output, target.unsqueeze(-1))
        stage1.zero_grad()
        stage2.zero_grad()
        loss.backward(retain_graph=True)
        image_grad = image.grad.data
        perturbed_image = fgsm_attack_fun(perturbed_image, epsilon, image_grad)
        adv_predicted_angle_speed = stage1(perturbed_image)
        final_vars = torch.abs(torch.sub(angle_speed, adv_predicted_angle_speed))
        adv_output = stage2(final_vars)
        # adv_pred = np.round(torch.sigmoid(adv_output.detach()))
    noise = torch.clamp(perturbed_image - image, 0, 1)

    return output, adv_output, perturbed_image, predicted_angle_speed, adv_predicted_angle_speed, noise


if __name__ == "__main__":
    pass
