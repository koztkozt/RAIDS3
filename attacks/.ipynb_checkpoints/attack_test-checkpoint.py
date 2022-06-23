import importlib
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# from data import UdacityDataset, Rescale, Preprocess, ToTensor
# from model import BaseCNN
# from viewer import draw
from torchvision import datasets, transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attacks.fgsm_attack import fgsm_attack
from attacks.advGAN.models import Generator
from attacks.optimization_attack import optimized_attack

# from scipy.misc import imread, imresize
from imageio import imread
from skimage.transform import resize

# def exp1_fig():
#     model = BaseCNN()
#     model_name = "baseline"
#     model.load_state_dict(torch.load("baseline.pt"))
#     device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
#     model = model.to(device)
#     model.eval()
#     target = 0.3
#     image = imread("F:\\udacity-data\\testing\\center\\1479425441182877835.jpg")[200:, :]
#     image = imresize(image, (128, 128))
#     image = image / 255.0
#     image = torch.from_numpy(image.transpose((2, 0, 1))).unsqueeze(0)
#     image = image.type(torch.FloatTensor)
#     image = image.to(device)
#     output = model(image)
#     print(output)

#     advGAN_generator = Generator(3, 3, model_name).to(device)
#     advGAN_uni_generator = Generator(3, 3, model_name).to(device)

#     # fgsm

#     _, perturbed_image_fgsm, _, adv_output_fgsm, noise_fgsm = fgsm_attack(model, image, target, device)
#     print("fgsm", adv_output_fgsm)
#     perturbed_image_fgsm = perturbed_image_fgsm.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
#     noise_fgsm = noise_fgsm.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
#     perturbed_image_fgsm = draw(perturbed_image_fgsm, adv_output_fgsm.item(), output.item())
#     perturbed_image_fgsm = imresize(perturbed_image_fgsm, (128, 128))
#     # opt
#     perturbed_image_opt, noise_opt, _, adv_output_opt = optimized_attack(model, target, image, device)
#     perturbed_image_opt = perturbed_image_opt.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
#     print("opt", adv_output_opt)

#     noise_opt = noise_opt.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
#     perturbed_image_opt = draw(perturbed_image_opt, adv_output_opt.item(), output.item())
#     perturbed_image_opt = imresize(perturbed_image_opt, (128, 128))
#     # optu
#     noise_optu = np.load(model_name + "_universal_attack_noise.npy")
#     noise_optu = torch.from_numpy(noise_optu).type(torch.FloatTensor).to(device)
#     perturbed_image_optu = image + noise_optu
#     perturbed_image_optu = torch.clamp(perturbed_image_optu, 0, 1)
#     adv_output_optu = model(perturbed_image_optu)
#     print("universal", adv_output_optu)
#     perturbed_image_optu = perturbed_image_optu.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
#     noise_optu = noise_optu.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
#     perturbed_image_optu = draw(perturbed_image_optu, adv_output_optu.item(), output.item())
#     perturbed_image_optu = imresize(perturbed_image_optu, (128, 128))
#     # advGAN
#     advGAN_generator.load_state_dict(torch.load("./models/" + model_name + "_netG_epoch_60.pth"))
#     noise_advGAN = advGAN_generator(image)
#     perturbed_image_advGAN = image + torch.clamp(noise_advGAN, -0.3, 0.3)
#     perturbed_image_advGAN = torch.clamp(perturbed_image_advGAN, 0, 1)
#     adv_output_advGAN = model(perturbed_image_advGAN)
#     print("advGAN", adv_output_advGAN)
#     perturbed_image_advGAN = perturbed_image_advGAN.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
#     noise_advGAN = noise_advGAN.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
#     perturbed_image_advGAN = draw(perturbed_image_advGAN, adv_output_advGAN.item(), output.item())
#     perturbed_image_advGAN = imresize(perturbed_image_advGAN, (128, 128))
#     # advGAN_U
#     advGAN_uni_generator.load_state_dict(torch.load("./models/" + model_name + "_universal_netG_epoch_60.pth"))
#     noise_seed = np.load(model_name + "_noise_seed.npy")
#     noise_advGAN_U = advGAN_uni_generator(torch.from_numpy(noise_seed).type(torch.FloatTensor).to(device))
#     perturbed_image_advGAN_U = image + torch.clamp(noise_advGAN_U, -0.3, 0.3)
#     perturbed_image_advGAN_U = torch.clamp(perturbed_image_advGAN_U, 0, 1)
#     adv_output_advGAN_U = model(perturbed_image_advGAN_U)
#     print("advGAN_uni", adv_output_advGAN_U)
#     perturbed_image_advGAN_U = perturbed_image_advGAN_U.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
#     noise_advGAN_U = noise_advGAN_U.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
#     perturbed_image_advGAN_U = draw(perturbed_image_advGAN_U, adv_output_advGAN_U.item(), output.item())
#     perturbed_image_advGAN_U = imresize(perturbed_image_advGAN_U, (128, 128))

#     plt.subplot(2, 5, 1)
#     plt.imshow(perturbed_image_fgsm)
#     # plt.text(0.3, 0.3, 'y: %.4f' % output.item())
#     plt.xticks([])
#     plt.yticks([])
#     plt.subplot(2, 5, 2)
#     plt.imshow(perturbed_image_opt)
#     plt.xticks([])
#     plt.yticks([])
#     plt.subplot(2, 5, 3)
#     plt.imshow(perturbed_image_optu)
#     plt.xticks([])
#     plt.yticks([])
#     plt.subplot(2, 5, 4)
#     plt.imshow(perturbed_image_advGAN)
#     plt.xticks([])
#     plt.yticks([])
#     plt.subplot(2, 5, 5)
#     plt.imshow(perturbed_image_advGAN_U)
#     plt.xticks([])
#     plt.yticks([])
#     plt.subplot(2, 5, 6)
#     plt.imshow(np.clip(noise_fgsm * 5, 0, 1))
#     plt.xticks([])
#     plt.yticks([])
#     plt.subplot(2, 5, 7)
#     plt.imshow(np.clip(noise_opt * 5, 0, 1))
#     plt.xticks([])
#     plt.yticks([])
#     plt.subplot(2, 5, 8)
#     plt.imshow(np.clip(noise_optu * 5, 0, 1))
#     plt.xticks([])
#     plt.yticks([])
#     plt.subplot(2, 5, 9)
#     plt.imshow(np.clip(noise_advGAN * 5, 0, 1))
#     plt.xticks([])
#     plt.yticks([])
#     plt.subplot(2, 5, 10)
#     plt.imshow(np.clip(noise_advGAN_U * 5, 0, 1))
#     plt.xticks([])
#     plt.yticks([])

#     plt.tight_layout(pad=0.5, w_pad=0, h_pad=0)
#     plt.show()


def generate_image(X, X_adv, noise, pred_steer, pred_speed, attack_steer, attack_speed, image_size):
    plt.gray()
    plt.figure(figsize=(6, 6))
    ax1 = plt.subplot(1, 3, 1)
    ax1.title.set_text("org steer: %.4f \n org speed: %.4f " % (pred_steer,pred_speed))
    # X = draw(X * 255, np.array([y_pred]))
    X = resize(X, image_size)
    plt.imshow(X)
    ax2 = plt.subplot(1, 3, 2)
    ax2.title.set_text("adv steer: %.4f \n adv speed: %.4f " % (attack_steer,attack_speed))
    # X_adv = draw(X_adv * 255, np.array([y_pred]), np.array([y_adv]))
    X_adv = resize(X_adv, image_size)
    plt.imshow(X_adv)
    ax3 = plt.subplot(1, 3, 3)
    ax3.title.set_text("5 * noise")
    plt.imshow(np.clip(noise * 5, 0, 1))
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)
    return plt


def fgsm_attack_test(stage1, stage2, image, angle, target, device, image_size, plot_fig, epsilon=0.01):
    # image = image.unsqueeze(0)
    image = image.type(torch.FloatTensor)
    angle = angle.type(torch.FloatTensor)
    target = target.type(torch.FloatTensor)
    image = image.to(device)
    angle = angle.to(device)
    target = target.to(device)
    output, adv_output, perturbed_image, predicted_angle, adv_predicted_angle, noise = fgsm_attack(
        stage1, stage2, image, angle, target, device, image_size
    )
    if plot_fig:
        plt = generate_image(
            image.detach().cpu().numpy().transpose(0, 2, 3, 1)[0, :, :, 0],
            perturbed_image.detach().cpu().numpy().transpose(0, 2, 3, 1)[0, :, :, 0],
            noise.detach().cpu().numpy().transpose(0, 2, 3, 1)[0, :, :, 0],
            predicted_angle.detach().cpu().numpy()[0][0],
            adv_predicted_angle.detach().cpu().numpy()[0][0],
            image_size,
        )
    else:
        plt = 0
    return (
        output,
        adv_output,
        predicted_angle,
        adv_predicted_angle,
        plt,
        noise.detach().cpu().numpy(),
        # np.sum(noise.detach().cpu().numpy()),
        perturbed_image.detach().cpu().numpy(),
    )


def optimized_attack_test(stage1, stage2, image, angle, target, device, image_size, plot_fig):
    image = image.type(torch.FloatTensor)
    angle = angle.type(torch.FloatTensor)
    target = target.type(torch.FloatTensor)
    image = image.to(device)
    angle = angle.to(device)
    target = target.to(device)
    perturbed_image, noise, output, adv_output, predicted_angle, adv_predicted_angle = optimized_attack(
        stage1, stage2, image, angle, target, device
    )
    # if not np.all(np.ravel(image == perturbed_image)):
    if plot_fig:
        plt = generate_image(
            image.detach().cpu().numpy().transpose(0, 2, 3, 1)[0, :, :, 0],
            perturbed_image.detach().cpu().numpy().transpose(0, 2, 3, 1)[0, :, :, 0],
            noise.detach().cpu().numpy().transpose(0, 2, 3, 1)[0, :, :, 0],
            predicted_angle.detach().cpu().numpy()[0][0],
            adv_predicted_angle.detach().cpu().numpy()[0][0],
            image_size,
        )
    else:
        plt = 0
    # plt.show()
    return (
        output,
        adv_output,
        predicted_angle,
        adv_predicted_angle,
        plt,
        noise.detach().cpu().numpy(),
        perturbed_image.detach().cpu().numpy(),
    )


def optimized_uni_test(stage1, stage2, image, angle, target, device, noise, image_size, plot_fig):
    image = image.type(torch.FloatTensor)
    angle = angle.type(torch.FloatTensor)
    target = target.type(torch.FloatTensor)
    image = image.to(device)
    angle = angle.to(device)
    target = target.to(device)

    predicted_angle = stage1(image)
    final_vars = torch.abs(torch.sub(angle, predicted_angle))
    output = stage2(final_vars)

    perturbed_image = image + noise
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    adv_predicted_angle = stage1(perturbed_image)
    final_vars = torch.abs(torch.sub(angle, adv_predicted_angle))
    adv_output = stage2(final_vars)

    if plot_fig:
        plt = generate_image(
            image.detach().cpu().numpy().transpose(0, 2, 3, 1)[0, :, :, 0],
            perturbed_image.detach().cpu().numpy().transpose(0, 2, 3, 1)[0, :, :, 0],
            noise.detach().cpu().numpy().transpose(0, 2, 3, 1)[0, :, :, 0],
            predicted_angle.detach().cpu().numpy()[0][0],
            adv_predicted_angle.detach().cpu().numpy()[0][0],
            image_size,
        )
    else:
        plt = 0
    return (
        output,
        adv_output,
        predicted_angle,
        adv_predicted_angle,
        plt,
        perturbed_image.detach().cpu().numpy(),
    )


def advGAN_test(stage1, stage2, image, angle_speed, target, advGAN_generator, device, image_size, plot_fig):
    predicted_angle_speed = stage1(image)
    final_vars = torch.abs(torch.sub(angle_speed, predicted_angle_speed))
    output = stage2(final_vars)

    noise = advGAN_generator(image)
    perturbed_image = image + torch.clamp(noise, -0.3, 0.3)
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    adv_predicted_angle_speed = stage1(perturbed_image)
    final_vars = torch.abs(torch.sub(angle_speed, adv_predicted_angle_speed))
    adv_output = stage2(final_vars)
    if plot_fig:
        plt = generate_image(
            image.detach().cpu().numpy().transpose(0, 2, 3, 1)[0, :, :, 0],
            perturbed_image.detach().cpu().numpy().transpose(0, 2, 3, 1)[0, :, :, 0],
            noise.detach().cpu().numpy().transpose(0, 2, 3, 1)[0, :, :, 0],
            predicted_angle_speed.detach().cpu().numpy()[0][0],
            predicted_angle_speed.detach().cpu().numpy()[0][1],
            adv_predicted_angle_speed.detach().cpu().numpy()[0][0],
            adv_predicted_angle_speed.detach().cpu().numpy()[0][1],
            image_size,
        )
    else:
        plt = 0
    return (
        output,
        adv_output,
        predicted_angle_speed,
        adv_predicted_angle_speed,
        plt,
        noise.detach().cpu().numpy(),
        perturbed_image.detach().cpu().numpy(),
    )


def advGAN_uni_test(stage1, stage2, image, angle, target, device, noise, image_size, plot_fig):
    image = image.type(torch.FloatTensor)
    angle = angle.type(torch.FloatTensor)
    target = target.type(torch.FloatTensor)
    image = image.to(device)
    angle = angle.to(device)
    target = target.to(device)

    predicted_angle = stage1(image)
    final_vars = torch.abs(torch.sub(angle, predicted_angle))
    output = stage2(final_vars)

    perturbed_image = image + torch.clamp(noise, -0.3, 0.3)
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    adv_predicted_angle = stage1(perturbed_image)
    final_vars = torch.abs(torch.sub(angle, adv_predicted_angle))
    adv_output = stage2(final_vars)

    if plot_fig:
        plt = generate_image(
            image.detach().cpu().numpy().transpose(0, 2, 3, 1)[0, :, :, 0],
            perturbed_image.detach().cpu().numpy().transpose(0, 2, 3, 1)[0, :, :, 0],
            noise.detach().cpu().numpy().transpose(0, 2, 3, 1)[0, :, :, 0],
            predicted_angle.detach().cpu().numpy()[0][0],
            adv_predicted_angle.detach().cpu().numpy()[0][0],
            image_size,
        )
    else:
        plt = 0
    return (
        output,
        adv_output,
        predicted_angle,
        adv_predicted_angle,
        plt,
        perturbed_image.detach().cpu().numpy(),
    )


if __name__ == "__main__":
    # target_model = 'cnn.pt'
    # target = 0.3
    # root_dir = '/media/dylan/Program/cg23-dataset'
    # test_composed = transforms.Compose([Rescale((128, 128)), Preprocess('baseline'), ToTensor()])
    # full_dataset = UdacityDataset(root_dir, ['testing'], test_composed, type_='test')
    # train_size = int(0.8*len(full_dataset))
    # test_size =len(full_dataset) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    # device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    # # dataloader = torch.utils.data.DataLoader(dataset,1,True)
    # model = BaseCNN()
    # model.to(device)
    # model.load_state_dict(torch.load(target_model))
    # model.eval()
    # advGAN_uni_test(model, test_dataset, target)
    # advGAN_test(model, test_dataset, target)
    # optimized_attack_test(model, train_dataset, target, device)
    # fgsm_attack_test(model, train_dataset, target, device)
    exp1_fig()
