import torch.nn as nn
import torch
import numpy as np
from . import models
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import os

# np.random.seed(0)
# torch.manual_seed(0)
# import torch.nn as nn


# import matplotlib.pyplot as plt
# from data import UdacityDataset, Rescale, Preprocess, ToTensor
# from model import BaseCNN
# from torchvision import datasets, transforms
models_path = "./models/"


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class AdvGAN_Attack:
    def __init__(
        self,
        device,
        stage1,
        stage2,
        model_name,
        #  model_num_labels,
        # target,
        image_nc,
        box_min,
        box_max,
        batch_size,
    ):
        output_nc = image_nc
        self.device = device
        # self.target = target
        # self.model_num_labels = model_num_labels
        self.stage1 = stage1
        self.stage2 = stage2
        self.model_name = model_name
        self.input_nc = image_nc
        self.output_nc = output_nc
        self.box_min = box_min
        self.box_max = box_max
        self.batch_size = batch_size

        self.gen_input_nc = image_nc
        self.netG = models.Generator(self.gen_input_nc, image_nc, self.model_name).to(self.device)
        self.netDisc = models.Discriminator(image_nc).to(self.device)

        # initialize all weights
        self.netG.apply(weights_init)
        self.netDisc.apply(weights_init)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=0.001)
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(), lr=0.001)

        if not os.path.exists(models_path):
            os.makedirs(models_path)

    def train_batch(self, x, angle_speed, target):
        # optimize D
        for i in range(1):
            perturbation = self.netG(x)

            # add a clipping trick
            adv_images = torch.clamp(perturbation, -0.3, 0.3) + x
            adv_images = torch.clamp(adv_images, self.box_min, self.box_max)

            self.optimizer_D.zero_grad()
            pred_real = self.netDisc(x)
            loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device))
            loss_D_real.backward()

            pred_fake = self.netDisc(adv_images.detach())
            loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device))
            loss_D_fake.backward()
            loss_D_GAN = loss_D_fake + loss_D_real
            self.optimizer_D.step()

        # optimize G
        for i in range(1):
            self.optimizer_G.zero_grad()

            # cal G's loss in GAN
            pred_fake = self.netDisc(adv_images)
            loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=self.device))
            loss_G_fake.backward(retain_graph=True)

            # calculate perturbation norm
            C = 0.1
            loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))

            predicted_angle_speed = self.stage1(adv_images)
            final_vars = torch.abs(torch.sub(angle_speed, predicted_angle_speed))
            output = self.stage2(final_vars)
            criterion = nn.BCEWithLogitsLoss()
            loss_adv = criterion(output, target.unsqueeze(-1))

            adv_lambda = 500
            pert_lambda = 1
            loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb
            loss_G.backward()
            self.optimizer_G.step()

        # print(loss_D_GAN.item(), loss_G_fake.item(), loss_perturb.item(), loss_adv.item())
        return loss_D_GAN.item(), loss_G_fake.item(), loss_perturb.item(), loss_adv.item() * loss_adv.item()

    def train(self, train_dataset, epochs):
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(1, epochs + 1):
            if epoch == 50:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=0.0001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(), lr=0.0001)
            if epoch == 80:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=0.00001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(), lr=0.00001)
            loss_D_sum = 0
            loss_G_fake_sum = 0
            loss_perturb_sum = 0
            loss_adv_sum = 0

            inter = 0.1
            for _, sample in enumerate(train_dataloader, start=0):
                if _ / len(train_dataloader) > inter:
                    print(f"epoch: {epoch} completed: {(inter):.0%}")
                    inter += 0.1

                batch_x, angle, speed, target = sample
                batch_x = batch_x.type(torch.FloatTensor)
                angle = angle.type(torch.FloatTensor)
                speed = speed.type(torch.FloatTensor)
                target = target.type(torch.FloatTensor)
                inv_target = 1 - target
                batch_x = batch_x.to(self.device)
                # angle = angle.to(self.device)
                # speed = speed.to(self.device)
                inv_target = inv_target.to(self.device)
                angle_speed = np.array([list(a) for a in zip(angle, speed)])
                angle_speed = torch.from_numpy(angle_speed)
                angle_speed = angle_speed.to(self.device)

                loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch = self.train_batch(
                    batch_x, angle_speed, inv_target
                )
                loss_D_sum += loss_D_batch
                loss_G_fake_sum += loss_G_fake_batch
                loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch

            # print statistics
            num_batch = len(train_dataloader)
            print(
                "epoch %d:\nloss_D: %.4f, loss_G_fake: %.4f,\
            \nloss_perturb: %.4f, loss_adv: %.4f, \n"
                % (
                    epoch,
                    loss_D_sum / num_batch,
                    loss_G_fake_sum / num_batch,
                    loss_perturb_sum / num_batch,
                    loss_adv_sum / num_batch,
                )
            )

            # save generator
            if epoch % 60 == 0:
                netG_file_name = models_path + self.model_name + "_netG_epoch_" + str(epoch) + ".pth"
                torch.save(self.netG.state_dict(), netG_file_name)


if __name__ == "__main__":
    pass
