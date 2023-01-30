import torch
import torch.nn as nn
import numpy as np
import os
from torchvision import transforms
from datasets.mnist_dataset import MNISTDataset
from torch.utils.data import DataLoader
from models.model import *
from models.loss import *
from torch.autograd import Variable
import matplotlib.pyplot as plt
from executors.logger import NeptuneLogger
import torchvision


class Trainer:
    def __init__(self, train_config, logger_config):
        self.train_config = train_config
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.__init_data()
        self.__init_model()
        self.log = logger_config is not None
        if self.log:
            self.logger = NeptuneLogger(logger_config)
        self.D_global_step = 0
        self.G_global_step = 0

    def __init_data(self):
        self.transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ]
        )
        self.dataset = MNISTDataset(data_path=self.train_config.datapath, dataset_type='train',
                                    transforms=self.transform, class_to_use=1)
        self.dataloader = DataLoader(self.dataset, batch_size=self.train_config.batch_size, shuffle=True,
                                     drop_last=True)

    def __init_model(self):
        self.netG = Generator(self.train_config)
        self.netG.apply(weights_init)
        self.netG.to(self.device)

        self.netD = Discriminator(self.train_config)
        self.netD.apply(weights_init)
        self.netD.to(self.device)
        self.sigmoid = nn.Sigmoid()

        self.noise = torch.FloatTensor(self.train_config.batch_size, self.train_config.nz, 1, 1).to(self.device)
        self.const_noise = torch.FloatTensor(self.train_config.batch_size, self.train_config.nz, 1, 1).to(
            self.device)
        self.const_noise.normal_(0, 1)

        self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=self.train_config.lr,
                                           betas=(self.train_config.beta1, 0.9))
        self.optimizerD = torch.optim.Adam(self.netD.parameters(), lr=self.train_config.lr,
                                           betas=(self.train_config.beta1, 0.9))

        self.criterionD = DiscriminatorLoss(self.train_config, device=self.device)
        self.criterionG = GeneratorLoss()

    def fit(self):

        for step in range(self.train_config.steps):
            batch_data = next(iter(self.dataloader))
            print('step : {}/{}'.format(step, self.train_config.steps))
            for j in range(self.train_config.discriminator_steps):
                self.optimizerD.zero_grad()
                real_imgs = batch_data['img'].to(self.device)
                real_imgs_var = Variable(real_imgs)
                output_D_real = self.netD(real_imgs_var)
                prob_D_read = self.sigmoid(output_D_real).cpu().detach()
                output_D_real = output_D_real.mean(dim=0)
                self.noise.normal_(0, 1)
                input_G = Variable(self.noise)
                fake_imgs = self.netG(input_G)
                fake_imgs_var = Variable(fake_imgs)
                output_D_fake = self.netD(fake_imgs_var)
                prob_D_fake = self.sigmoid(output_D_fake)
                output_D_fake = output_D_fake.mean(dim=0)
                lossD = self.criterionD(self.netD, fake_imgs, output_D_fake, real_imgs, output_D_real)
                lossD.backward()
                self.optimizerD.step()
                acc_real = (torch.round(prob_D_read) == torch.ones(
                    self.train_config.batch_size)).sum() / self.train_config.batch_size
                acc_fake = (torch.round(prob_D_fake) == torch.zeros(
                    self.train_config.batch_size)).sum() / self.train_config.batch_size
                if self.log:
                    self.logger.log_metrics('discriminator/loss', lossD.item(), self.D_global_step)
                    self.logger.log_metrics('discriminator/acc_real', acc_real.item(), self.D_global_step)
                    self.logger.log_metrics('discriminator/acc_fake', acc_fake.item(), self.D_global_step)

                self.D_global_step += 1


            self.optimizerG.zero_grad()
            self.noise.normal_(0, 1)
            input_G = Variable(self.noise)
            fake_imgs = self.netG(input_G)
            output_D_fake = self.netD(fake_imgs)
            output_D_fake = output_D_fake.mean(dim=0)
            lossG = self.criterionG(output_D_fake)
            lossG.backward()
            self.optimizerG.step()
            if self.log:
                self.logger.log_metrics('generator/loss', lossG, self.G_global_step)
            self.G_global_step += 1
            # print('acc_real {}, acc_fake {}, loss_D {}, loss_G {}'.format(acc_real.item(), acc_fake.item(), lossD.item(), lossG.item()))
            if step % 1000 == 0:
                fig = self.show_images(step)
                self.logger.log_images(f'const_noise_{step}', [fig])

    def show_images(self, step):
        inputG = Variable(self.const_noise)
        fake_imgs = self.netG(inputG).detach().cpu()
        fig = plt.figure(figsize=(8, 8))
        img = np.transpose(torchvision.utils.make_grid(fake_imgs, padding=2, normalize=True), (1, 2, 0))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'step = {step}')
        plt.show()
        return fig







if __name__ == '__main__':
    from configs.train_config import cfg as config

    trainer = Trainer(config, logger_config=None)
    trainer.fit()
