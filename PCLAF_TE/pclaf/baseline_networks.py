"""Convolutional VAE and BCE-GAN networks for 24-channel endpoint pairs."""
import numpy as np
import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_channels=19, time_steps=20, latent_dim=100):
        super(Encoder, self).__init__()
        self.channels = input_channels
        self.time_steps = time_steps
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(

            nn.Conv1d(input_channels, 19, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(19),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),

            nn.Conv1d(19, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),

            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),

            nn.Flatten()
        )

        flattened_size = self._calculate_flattened_size()

        self.fc_mu = nn.Linear(flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(flattened_size, latent_dim)

    def _calculate_flattened_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.channels, self.time_steps)
            x = self.encoder(dummy_input)
            return x.size(1)

    def forward(self, x):

        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=100, output_channels=19, output_time_steps=20):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.output_time_steps = output_time_steps

        self.init_time_steps = 5
        self.init_channels = 64

        self.fc = nn.Linear(latent_dim, self.init_channels * self.init_time_steps)

        self.decoder = nn.Sequential(

            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),

            nn.ConvTranspose1d(32, 19, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(19),
            nn.ReLU(True),

            nn.ConvTranspose1d(19, output_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, z):

        x = self.fc(z)
        x = x.view(x.size(0), self.init_channels, self.init_time_steps)
        x = self.decoder(x)
        return x


class VAE19Channel(nn.Module):
    def __init__(self, input_channels=19, time_steps=20, latent_dim=100, beta=0.0):
        super(VAE19Channel, self).__init__()
        self.input_channels = input_channels
        self.time_steps = time_steps
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = Encoder(input_channels, time_steps, latent_dim)
        self.decoder = Decoder(latent_dim, input_channels, time_steps)

        self.device = torch.device("cpu")
        self.to(self.device)

        self.train_losses = []
        self.recon_losses = []
        self.kl_losses = []

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):

        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)

        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):

        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum') / x.size(0)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        total_loss = recon_loss + self.beta * kl_div

        return total_loss, recon_loss, kl_div

    def generate(self, num_samples=1, z=None):

        self.eval()
        with torch.no_grad():
            if z is None:

                z = torch.randn(num_samples, self.latent_dim).to(self.device)
            generated = self.decoder(z)
        self.train()
        return generated

    def reconstruct(self, x):

        self.eval()
        with torch.no_grad():
            mu, logvar = self.encoder(x)
            z = self.reparameterize(mu, logvar)
            recon_x = self.decoder(z)
        self.train()
        return recon_x

    def get_latent_representation(self, x):

        self.eval()
        with torch.no_grad():
            mu, logvar = self.encoder(x)
            z = self.reparameterize(mu, logvar)
        self.train()
        return z


class Generator(nn.Module):
    def __init__(self, noise_dim=100, channels=19, time_steps=20):
        super(Generator, self).__init__()
        self.channels = channels
        self.time_steps = time_steps
        self.noise_dim = noise_dim
        self.init_time_steps = 5
        self.init_channels = 128
        self.init_features = self.init_channels * self.init_time_steps
        self.fc = nn.Linear(noise_dim, self.init_features)
        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(True)
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose1d(32, channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), self.init_channels, self.init_time_steps)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, channels=19, time_steps=20):
        super(Discriminator, self).__init__()
        self.channels = channels
        self.time_steps = time_steps


        self.down1 = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25)
        )

        self.down2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25)
        )

        self.down3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25)
        )


        flattened_size = self.calculate_flattened_size()

        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def calculate_flattened_size(self):

        with torch.no_grad():
            dummy_input = torch.zeros(1, self.channels, self.time_steps)
            x = self.down1(dummy_input)
            x = self.down2(x)
            x = self.down3(x)
            return int(np.prod(x.size()[1:]))

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
