import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau  
from tensorboardX import SummaryWriter

class Discriminator(nn.Module):
    def __init__(self, input_size, condition_size):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.condition_size = condition_size

        self.fc1 = nn.Linear(self.input_size + self.condition_size, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x, c):
        xc = torch.cat((x, c), dim=1)
        x1 = F.relu(self.fc1(xc))
        x2 = torch.sigmoid(self.fc2(x1))
        return x2

class Generator(nn.Module):
    def __init__(self, input_size, condition_size, latent_size):
        super(Generator, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.condition_size = condition_size

        self.layers = nn.Sequential(
            nn.Linear(self.latent_size + self.condition_size, 1028),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1028),
            nn.Linear(1028, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
        )
        self.output3 = nn.Linear(128, self.input_size)

    def forward(self, z, c):
        zc = torch.cat((z, c), dim=1)
        x = self.layers(zc)
        x = self.output3(x)
        return x

class CGAN(nn.Module):
    def __init__(self, input_size, latent_size, condition_size,device, type='FCN'):
        super(CGAN, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.condition_size = condition_size
        self.device = device
        self.to(self.device)
        
        if type=='FCN':
            self.discriminator = Discriminator(self.input_size, self.condition_size)
            self.generator = Generator(self.input_size, self.condition_size, self.latent_size)
        
        
    def loss_function(self, real_output, fake_output):
        real_loss = F.binary_cross_entropy(real_output, torch.ones_like(real_output))
        fake_loss = F.binary_cross_entropy(fake_output, torch.zeros_like(fake_output))
        return real_loss + fake_loss


    def generate_samples(self, num_samples, condition, device):
        self.eval()
        c = condition.repeat(num_samples, 1).to(device)
        z = torch.randn((num_samples, self.latent_size)).to(device)

        with torch.no_grad():
            generated_samples = self.generator(z, c)

        return generated_samples.cpu().numpy()
        
    def train_model(self, train_loader, val_loader, epochs, output_dir, device, lr=0.001, beta1=0.01):

        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr, betas=(beta1, 0.999))
        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr, betas=(beta1, 0.999))

        #Set up summary writer
        os.makedirs(output_dir, exist_ok=True)
        writer = SummaryWriter(output_dir)

        for epoch in range(epochs):

            #TRAINING
            d_train_losses = []
            g_train_losses = []
            for x, c in train_loader:
                x = x.to(device,dtype=torch.float32)
                c = c.to(device,dtype=torch.float32)
                
                #Train discriminator
                d_optimizer.zero_grad()

                real_output = self.discriminator(x, c)
                z = torch.randn((x.size(0), self.latent_size)).to(device)
                fake_x = self.generator(z, c)
                fake_output = self.discriminator(fake_x.detach(), c)

                d_loss = self.loss_function(real_output, fake_output)
                d_loss.backward()
                d_optimizer.step()

                # Train generator
                g_optimizer.zero_grad()

                fake_output = self.discriminator(fake_x, c)
                g_loss = F.binary_cross_entropy(fake_output, torch.ones_like(fake_output))
                g_loss.backward()
                g_optimizer.step()

                d_train_losses.append(d_loss.item())
                g_train_losses.append(g_loss.item())

            d_train_loss_mean = sum(d_train_losses) / len(d_train_losses)
            g_train_loss_mean = sum(g_train_losses) / len(g_train_losses)
            #ADD TO SUMMARY WRITER
            writer.add_scalar("Loss/train/D_loss", d_train_loss_mean, epoch)
            writer.add_scalar("Loss/train/G_loss", g_train_loss_mean, epoch)
            #VALIDATION
            with torch.no_grad():
                d_val_losses = []
                g_val_losses = []
                for x_val, c_val in val_loader:
                    x_val = x_val.to(device, dtype=torch.float32)
                    c_val = c_val.to(device, dtype=torch.float32)

                    real_output = self.discriminator(x_val, c_val)
                    z_val = torch.randn((x_val.size(0), self.latent_size)).to(device)
                    fake_x_val = self.generator(z_val, c_val)
                    fake_output_val = self.discriminator(fake_x_val, c_val)

                    d_val_loss = self.loss_function(real_output, fake_output_val)
                    g_val_loss = F.binary_cross_entropy(fake_output_val, torch.ones_like(fake_output_val))

                    d_val_losses.append(d_val_loss.item())
                    g_val_losses.append(g_val_loss.item())

                d_val_loss_mean = sum(d_val_losses) / len(d_val_losses)
                g_val_loss_mean = sum(g_val_losses) / len(g_val_losses)
            
                writer.add_scalar("Loss/val/D_loss", d_val_loss_mean, epoch)
                writer.add_scalar("Loss/val/G_loss", g_val_loss_mean, epoch)
            
            if (epoch+1) % 10 == 0 and epoch != 0:
                print(f"Epoch {epoch + 1}/{epochs}, D_Loss: {d_train_loss_mean}, G_Loss: {d_train_loss_mean},  Validation D_Loss: {d_val_loss_mean},  Validation G_Loss: {g_val_loss_mean}")

        writer.close()