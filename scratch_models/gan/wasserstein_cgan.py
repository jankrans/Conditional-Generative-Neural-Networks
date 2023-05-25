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

        self.model = nn.Sequential(
            nn.Linear(self.input_size + self.condition_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x, c):
        xc = torch.cat((x, c), dim=1)
        x2 = self.model(xc)
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

class Was_CGAN(nn.Module):
    def __init__(self, input_size, latent_size, condition_size,device, type='FCN'):
        super(Was_CGAN, self).__init__()
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


    def wasserstein_loss(self, real_output, fake_output):
        real_loss = -torch.mean(real_output)
        fake_loss = torch.mean(fake_output)
        return real_loss + fake_loss
    
    def gradient_penalty(self, real_samples, fake_samples, condition, device, gp_weight=10):
        alpha = torch.rand(real_samples.size(0), 1).to(device)
        alpha = alpha.expand(real_samples.size()).to(device)

        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True).to(device)
        d_interpolates = self.discriminator(interpolates, condition)
        gradients = torch.autograd.grad(
            outputs=d_interpolates, inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = gp_weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
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
            for i,(x, c) in enumerate(train_loader):
                x = x.to(device,dtype=torch.float32)
                c = c.to(device,dtype=torch.float32)
                
                #Train discriminator
                d_optimizer.zero_grad()

                real_output = self.discriminator(x, c)
                z = torch.randn((x.size(0), self.latent_size)).to(device)
                fake_x = self.generator(z, c)
                fake_output = self.discriminator(fake_x.detach(), c)

                d_loss = self.wasserstein_loss(real_output, fake_output)
                gp = self.gradient_penalty(x, fake_x, c, device)
                d_loss_with_gp = d_loss + gp
                d_loss_with_gp.backward()
                d_optimizer.step()

                if (i + 1) % 5 == 0:  # Update generator every 5 steps
                    g_optimizer.zero_grad()

                    z_new = torch.randn((x.size(0), self.latent_size)).to(device)
                    fake_x_new = self.generator(z_new, c)
                    fake_output_new = self.discriminator(fake_x_new, c)
                    g_loss = -torch.mean(fake_output_new)
                    g_loss.backward()
                    g_optimizer.step()

                    g_train_losses.append(g_loss.item())
                    g_train_loss_mean = sum(g_train_losses) / len(g_train_losses)
                    writer.add_scalar("Loss/train/G_loss", g_train_loss_mean, epoch)
                
                d_train_losses.append(d_loss.item())

            d_train_loss_mean = sum(d_train_losses) / len(d_train_losses)
            
            #ADD TO SUMMARY WRITER
            writer.add_scalar("Loss/train/D_loss", d_train_loss_mean, epoch)
            
            #VALIDATION
            with torch.no_grad():
                d_val_losses = []
                g_val_losses = []
                for x_val, c_val in val_loader:
                    x_val = x_val.to(device, dtype=torch.float32)
                    c_val = c_val.to(device, dtype=torch.float32)

                    real_output_val = self.discriminator(x_val, c_val)
                    z_val = torch.randn((x_val.size(0), self.latent_size)).to(device)
                    fake_x_val = self.generator(z_val, c_val)
                    fake_output_val = self.discriminator(fake_x_val, c_val)

                    d_val_loss = self.wasserstein_loss(real_output_val, fake_output_val)
                    d_val_losses.append(d_val_loss.item())

                    g_val_loss = -torch.mean(fake_output_val)
                    g_val_losses.append(g_val_loss.item())

                d_val_loss_mean = sum(d_val_losses) / len(d_val_losses)
                g_val_loss_mean = sum(g_val_losses) / len(g_val_losses)
            
                writer.add_scalar("Loss/val/D_loss", d_val_loss_mean, epoch)
                writer.add_scalar("Loss/val/G_loss", g_val_loss_mean, epoch)
            
            if (epoch+1) % 10 == 0 and epoch != 0:
                print(f"Epoch {epoch + 1}/{epochs}, D_Loss: {d_train_loss_mean}, G_Loss: {d_train_loss_mean},  Validation D_Loss: {d_val_loss_mean},  Validation G_Loss: {g_val_loss_mean}")

        writer.close()