import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau  
from tensorboardX import SummaryWriter

class EncoderCNN(nn.Module):
    def __init__(self, input_size, condition_size, latent_size):
        super(EncoderCNN, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.condition_size = condition_size

        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(self.input_size * 128 // 2 + self.condition_size, 256)

        self.fc_mu = nn.Linear(256, self.latent_size)
        self.fc_var = nn.Linear(256, self.latent_size)

    def forward(self, x, c):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)

        xc = torch.cat((x, c), dim=1)
        i1 = F.relu(self.fc1(xc))

        mu = self.fc_mu(i1)
        logvar = self.fc_var(i1)
        return mu, logvar

class DecoderCNN(nn.Module):
    def __init__(self, input_size, condition_size, latent_size):
        super(DecoderCNN, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.condition_size = condition_size

        self.fc1 = nn.Linear(self.latent_size + self.condition_size, 256)
        self.fc2 = nn.Linear(256, 128 * self.input_size)

        self.deconv1 = nn.ConvTranspose1d(128, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.deconv2 = nn.ConvTranspose1d(64, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.deconv3 = nn.ConvTranspose1d(32, 1, kernel_size=5, padding=2)

    def forward(self, z, c):
        zc = torch.cat((z, c), dim=1)

        x = F.relu(self.fc1(zc))
        x = F.relu(self.fc2(x))

        x = x.view(x.size(0), 128, -1)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = self.deconv3(x)

        output = x.squeeze(1)
        return output

class Encoder(nn.Module):

    def __init__(self, input_size, condition_size, latent_size):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.condition_size = condition_size

        self.layers = nn.Sequential(
            nn.Linear(self.input_size + self.condition_size, 1028),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(1028),
            nn.Linear(1028, 512),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(128)
        )

        self.mu = nn.Linear(128, self.latent_size)
        self.var = nn.Linear(128, self.latent_size)

    def forward(self, x, c):
        xc = torch.cat((x,c),dim=1)
        i1 = self.layers(xc)
        mu = self.mu(i1)
        logvar = self.var(i1)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, input_size, condition_size, latent_size):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.condition_size = condition_size

        self.layers = nn.Sequential(
            nn.Linear(self.latent_size + self.condition_size, 1028),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(1028),
            nn.Linear(1028, 512),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(128),
        )
        self.output3 = nn.Linear(128, self.input_size)

    def forward(self, z, c):
        zc = torch.cat((z,c),dim=1)
        i1 = self.layers(zc)
        output = self.output3(i1)
        return output

class CVAE(nn.Module):
    def __init__(self, input_size, latent_size, condition_size,device, type='FCN'):
        super(CVAE, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.condition_size = condition_size
        self.device = device
        self.to(self.device)
        
        if type=='FCN':
            self.encoder = Encoder(self.input_size, self.condition_size, self.latent_size)
            self.decoder = Decoder(self.input_size, self.condition_size, self.latent_size)
        elif type == 'CNN':
            self.encoder = EncoderCNN(self.input_size, self.condition_size, self.latent_size)
            self.decoder = DecoderCNN(self.input_size, self.condition_size, self.latent_size)
        #this should be lstm network with recurrent behaviour
        else:
            self.encoder = EncoderCNN(self.input_size, self.condition_size, self.latent_size)
            self.decoder = DecoderCNN(self.input_size, self.condition_size, self.latent_size)
        
    def forward(self, x, c):

        mu, logvar = self.encoder(x,c)

        # Create a sample
        batch_size = x.shape[0]

        std = torch.exp(0.5 * logvar)
        eps = torch.randn((batch_size,self.latent_size)).to(self.device)
        z = mu + eps * std

        x_hat = self.decoder(z,c)
        return x_hat, mu, logvar

        
    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return (recon_loss + kld_loss)/x.shape[0], recon_loss/x.shape[0], kld_loss/x.shape[0]


    def generate_samples(self,num_samples,condition,device):
        self.eval() 

        c = condition.repeat(num_samples, 1).to(device)

        z = torch.randn((num_samples, self.latent_size)).to(device)

        # Generate new samples by passing it through the decoder
        with torch.no_grad():
            generated_samples = self.decoder(z, c)

        return generated_samples.cpu().numpy()
        
    def train_model(self, train_loader, val_loader, epochs, output_dir, device, lr=0.001, patience=20, optim_scheduler=False):

        #define optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr)
        if optim_scheduler:
            scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=20, verbose=True)

        #Set up summary writer
        os.makedirs(output_dir, exist_ok=True)
        writer = SummaryWriter(output_dir)

        best_val_loss = float('inf')
        epochs_since_improvement = 0

        for epoch in range(epochs):

            #TRAINING
            self.train()
            epoch_loss, epoch_recon_loss, epoch_kld_loss = 0, 0, 0
            for x, c in train_loader:
                x = x.to(device,dtype=torch.float32)
                c = c.to(device,dtype=torch.float32)
                optimizer.zero_grad()

                pred_x, mu, logvar = self.forward(x,c)

                loss, recon_loss, kld_loss = self.loss_function(pred_x, x, mu, logvar)
                epoch_loss += loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kld_loss += kld_loss.item()
                loss.backward()
                optimizer.step()

            #ADD TO SUMMARY WRITER
            writer.add_scalar('Loss/train', epoch_loss / len(train_loader), epoch)
            writer.add_scalar('Loss/train_recon', epoch_recon_loss / len(train_loader), epoch)
            writer.add_scalar('Loss/train_kld', epoch_kld_loss / len(train_loader), epoch)

            #VALIDATION
            self.eval()
            val_epoch_loss, val_epoch_recon_loss, val_epoch_kld_loss = 0, 0, 0
            with torch.no_grad():
                for x_val, c_val in val_loader:
                    x_val = x_val.to(device)
                    c_val = c_val.to(device)

                    # Forward pass
                    pred_x_val, mu, logvar = self.forward(x_val, c_val)

                    # Calculate loss
                    val_loss, val_recon_loss, val_kld_loss  = self.loss_function(pred_x_val, x_val, mu, logvar)
                    val_epoch_loss += val_loss.item()
                    val_epoch_recon_loss += val_recon_loss.item()
                    val_epoch_kld_loss += val_kld_loss.item()

            val_epoch_loss = val_epoch_loss / len(val_loader)
            
            if optim_scheduler:
                scheduler.step(val_epoch_loss)
            
            #ADD TO SUMMARY WRITER
            writer.add_scalar('Loss/val', val_epoch_loss, epoch)
            writer.add_scalar('Loss/val_recon', val_epoch_recon_loss / len(val_loader), epoch)
            writer.add_scalar('Loss/val_kld', val_epoch_kld_loss / len(val_loader), epoch)

            #EARLY STOPPING & SAVING
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                torch.save(self.state_dict(), os.path.join(output_dir,'best_model.pt'))
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if epochs_since_improvement >= patience:
                print("Early stopping triggered.")
                break
            
            if (epoch+1) % 10 == 0 and epoch != 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader)}, Validation Loss: {val_epoch_loss}")

        writer.close()