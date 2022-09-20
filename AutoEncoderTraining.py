import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import wandb
from AutoencoderDataset import npDataset
from Autoencoder import Encoder, Decoder
from GlobalValues import training_dir

im_dir = training_dir
n_epochs = 100
# n_latents = 512
batch_size = 4
learning_rate = 1e-5
log_interval = 10

wandb.init(project="wildfire-project")
wandb.config = {
    "learning_rate": 1e-5,
    "epochs": 100,
    "batch_size": 16
}
def train(train_loader, test_loader, encoder, decoder, opt):
    # Optional
    batch_size = len(train_loader)
    # print(batch_size)

    for epoch in range(n_epochs + 1):
        encoder.train()
        decoder.train()
        training_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.cuda(), y.cuda()
            x, y = torch.squeeze(x, dim=0), torch.squeeze(y, dim=0)
            opt.zero_grad()
            encoder_output = encoder(x)
            decoder_output = decoder(encoder_output)
            loss = F.mse_loss(decoder_output, y)
            loss.backward()
            opt.step()
            wandb.log({"batch_loss": loss, "epoch": epoch})
            training_loss += loss.item()
            if batch_idx % log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader.dataset)} ({loss.item()})]')
        wandb.log({"training_loss": training_loss / batch_size, "epoch": epoch})
        encoder.eval()
        decoder.eval()
        test_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                x, y = torch.squeeze(x, dim=0), torch.squeeze(y, dim=0)
                encoder_output = encoder(x)
                decoder_output = decoder(encoder_output)
                val_loss = F.mse_loss(decoder_output, y)
                test_loss += val_loss

        test_loss /= len(test_loader.dataset)
        wandb.log({"val_loss": test_loss, "epoch": epoch})
        print(f'Test Reconstruction Loss: ({test_loss})]')


def main():
    # Get List of downloaded files and set up reference_data loader
    file_list = os.listdir(im_dir)
    print(f'{len(file_list)} reference_data samples found')
    train_files, test_files = train_test_split(file_list, test_size=0.2, random_state=42)
    train_files, validation_files = train_test_split(train_files, test_size=0.2, random_state=42)

    train_loader = DataLoader(npDataset(train_files, batch_size,im_dir))
    validation_loader = DataLoader(npDataset(validation_files, batch_size,im_dir))
    test_loader = DataLoader(npDataset(test_files, batch_size,im_dir))
    print(
        f'Training {len(train_files)} reference_data samples , validation {len(validation_files)} and testing {len(test_files)}')
    # Set up the encoder, decoder. and optimizer
    encoder = Encoder(1)
    decoder = Decoder(256)
    encoder.cuda()
    decoder.cuda()
    opt = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

    # Train and save the model components
    train(train_loader, validation_loader, encoder, decoder, opt)
    torch.save(encoder.state_dict(), 'SuperRes_Encoder.pth')
    torch.save(decoder.state_dict(), 'SuperRes_Decoder.pth')
    torch.save(opt.state_dict(), 'SuperRes_Opt.pth')


if __name__ == "__main__":
    main()
