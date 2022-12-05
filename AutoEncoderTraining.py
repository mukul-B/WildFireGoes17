import os
from datetime import datetime

import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import wandb
from Autoencoder import Encoder, Decoder
from AutoencoderDataset import npDataset
from GlobalValues import training_dir, model_path, RES_ENCODER_PTH, RES_DECODER_PTH, RES_OPT_PTH, BATCH_SIZE, EPOCHS, \
    LEARNING_RATE, random_state, BETA, LOSS_FUNCTION
from HyperparameterConfigs import use_config
from LossFunctions import GetLossFunction

im_dir = training_dir
log_interval = 10


def test_accuracy(test_loader, encoder, decoder, criteria):
    # evaluate model
    encoder.eval()
    decoder.eval()
    validation_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            x, y = torch.squeeze(x, dim=0), torch.squeeze(y, dim=0)
            encoder_output = encoder(x)
            decoder_output = decoder(encoder_output)
            val_loss = criteria(decoder_output, y)
            validation_loss += val_loss
    validation_loss /= len(test_loader.dataset)
    return validation_loss


def train(train_loader, test_loader, encoder, decoder, optimizer, n_epochs, criteria):
    batch_size = len(train_loader)
    # wandb.watch(decoder, log_freq=10)
    #  training for each epoch
    for epoch in range(n_epochs + 1):
        #  tells your model that you are training the model
        encoder.train()
        decoder.train()
        # per epoch training loss
        training_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            # get the inputs
            x, y = x.cuda(), y.cuda()
            x, y = torch.squeeze(x, dim=0), torch.squeeze(y, dim=0)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            # output of encoder
            encoder_output = encoder(x)
            # output of decoder
            decoder_output = decoder(encoder_output)
            wandb.log({"output": torch.sum(decoder_output), "epoch": epoch})
            loss = criteria(decoder_output, y)
            # backtracking and optimizer step
            loss.backward()
            optimizer.step()
            # print statistics
            training_loss += loss.item()
            if batch_idx % log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader.dataset)} ({loss.item()})]')

        print(f"training_loss : {training_loss / batch_size} epoch: {epoch}")
        wandb.log({"training_loss": training_loss / batch_size, "epoch": epoch})

        # Validation loss
        validation_loss = test_accuracy(test_loader, encoder, decoder, criteria)
        wandb.log({"val_loss": validation_loss, "epoch": epoch})
        print(f'validation Loss: ({validation_loss})]')
    print(f"Finished Training")


def main(config=None):
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H:%M:%S")
    print("Current Time =", current_time)
    run = wandb.init(project="wildfire-project_Iou_LRMSE'", name="run_" + current_time)
    if config:
        wandb.config = config
    # setting hyper parameters
    n_epochs = wandb.config.get(EPOCHS)
    batch_size = wandb.config.get(BATCH_SIZE)
    learning_rate = wandb.config.get(LEARNING_RATE)
    beta = wandb.config.get(BETA)
    loss_function = wandb.config.get(LOSS_FUNCTION)
    print(f'Train with n_epochs : {n_epochs} , batch_size : {batch_size} , learning_rate : {learning_rate}')
    print(f'beta : {beta}, loss function :{loss_function}')
    # loss Function
    criteria = GetLossFunction(loss_function)
    # Set up the encoder, decoder. and optimizer
    encoder = Encoder(1)
    decoder = Decoder(256)
    encoder.cuda()
    decoder.cuda()
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

    # Get List of downloaded files and set up reference_data loader
    file_list = os.listdir(im_dir)
    print(f'{len(file_list)} reference_data samples found')
    train_files, test_files = train_test_split(file_list, test_size=0.2, random_state=random_state)
    train_files, validation_files = train_test_split(train_files, test_size=0.2, random_state=random_state)

    train_loader = DataLoader(npDataset(train_files, batch_size, im_dir), shuffle=True)
    validation_loader = DataLoader(npDataset(validation_files, batch_size, im_dir), shuffle=True)
    # test_loader = DataLoader(npDataset(test_files, batch_size, im_dir))
    print(
        f'Training sample : {len(train_files)} , validation samples : {len(validation_files)} , testing samples : {len(test_files)}')

    # Train and save the model components
    train(train_loader, validation_loader, encoder, decoder, optimizer, n_epochs, criteria)
    torch.save(encoder.state_dict(), model_path + RES_ENCODER_PTH)
    torch.save(decoder.state_dict(), model_path + RES_DECODER_PTH)
    torch.save(optimizer.state_dict(), model_path + RES_OPT_PTH)


if __name__ == "__main__":
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    # Initialize sweep by passing in config. (Optional) Provide a name of the project.
    # wandb.login()
    # sweep_configuration = sweep_configuration_IOU_LRMSE
    # sweep_id = wandb.sweep(sweep=sweep_configuration, project='sweep_Iou_LRMSE')
    # wandb.agent(sweep_id, function=main, count=50)

    config = use_config
    main(config)

# https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
