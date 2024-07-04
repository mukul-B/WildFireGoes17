import os
from datetime import datetime

import torch
torch.cuda.empty_cache()
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import logging

import wandb
from Autoencoder import Encoder, Decoder
from AutoencoderDataset import npDataset
from GlobalValues import GOES_Bands, training_dir, model_path, RES_ENCODER_PTH, RES_DECODER_PTH, RES_OPT_PTH, BATCH_SIZE, EPOCHS, \
    LEARNING_RATE, random_state, BETA, LOSS_FUNCTION
from LossFunctionConfig import SWEEP_OPERATION, use_config,sweep_loss_funtion

im_dir = training_dir
log_interval = 10


def test_accuracy(test_loader, encoder, decoder, criteria, epoch):
    # evaluate model
    encoder.eval()
    decoder.eval()
    validation_loss = 0
    vloss2 = 0
    vloss1 = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            x, y = torch.squeeze(x, dim=0), torch.squeeze(y, dim=0)
            encoder_output = encoder(x)
            decoder_output = decoder(encoder_output)
            target = y
            # if epoch > 100 else x
            val_loss = criteria(decoder_output, target)
            if type(val_loss) == tuple and len(val_loss) == 3:
                loss1, loss2,val_loss = val_loss[0], val_loss[1],val_loss[2]
                vloss2 += loss2
                vloss1 += loss1
                # val_loss = conbine_loss(loss1, loss2)
            validation_loss += val_loss
    validation_loss /= len(test_loader.dataset)
    vloss2 /= len(test_loader.dataset)
    vloss1 /= len(test_loader.dataset)
    return validation_loss, vloss1, vloss2


def train(train_loader, test_loader, encoder, decoder, optimizer, n_epochs, criteria):
    batch_size = len(train_loader)
    # wandb.watch(decoder, log_freq=10)
    #  training for each epoch
    early_stop = 30
    early_stop_loss = 0
    scheduler = ReduceLROnPlateau(optimizer, 'min',threshold=1e-5)
    for epoch in range(n_epochs + 1):
        #  tells your model that you are training the model
        encoder.train()
        decoder.train()
        # per epoch training loss
        training_loss = 0
        tloss_loss2 = 0
        tloss_loss1 = 0
        loses_count = 0
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




            wandb.log({"output": torch.sum(decoder_output[0]), "epoch": epoch})
            target = y
            # if epoch >100 else x
            loss = criteria(decoder_output, target)
            if loses_count == 0:
                loses_count = len(loss) if type(loss) == tuple else 1
            if loses_count == 3:
                loss1, loss2,loss = loss[0], loss[1],loss[2]
                # loss1, loss2 = local_rmse,global_rmse
                # loss1, loss2 = rmse_loss,jaccard_loss
                # loss = conbine_loss(loss1, loss2)

            # backtracking and optimizer step
            loss.backward()
            optimizer.step()
            # print statistics
            if loses_count == 3:
                tloss_loss1 += loss1.item()
                tloss_loss2 += loss2.item()

            training_loss += loss.item()
            if batch_idx % log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader.dataset)} ({loss.item()})]')

        print(f"training_loss : {training_loss / batch_size} epoch: {epoch}")
        wandb.log({"training_loss": training_loss / batch_size, "epoch": epoch})

        loss = training_loss / batch_size
        if early_stop_loss == loss:
            early_stop -= 1
        else:
            early_stop = 30
        early_stop_loss = loss
        if early_stop == 0:
            print("early stopping")
            break

        # Validation loss
        validation_loss, vloss1, vloss2 = test_accuracy(test_loader, encoder, decoder, criteria, epoch)
        wandb.log({"val_loss": validation_loss, "epoch": epoch})
        if(len(optimizer.param_groups)>1):
            print("this")
        scheduler.step(validation_loss)
        wandb.log({"lr": optimizer.param_groups[0]['lr'], "epoch": epoch})
        if (loses_count == 3):
            wandb.log({"tloss1": tloss_loss1 / batch_size, "epoch": epoch})
            wandb.log({"tloss2": tloss_loss2 / batch_size, "epoch": epoch})
            wandb.log({"vloss1": vloss1, "epoch": epoch})
            wandb.log({"vloss2": vloss2, "epoch": epoch})

        print(f'validation Loss: ({validation_loss})]')
    print(f"Finished Training")


def main(config=None):
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H:%M:%S")
    print("Current Time =", current_time)

    if config:
        wandb.config = config
    else:
        wandb.init()


        # setting hyper parameters
    #
    n_epochs = wandb.config.get(EPOCHS)
    batch_size = wandb.config.get(BATCH_SIZE)
    learning_rate = wandb.config.get(LEARNING_RATE)
    beta = wandb.config.get(BETA)
    loss_function = wandb.config.get(LOSS_FUNCTION)

    loss_function = sweep_loss_funtion if loss_function is None else loss_function
    loss_function_name = str(loss_function).split("'")[1].split(".")[1]
    project_name = f"wildfire_{loss_function_name}_{n_epochs}epochs_{batch_size}batchsize_{learning_rate}lr"
    print(project_name)
    run = wandb.init(project=project_name, name="run_" + current_time)
    print(f'Train with n_epochs : {n_epochs} , batch_size : {batch_size} , learning_rate : {learning_rate}')
    print(f'beta : {beta}, loss function :{loss_function}')
    # loss Function
    criteria = loss_function(beta)
    # criteria = two_branch_loss(beta)
    OUTPUT_ACTIVATION = criteria.last_activation if criteria.last_activation else "relu"
    # Set up the encoder, decoder. and optimizer
    encoder = Encoder(GOES_Bands)
    decoder = Decoder(256, OUTPUT_ACTIVATION)
    encoder.cuda()
    decoder.cuda()
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

    # Get List of downloaded files and set up reference_data loader
    file_list = os.listdir(im_dir)
    print(f'{len(file_list)} reference_data samples found')
    train_files, test_files = train_test_split(file_list, test_size=0.2, random_state=random_state)
    train_files, validation_files = train_test_split(train_files, test_size=0.2, random_state=random_state)

    train_loader = DataLoader(npDataset(train_files, batch_size, im_dir,True,False), shuffle=True)
    validation_loader = DataLoader(npDataset(validation_files, batch_size, im_dir,True,False), shuffle=False)
    # test_loader = DataLoader(npDataset(test_files, batch_size, im_dir))
    print(
        f'Training sample : {len(train_files)} , validation samples : {len(validation_files)} , testing samples : {len(test_files)}')

    # Train and save the model components
    train(train_loader, validation_loader, encoder, decoder, optimizer, n_epochs, criteria)
    mp = model_path  + project_name
    end =  datetime.now()
    duration = end - now
    seconds = duration.total_seconds()
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    if not os.path.exists(mp):
        os.mkdir(mp)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(f"{mp}/training.log"),
            logging.StreamHandler()
        ]
    )
    logging.info(f'{current_time} run took {hours} hours {minutes} minutes {seconds} seconds\n details : {wandb.run.get_url()}')
    torch.save(encoder.state_dict(), mp + "/" + RES_ENCODER_PTH)
    torch.save(decoder.state_dict(), mp + "/" + RES_DECODER_PTH)
    torch.save(optimizer.state_dict(), mp + "/" + RES_OPT_PTH)


if __name__ == "__main__":
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if SWEEP_OPERATION:
        # Initialize sweep by passing in config. (Optional) Provide a name of the project.
        # # wandb.login()
        from LossFunctionConfig import sweep_configuration_IOU_LRMSE
        sweep_configuration = sweep_configuration_IOU_LRMSE
        sweep_id = wandb.sweep(sweep=sweep_configuration, project='sweep_config')
        wandb.agent(sweep_id, function=main, count=1)
    else:
        config = use_config
        main(config)

# https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
