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
from Autoencoder import Autoencoder, Encoder, Decoder
# from CustomDataset import npDataset
from CustomDataset_with_MASK import npDataset
from GlobalValues import RES_AUTOENCODER_PTH, GOES_Bands, training_dir, model_path, RES_ENCODER_PTH, RES_DECODER_PTH, RES_OPT_PTH, BATCH_SIZE, EPOCHS, \
    LEARNING_RATE, random_state, BETA, LOSS_FUNCTION, project_name_template, validation_split, test_split, model_specific_postfix
from ModelRunConfiguration import SWEEP_OPERATION, use_config,sweep_loss_funtion

im_dir = training_dir
log_interval = 10


def test_accuracy(test_loader, selected_model, criteria, epoch):
    # evaluate model

    selected_model.eval()
    validation_loss = 0
    vloss2 = 0
    vloss1 = 0
    with torch.no_grad():
        for x, y,k in test_loader:
            x, y = x.cuda(), y.cuda()
            x, y = torch.squeeze(x, dim=0), torch.squeeze(y, dim=0)
            k = k.cuda()
            k = torch.squeeze(k, dim=0)

            decoder_output = selected_model(x)
            target = y
            # if epoch > 100 else x
            val_loss = criteria(decoder_output[k!=0], target[k!=0])
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


def train(train_loader, test_loader, selected_model, optimizer, n_epochs, criteria):
    batch_size = len(train_loader)
    # wandb.watch(decoder, log_freq=10)
    #  training for each epoch
    early_stop = 30
    early_stop_loss = 0
    scheduler = ReduceLROnPlateau(optimizer, 'min',threshold=1e-5)
    for epoch in range(n_epochs + 1):
        #  tells your model that you are training the model
        selected_model.train()
        # per epoch training loss
        training_loss = 0
        tloss_loss2 = 0
        tloss_loss1 = 0
        loses_count = 0
        for batch_idx, (x, y,k) in enumerate(train_loader):
            # get the inputs
            x, y = x.cuda(), y.cuda()
            k = k.cuda()
            k = torch.squeeze(k, dim=0)
            x, y = torch.squeeze(x, dim=0), torch.squeeze(y, dim=0)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize

            decoder_output = selected_model(x)
            # wandb.log({"output": torch.sum(decoder_output[0]), "epoch": epoch})
            target = y
            loss = criteria(decoder_output[k!=0], target[k!=0])
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
        validation_loss, vloss1, vloss2 = test_accuracy(test_loader, selected_model, criteria, epoch)
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

def train_runner( selected_model, n_epochs, batch_size, criteria, optimizer):
    # Get List of downloaded files and set up reference_data loader
    file_list = os.listdir(im_dir)
    file_list = balance_dataset_if_TH(file_list)
    print(f'{len(file_list)} reference_data samples found')
    train_files, test_files = train_test_split(file_list, test_size=test_split, random_state=random_state)
    train_files, validation_files = train_test_split(train_files, test_size=validation_split, random_state=random_state)

    train_loader = DataLoader(npDataset(train_files, batch_size, im_dir,True,False), shuffle=True)
    validation_loader = DataLoader(npDataset(validation_files, batch_size, im_dir,True,False), shuffle=False)
    # test_loader = DataLoader(npDataset(test_files, batch_size, im_dir))
    logging.info(
        f'Training sample : {len(train_files)} , validation samples : {len(validation_files)} , testing samples : {len(test_files)}')
    #starting training
    selected_model.cuda()
    train(train_loader, validation_loader, selected_model, optimizer, n_epochs, criteria)

def balance_dataset_if_TH(file_list):
    logging.info(f'{len(file_list)} reference_data samples found')
    
    # positive_scoop , th_scoop , negitive_scoop  = 1,0.91,0.95
    # positive_scoop , th_scoop , negitive_scoop  = 0.6,0.9,0.9
    # positive_scoop , th_scoop , negitive_scoop  = 0,1,0.58
    # positive_scoop , th_scoop , negitive_scoop  = 1,1,0.35
    # positive_scoop , th_scoop , negitive_scoop  = 1,0,0.89
    # positive_scoop , th_scoop , negitive_scoop  = 1,0.73,0
    positive_scoop , th_scoop , negitive_scoop  = 1,0,0

    # positive_scoop , th_scoop , negitive_scoop  = 0,1,0


    file_list_pos = os.listdir(im_dir.replace('training_data','training_data_pos'))
    file_list_neg = os.listdir(im_dir.replace('training_data','training_data_neg'))
    file_list_TH = os.listdir(im_dir.replace('training_data','training_data_TH'))


    file_list_pos, reject_pos = train_test_split(file_list_pos, test_size=positive_scoop, random_state=random_state) if(positive_scoop != 0) else [[],[]]
    file_list_neg, reject_neg = train_test_split(file_list_neg, test_size=negitive_scoop, random_state=random_state) if(negitive_scoop != 0) else [[],[]]
    file_list_TH, reject_TH = train_test_split(file_list_TH, test_size=th_scoop, random_state=random_state) if(th_scoop != 0) else [[],[]]
    
    print(f'{len(file_list_pos)} reference_data samples found pos')
    print(f'{len(file_list_neg)} reference_data samples found neg')
    print(f'{len(file_list_TH)} reference_data samples found TH')
    file_list_total = file_list_pos + file_list_neg + file_list_TH
    print(f'{len(file_list_total)} reference_data samples found')
    return file_list_total

def main(config=None):
    start_time = datetime.now()
    current_time = start_time.strftime("%Y-%m-%d_%H:%M:%S")
    print("Current Time =", current_time)

    if config:
        wandb.config = config
    else:
        run = wandb.init()


    # setting hyper parameters
    n_epochs = wandb.config.get(EPOCHS)
    batch_size = wandb.config.get(BATCH_SIZE)
    learning_rate = wandb.config.get(LEARNING_RATE)
    beta = wandb.config.get(BETA)
    loss_function = wandb.config.get(LOSS_FUNCTION)

    loss_function = sweep_loss_funtion if loss_function is None else loss_function
    loss_function_name = str(loss_function).split("'")[1].split(".")[1]

    # loss Function
    criteria = loss_function(beta)
    # criteria = two_branch_loss(beta)
    OUTPUT_ACTIVATION = criteria.last_activation if criteria.last_activation else "relu"
    # Set up the model. and optimizer
    selected_model = Autoencoder(GOES_Bands,OUTPUT_ACTIVATION)
    model_name = type(selected_model).__name__
    optimizer = optim.Adam(list(selected_model.parameters()), lr=learning_rate)

    project_name = project_name_template.format(
    model_name = model_name,
    loss_function_name=loss_function_name,
    n_epochs=n_epochs,
    batch_size=batch_size,
    learning_rate=learning_rate,
    model_specific_postfix=model_specific_postfix
)
    print(project_name)

    mp = model_path  + project_name
    if not os.path.exists(mp):
        os.mkdir(mp)

    # Creating logging
    file_handler = logging.FileHandler(f"{mp}/training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            file_handler,
            logging.StreamHandler()
        ]
    )

    if config:
        run = wandb.init(project=project_name, name="run_" + current_time)
    else:
        run.name = project_name
    run_url = run.get_url()

    print(f"The log file path is: {file_handler.baseFilename}")
    logging.info(f'Starting training at {current_time} \n\t Url : {run_url}')

    print(f'Train with n_epochs : {n_epochs} , batch_size : {batch_size} , learning_rate : {learning_rate}')
    print(f'beta : {beta}, loss function :{loss_function}')
    # Train and save the model components
    train_runner(selected_model, n_epochs, batch_size, criteria, optimizer)

    log_end_process(start_time)
    print(f"The log file path is: {file_handler.baseFilename}")
    save_selected_model(selected_model, mp)
    torch.save(optimizer.state_dict(), mp + "/" + RES_OPT_PTH)
    reset_logging()

def save_selected_model(selected_model, mp):
    torch.save(selected_model.state_dict(), mp + "/" + RES_AUTOENCODER_PTH)

def log_end_process(start_time):
    end =  datetime.now()
    duration = end - start_time
    seconds = duration.total_seconds()
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    logging.info(f'\tTime Taken : {hours} hours {minutes} minutes {seconds} seconds')

def reset_logging():
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Remove all handlers associated with the root logger
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Clear existing handlers
    root_logger.handlers = []


if __name__ == "__main__":
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if SWEEP_OPERATION:
        # Initialize sweep by passing in config. (Optional) Provide a name of the project.
        # # wandb.login()
        from ModelRunConfiguration import sweep_configuration_IOU_LRMSE
        sweep_configuration = sweep_configuration_IOU_LRMSE
        sweep_id = wandb.sweep(sweep=sweep_configuration, project='sweep_config')
        wandb.agent(sweep_id, function=main, count=14)
    else:
        config = use_config
        main(config)

# https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html