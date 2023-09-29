# will pass the arguments into here as a dictionary
# make a class that has the logger, as well as the method of printing i guess
import torch
import numpy as np
import logging

def load_model_from_file(model, folder_path, filename):
    import os
    import torch
    # Add the ".pth" extension to the filename if missing
    if not filename.endswith(".pth"):
        filename += ".pth"

    # getting the filename of the path
    save_path = os.path.join(folder_path, filename)
    # loading it up
    try:
        model.load_state_dict(torch.load(save_path))
    except FileNotFoundError:
        print('file is not found')
    except Exception as e:
        print('a weird error: ', e)

    return model

def save_model_to_file(model, folder_path, filename):
    import os
    import torch
    # Check if the folder exists, and create it if not
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Add the ".pth" extension to the filename if missing
    if not filename.endswith(".pth"):
        filename += ".pth"

    # Save the model to the specified path
    save_path = os.path.join(folder_path, filename)
    torch.save(model.state_dict(), save_path)

def save_dict_to_file(dict, folder_path, filename):
    import json
    import os
    def is_json_serializable(obj):
        try:
            json.dumps(obj)
            return True
        except:
            return False

    def copy_dict_with_serializable_items(original_dict):
        new_dict = {}
        for key, value in original_dict.items():
            if is_json_serializable(value):
                new_dict[key] = value
        return new_dict

    # Check if the folder exists, and create it if not
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Add the ".pth" extension to the filename if missing
    if not filename.endswith(".json"):
        filename += ".json"

    # modify the .json file so that it can be saved to file
    new_dict = copy_dict_with_serializable_items(dict)

    # Save the dictionary to a JSON file
    filename = os.path.join(folder_path, filename)
    with open(filename, "w") as json_file:
        json.dump(new_dict, json_file)

class MyLogger:
    def __init__(self, log_file, log_level=logging.DEBUG):
        # setting up the logging location
        import os
        directory = os.path.dirname(log_file)

        # Check if the directory exists, and create it if it doesn't
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Configure logging settings
        self.log_file = log_file
        self.log_level = log_level
        logger = logging.getLogger()
        fhandler = logging.FileHandler(filename=log_file, mode='a')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fhandler.setFormatter(formatter)
        logger.addHandler(fhandler)
        logger.setLevel(logging.DEBUG)


    def log(self, message, level=logging.INFO):
        if level == logging.DEBUG:
            logging.debug(message)
        elif level == logging.INFO:
            logging.info(message)
        elif level == logging.WARNING:
            logging.warning(message)
        elif level == logging.ERROR:
            logging.error(message)
        elif level == logging.CRITICAL:
            logging.critical(message)
        else:
            raise ValueError("Invalid log level")
        # this then also prints to console as well
        print(message)


# this is the evaluation fucntion, where it trains it for an epoch, and returns the validation accuracy
def eval(model, argDict, givenDataloader):
    import numpy as np
    # setting the device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # forces it to train on cpu real quick
    # device = torch.device('cpu')
    model.to(device)

    eval_accuracy = 0
    eval_loss = 0

    # setting evaluation mode
    with torch.no_grad():
        accuracy_values = []
        loss_values = []
        for idx, (data, label) in enumerate(givenDataloader):
            try:
                data = data.to(device)
                label = label.to(device)

                # getting the predictions
                outputs = model(data)

                # getting the loss as well
                loss = argDict['criterion'](outputs, label)
                loss_values.append((loss.item()))

                # getting the accuracy
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == label).float().mean()
                accuracy_values.append(accuracy)
            except Exception as e:
                # this section logs the whatever
                errString = f'error located at index: {str(idx)} at epoch evalLoop'
                argDict['logger'].log(str(errString))
                argDict['logger'].log(str(e))

        # calculating the accuracy
        eval_loss = np.mean(loss_values)
        eval_accuracy = torch.mean(torch.stack(accuracy_values))

    return eval_accuracy, eval_loss

def test(model, argDict, givenDataloader):
    # this is basically the same as the evaluation one, but it is just given a different name to make things easier i guess. There should be no reason that they are two separate functions
    # setting the device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model.to(device)

    test_accuracy = 0

    # setting evaluation mode
    with torch.no_grad():
        accuracy_values = []
        for idx, (data, label) in enumerate(givenDataloader):
            try:
                data = data.to(device)
                label = label.to(device)

                # getting the predictions
                outputs = model(data)

                # getting the accuracy
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == label).float().mean()
                accuracy_values.append(accuracy)
            except Exception as e:
                # this section logs the whatever
                errString = f'error located at index: {str(idx)} at testing section'
                argDict['logger'].log(str(errString))
                argDict['logger'].log(str(e))


        # calculating the accuracy
        test_accuracy = torch.mean(torch.stack(accuracy_values))

    return test_accuracy

def train(model, argDict, givenDataloader, evalDataloader=None, testDataloader=None):
    # section to check for presence of all the needed loaders
    import time
    start_time = time.time()

    if evalDataloader is None:
        print('you forgot eval loader')
        return
    if testDataloader is None:
        print('you forgot test laoder')
        return

    # get all the stuff out
    # update the learning rate of the optimizer
    for param_group in argDict['optimizer'].param_groups:
        param_group['lr'] = argDict['lr']

    # get the device type, and set it to cuda
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # casting model to device
    model.to(device)

    # training for multiple epochs
    epoch_accuracy_values_train = []
    epoch_accuracy_values_eval = []
    epoch_loss_values_train = []
    epoch_loss_values_eval = []


    best_epoch_value = 0
    best_epoch_epoch = 0

    for currentEpoch in range(argDict['maxEpoch']):
        accuracy_values = []
        loss_values = []

        for idx, (data, label) in enumerate(givenDataloader):
            #  this is captured inside a try because of some weird thing breaking in the middle sometimes when there is only 1 label
            try:
                data, label = data.to(device), label.to(device)

                # this will be the training loop
                outputs = model(data)

                loss = argDict['criterion'](outputs, label)

                # backward pass and optimization
                argDict['optimizer'].zero_grad()
                loss.backward()
                argDict['optimizer'].step()

                # data logging phase, obtains loss and accuracy
                loss_values.append((loss.item()))

                # getting the accuracy
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == label).float().mean()
                accuracy_values.append(accuracy)
            except Exception as e:
                # this section logs the whatever
                errString = f'error located at index: {str(idx)} at epoch {str(currentEpoch)}'
                argDict['logger'].log(str(errString))
                argDict['logger'].log(str(e))

        # calculating epoch losses
        epoch_loss = np.mean(loss_values)
        epoch_loss_values_train.append(epoch_loss)
        epoch_accuracy = torch.mean(torch.stack(accuracy_values))   # due to it being tensor
        epoch_accuracy_values_train.append(epoch_accuracy.item())

        tempString = 'currently at epoch ' + str(currentEpoch) + ' train accuracy: ' + str(epoch_accuracy) + ' loss of: ' + str(epoch_loss)

        # this section is for evaluation of the model on the eval set
        eval_accuracy, eval_loss = eval(model, argDict, evalDataloader)
        epoch_accuracy_values_eval.append(eval_accuracy.item())
        epoch_loss_values_eval.append(eval_loss)

        # log it as well
        tempString = tempString + ' eval accuracy: ' + str(eval_accuracy)
        argDict['logger'].log(tempString)

        # if it improves, no need to break, else, break after reaching max idle epoch
        if eval_accuracy > best_epoch_value:
            best_epoch_value = eval_accuracy
            best_epoch_epoch = currentEpoch
            # save the model as well
            save_model_to_file(model, argDict['outputName'], argDict['outputName'])
        else:
            if (currentEpoch - best_epoch_epoch) > argDict['idleEpoch']:
                # this means that this is the max trained  epoch
                break

    argDict['epoch_loss_values_train'] = epoch_loss_values_train
    argDict['epoch_loss_values_eval'] = epoch_loss_values_eval
    argDict['epoch_accuracy_values_train'] = epoch_accuracy_values_train
    argDict['epoch_accuracy_values_eval'] = epoch_accuracy_values_eval
    argDict['trainingStopEpoch'] = currentEpoch

    # records the time taken for all these
    end_time = time.time()
    elapsed_time = end_time - start_time
    argDict['elapsed_time'] = elapsed_time

    # saves the dictionary as well
    return argDict

def check_folder_exists(folder_name):
    import os

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    else:
        return
    return