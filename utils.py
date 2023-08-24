# will pass the arguments into here as a dictionary
# make a class that has the logger, as well as the method of printing i guess

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

import logging

class MyLogger:
    def __init__(self, log_file, log_level=logging.DEBUG):
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


def train(self, inputs):
    print('define the self')

def test(self, inputs):
    print('define the self')

def eval(self, inputs):
    print('define the self')