import os
import pickle

class LocalLogger:
    def __init__(self, 
                 path_to_log_folder, 
                 filename="train_log.pkl"):
        
        self.path_to_log_folder = path_to_log_folder
        self.path_to_file = os.path.join(path_to_log_folder, filename)

        self.log_exists = os.path.isfile(self.path_to_file)

        if self.log_exists:
            with open(self.path_to_file, "rb") as f:
                self.logger = pickle.load(f)
            
        else:
            self.logger = {"epoch": [], 
                           "train_loss": [], 
                           "train_acc": [], 
                           "val_loss": [], 
                           "val_acc": []}
            
    def log(self, epoch, train_loss, train_acc, test_loss, test_acc):
        self.logger["epoch"].append(epoch)
        self.logger["train_loss"].append(train_loss)
        self.logger["train_acc"].append(train_acc)
        self.logger["val_loss"].append(test_loss)
        self.logger["val_acc"].append(test_acc)

        with open(self.path_to_file, "wb") as f:
            pickle.dump(self.logger, f)