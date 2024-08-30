import os
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def combine_files(path_to_data, 
                  test_size=0.05):
    
    all_data = []

    paths_to_raw_data = [file for file in os.listdir(path_to_data) if ".txt" in file]
    for file in tqdm(paths_to_raw_data):
        
        ### For .txt Files only ###
        if ".txt" in file:

            ### Grab Language Name ###
            language_name = file.split(".")[0].replace("english2", "").title()

            ### Load English 2 Language Data ###
            path_to_language = os.path.join(path_to_data, file)
            df = pd.read_csv(path_to_language, sep='\t')
            
            ### Grab English Column and Target Language ###
            english = df.iloc[:, 0]
            target = df.iloc[:, 1]

            ### Remove final punctuation on english and targets ###
            english = [e.strip() for e in english]
            target = [t.strip() for t in target]

            ### Store data and append to all_data ###
            cleaned_data = pd.DataFrame({"english": english, "target": target})
            cleaned_data["target_language"] = language_name
            all_data.append(cleaned_data)

    ### Concatenate Data ###
    all_data = pd.concat(all_data)

    ### Split into Training and Testing ###
    training_data, testing_data = train_test_split(all_data, test_size=test_size)

    ### Save Final Dataset ###
    path_to_save = os.path.join(path_to_data, "english2mutilingual_train.csv")
    training_data.to_csv(path_to_save, index=False)
    print(f"Saved Training Data to: {path_to_save}")
    print(f"Total Number of Training Samples: {len(training_data)}")
    print(f"Number of Target Languages: {len(training_data["target_language"].unique())}")

    path_to_save = os.path.join(path_to_data, "english2mutilingual_test.csv")
    testing_data.to_csv(path_to_save, index=False)
    print(f"Saved Testing Data to: {path_to_save}")
    print(f"Total Number of Training Samples: {len(testing_data)}")
    print(f"Number of Target Languages: {len(testing_data["target_language"].unique())}")

if __name__ == "__main__":
    path_to_data = "/mnt/datadrive/data/machine_translation"
    combine_files(path_to_data=path_to_data)