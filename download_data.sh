#!/bin/bash

### Download Dogs vs Cats Zip ###
wget https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip -P data/

### Download IMDB Dataset ###
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz -P data/
tar -xf data/aclImdb_v1.tar.gz -C data/

### Download Harry Potter Books ###
mkdir data/harry_potter_txt
wget "https://raw.githubusercontent.com/priyammaz/HAL-DL-From-Scratch/main/data/harry_potter_txt/Book%201%20-%20The%20Philosopher's%20Stone.txt" -P data/harry_potter_txt/
wget "https://raw.githubusercontent.com/priyammaz/HAL-DL-From-Scratch/main/data/harry_potter_txt/Book%202%20-%20The%20Chamber%20of%20Secrets.txt" -P data/harry_potter_txt/
wget "https://raw.githubusercontent.com/priyammaz/HAL-DL-From-Scratch/main/data/harry_potter_txt/Book%203%20-%20The%20Prisoner%20of%20Azkaban.txt" -P data/harry_potter_txt/
wget "https://raw.githubusercontent.com/priyammaz/HAL-DL-From-Scratch/main/data/harry_potter_txt/Book%204%20-%20The%20Goblet%20of%20Fire.txt" -P data/harry_potter_txt/
wget "https://raw.githubusercontent.com/priyammaz/HAL-DL-From-Scratch/main/data/harry_potter_txt/Book%205%20-%20The%20Order%20of%20the%20Phoenix.txt" -P data/harry_potter_txt/
wget "https://raw.githubusercontent.com/priyammaz/HAL-DL-From-Scratch/main/data/harry_potter_txt/Book%206%20-%20The%20Half%20Blood%20Prince.txt" -P data/harry_potter_txt/
wget "https://raw.githubusercontent.com/priyammaz/HAL-DL-From-Scratch/main/data/harry_potter_txt/Book%207%20-%20The%20Deathly%20Hallows.txt" -P data/harry_potter_txt/

### Run Prep Data ###
python -m prep_data -p data --all

### Cleanup ###
rm data/aclImdb_v1.tar.gz
rm data/CDLA-Permissive-2.0.pdf
rm data/kagglecatsanddogs_5340.zip
rm data/readme[1].txt