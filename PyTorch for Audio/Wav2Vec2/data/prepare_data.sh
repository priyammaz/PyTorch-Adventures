### Change Path to What you want! ###
path_to_store="data/"

train_clean_100="train-clean-100"
train_clean_360="train-clean-360"
train_clean_500="train-clean-500"
dev_clean="dev-clean"
test_clean="test-clean"

wget https://www.openslr.org/resources/12/$train_clean_100.tar.gz -P $path_to_store
wget https://www.openslr.org/resources/12/$train_clean_360.tar.gz -P $path_to_store
wget https://www.openslr.org/resources/12/$train_clean_500.tar.gz -P $path_to_store
wget https://www.openslr.org/resources/12/$dev_clean.tar.gz -P $path_to_store
wget https://www.openslr.org/resources/12/$test_clean.tar.gz -P $path_to_store

tar -xvzf $path_to_store$train_clean_100.tar.gz -C $path_to_store
tar -xvzf $path_to_store$train_clean_360.tar.gz -C $path_to_store
tar -xvzf $path_to_store$train_clean_500.tar.gz -C $path_to_store
tar -xvzf $path_to_store$dev_clean.tar.gz -C $path_to_store
tar -xvzf $path_to_store$test_clean.tar.gz -C $path_to_store