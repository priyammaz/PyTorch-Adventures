python train.py \
    --dataset mnist \
    --batch_size 32 \
    --epochs 100 \
    --lr "0.00025" \
    --device cuda:0 \
    --checkpoint_dir work_dir/mnist_chkpts \
    --gens_dir work_dir/mnist_gens \
    --bf16
