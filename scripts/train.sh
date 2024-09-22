CHECKPOINT_DIR=/data/DSEC-Flow/STFlow-std && \
CUDA_VISIBLE_DEVICES=0 
python train.py \
--checkpoint_dir ${CHECKPOINT_DIR} \
--num_steps 200000 \
--batch_size 6 \
--lr 2e-4 




