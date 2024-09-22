CHECKPOINT_DIR=/data/DSEC-Flow/pretrain && \


python gen_upload.py \
--ckpt_path ${CHECKPOINT_DIR}/STFlow-std.pth \
--test_path '/data/DSEC-Flow/DSEC_Event_v0_15bins/' \
--test_save ${CHECKPOINT_DIR}/upload




