#!/bin/sh

# 2D
: <<'END'
accelerate launch --config_file /workspace/1.Hemorrhage/SMART-Net/default_config.yaml \
/workspace/1.Hemorrhage/SMART-Net/train.py \
--dataset=coreline_dataset \
--train-batch-size=60 \
--valid-batch-size=60 \
--train-num-workers=5 \
--valid-num-workers=5 \
--model=SMART-Net-2D \
--backbone=resnet-50 \
--roi_size=256 \
--use_skip=True \
--pool_type=gem \
--use_consist=True \
--loss=MTL_Loss \
--optimizer=adamw \
--scheduler=poly_lr \
--epochs=500 \
--lr=1e-4 \
--multi-gpu=DDP \
--save-dir=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240921-SMART-Net-2D-resnet \
--memo=None
END

: <<'END'
accelerate launch --config_file /workspace/1.Hemorrhage/SMART-Net/zero2_config.yaml \
/workspace/1.Hemorrhage/SMART-Net/train.py \
--dataset=coreline_dataset \
--train-batch-size=20 \
--valid-batch-size=10 \
--train-num-workers=2 \
--valid-num-workers=2 \
--model=SMART-Net-2D \
--backbone=efficientnet-b7 \
--roi_size=256 \
--use_skip=True \
--pool_type=gem \
--use_consist=True \
--loss=MTL_Loss \
--optimizer=adamw \
--scheduler=poly_lr \
--epochs=500 \
--lr=1e-4 \
--multi-gpu=DDP \
--save-dir=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240920-SMART-Net-2D-efficient \
--memo=None
END


: <<'END'
accelerate launch --config_file /workspace/1.Hemorrhage/SMART-Net/default_config.yaml \
/workspace/1.Hemorrhage/SMART-Net/train.py \
--dataset=coreline_dataset \
--train-batch-size=28 \
--valid-batch-size=28 \
--train-num-workers=0 \
--valid-num-workers=0 \
--model=SMART-Net-2D \
--backbone=maxvit-small \
--roi_size=256 \
--use_skip=True \
--pool_type=gem \
--use_consist=True \
--loss=MTL_Loss \
--optimizer=adamw \
--scheduler=poly_lr \
--epochs=500 \
--lr=1e-4 \
--multi-gpu=DDP \
--save-dir=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240922-SMART-Net-2D-maxvit-small \
--memo=None
END


: <<'END'
accelerate launch --config_file /workspace/1.Hemorrhage/SMART-Net/default_config.yaml \
/workspace/1.Hemorrhage/SMART-Net/train.py \
--dataset=coreline_dataset \
--train-batch-size=2 \
--valid-batch-size=20 \
--train-num-workers=0 \
--valid-num-workers=0 \
--model=SMART-Net-2D \
--backbone=maxvit-xlarge \
--roi_size=256 \
--use_skip=True \
--pool_type=gem \
--use_consist=True \
--loss=MTL_Loss \
--optimizer=adamw \
--scheduler=poly_lr \
--epochs=500 \
--lr=1e-4 \
--multi-gpu=DDP \
--save-dir=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240921-SMART-Net-2D-maxvit \
--memo=None
END








# 3D - 2D transfer CLS
: <<'END'
accelerate launch --config_file /workspace/1.Hemorrhage/SMART-Net/default_config.yaml \
/workspace/1.Hemorrhage/SMART-Net/train.py \
--dataset=coreline_dataset-3d_cls-2d_transfer \
--train-batch-size=72 \
--valid-batch-size=72 \
--train-num-workers=2 \
--valid-num-workers=2 \
--model=SMART-Net-3D-CLS \
--backbone=maxvit-small \
--roi_size=256 \
--use_skip=True \
--operator_3d=bert \
--loss=CLS_Loss \
--optimizer=adamw \
--scheduler=poly_lr \
--epochs=500 \
--lr=1e-4 \
--multi-gpu=DDP \
--transfer-pretrained=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240922-SMART-Net-2D-maxvit-small/weights/epoch_327_checkpoint.pth \
--use-pretrained-encoder=True \
--freeze-encoder=True \
--save-dir=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240924-SMART-Net-3D-maxvit-small-TR-freeze-encoder \
--memo=None
END

: <<'END'
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --main_process_port 3683 \
--config_file /workspace/1.Hemorrhage/SMART-Net/default_config.yaml \
/workspace/1.Hemorrhage/SMART-Net/train.py \
--dataset=coreline_dataset-3d_cls-2d_transfer \
--train-batch-size=72 \
--valid-batch-size=72 \
--train-num-workers=2 \
--valid-num-workers=2 \
--model=SMART-Net-3D-CLS \
--backbone=maxvit-small \
--roi_size=256 \
--use_skip=True \
--operator_3d=lstm \
--loss=CLS_Loss \
--optimizer=adamw \
--scheduler=poly_lr \
--epochs=500 \
--lr=1e-4 \
--multi-gpu=DDP \
--transfer-pretrained=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240922-SMART-Net-2D-maxvit-small/weights/epoch_327_checkpoint.pth \
--use-pretrained-encoder=True \
--freeze-encoder=True \
--save-dir=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240924-SMART-Net-3D-maxvit-small-LSTM-freeze-encoder \
--memo=None
END

# work
# resnet
: <<'END'
accelerate launch --config_file /workspace/1.Hemorrhage/SMART-Net/default_config.yaml \
/workspace/1.Hemorrhage/SMART-Net/train.py \
--dataset=coreline_dataset-3d_cls-2d_transfer \
--train-batch-size=72 \
--valid-batch-size=72 \
--train-num-workers=2 \
--valid-num-workers=2 \
--model=SMART-Net-3D-CLS \
--backbone=resnet-50 \
--roi_size=256 \
--use_skip=True \
--operator_3d=bert \
--loss=CLS_Loss \
--optimizer=adamw \
--scheduler=poly_lr \
--epochs=500 \
--lr=1e-4 \
--multi-gpu=DDP \
--transfer-pretrained=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240921-SMART-Net-2D-resnet/weights/epoch_415_checkpoint.pth \
--use-pretrained-encoder=True \
--freeze-encoder=True \
--save-dir=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240924-SMART-Net-3D-resnet-TR-freeze-encoder \
--memo=None
END

: <<'END'
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 0 \
--config_file /workspace/1.Hemorrhage/SMART-Net/default_config.yaml \

accelerate launch --config_file /workspace/1.Hemorrhage/SMART-Net/default_config.yaml \
/workspace/1.Hemorrhage/SMART-Net/train.py \
--dataset=coreline_dataset-3d_cls-2d_transfer \
--train-batch-size=72 \
--valid-batch-size=72 \
--train-num-workers=2 \
--valid-num-workers=2 \
--model=SMART-Net-3D-CLS \
--backbone=resnet-50 \
--roi_size=256 \
--use_skip=True \
--operator_3d=lstm \
--loss=CLS_Loss \
--optimizer=adamw \
--scheduler=poly_lr \
--epochs=500 \
--lr=1e-4 \
--multi-gpu=DDP \
--transfer-pretrained=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240921-SMART-Net-2D-resnet/weights/epoch_415_checkpoint.pth \
--use-pretrained-encoder=True \
--freeze-encoder=True \
--save-dir=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240924-SMART-Net-3D-resnet-LSTM-freeze-encoder \
--memo=None
END

# efficientnet
: <<'END'
accelerate launch --config_file /workspace/1.Hemorrhage/SMART-Net/default_config.yaml \
/workspace/1.Hemorrhage/SMART-Net/train.py \
--dataset=coreline_dataset-3d_cls-2d_transfer \
--train-batch-size=72 \
--valid-batch-size=72 \
--train-num-workers=0 \
--valid-num-workers=0 \
--model=SMART-Net-3D-CLS \
--backbone=efficientnet-b7 \
--roi_size=256 \
--use_skip=True \
--operator_3d=bert \
--loss=CLS_Loss \
--optimizer=adamw \
--scheduler=poly_lr \
--epochs=500 \
--lr=1e-4 \
--multi-gpu=DDP \
--transfer-pretrained=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240920-SMART-Net-2D-efficient/weights/epoch_293_checkpoint.pth \
--use-pretrained-encoder=True \
--freeze-encoder=True \
--save-dir=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240924-SMART-Net-3D-efficientnet-TR-freeze-encoder \
--memo=None
END

# work
: <<'END'
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --main_process_port 3683 \
--config_file /workspace/1.Hemorrhage/SMART-Net/default_config.yaml \

accelerate launch --config_file /workspace/1.Hemorrhage/SMART-Net/default_config.yaml \
/workspace/1.Hemorrhage/SMART-Net/train.py \
--dataset=coreline_dataset-3d_cls-2d_transfer \
--train-batch-size=72 \
--valid-batch-size=72 \
--train-num-workers=0 \
--valid-num-workers=0 \
--model=SMART-Net-3D-CLS \
--backbone=efficientnet-b7 \
--roi_size=256 \
--use_skip=True \
--operator_3d=lstm \
--loss=CLS_Loss \
--optimizer=adamw \
--scheduler=poly_lr \
--epochs=500 \
--lr=1e-4 \
--multi-gpu=DDP \
--transfer-pretrained=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240920-SMART-Net-2D-efficient/weights/epoch_293_checkpoint.pth \
--use-pretrained-encoder=True \
--freeze-encoder=True \
--save-dir=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240924-SMART-Net-3D-efficientnet-LSTM-freeze-encoder \
--memo=None
END

# work
# maxvit-xlarge
: <<'END'
accelerate launch --config_file /workspace/1.Hemorrhage/SMART-Net/default_config.yaml \
/workspace/1.Hemorrhage/SMART-Net/train.py \
--dataset=coreline_dataset-3d_cls-2d_transfer \
--train-batch-size=36 \
--valid-batch-size=36 \
--train-num-workers=0 \
--valid-num-workers=0 \
--model=SMART-Net-3D-CLS \
--backbone=maxvit-xlarge \
--roi_size=256 \
--use_skip=True \
--operator_3d=bert \
--loss=CLS_Loss \
--optimizer=adamw \
--scheduler=poly_lr \
--epochs=500 \
--lr=1e-4 \
--multi-gpu=DDP \
--transfer-pretrained=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240921-SMART-Net-2D-maxvit-xlarge/weights/epoch_332_checkpoint.pth \
--use-pretrained-encoder=True \
--freeze-encoder=True \
--save-dir=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240924-SMART-Net-3D-maxvit-xlarge-TR-freeze-encoder \
--memo=None
END

# work
: <<'END'
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --main_process_port 3683 \
--config_file /workspace/1.Hemorrhage/SMART-Net/default_config.yaml \

accelerate launch --config_file /workspace/1.Hemorrhage/SMART-Net/default_config.yaml \
/workspace/1.Hemorrhage/SMART-Net/train.py \
--dataset=coreline_dataset-3d_cls-2d_transfer \
--train-batch-size=72 \
--valid-batch-size=72 \
--train-num-workers=0 \
--valid-num-workers=0 \
--model=SMART-Net-3D-CLS \
--backbone=maxvit-xlarge \
--roi_size=256 \
--use_skip=True \
--operator_3d=lstm \
--loss=CLS_Loss \
--optimizer=adamw \
--scheduler=poly_lr \
--epochs=500 \
--lr=1e-4 \
--multi-gpu=DDP \
--transfer-pretrained=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240921-SMART-Net-2D-maxvit-xlarge/weights/epoch_332_checkpoint.pth \
--use-pretrained-encoder=True \
--freeze-encoder=True \
--save-dir=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240924-SMART-Net-3D-maxvit-xlarge-LSTM-freeze-encoder \
--memo=None
END