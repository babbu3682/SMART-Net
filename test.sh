#!/bin/sh


# 2D
: <<'END'
CUDA_VISIBLE_DEVICES=0 python -W ignore test.py \
--dataset=coreline_dataset_test \
--test-batch-size=10 \
--test-num_workers=0 \
--model=SMART-Net-2D \
--backbone=resnet-50 \
--roi_size=256 \
--sw_batch_size=32 \
--use_skip=True \
--pool_type=gem \
--use_consist=True \
--multi-gpu=Single \
--resume=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240921-SMART-Net-2D-resnet/weights/epoch_350_checkpoint.pth \
--save-dir=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240921-SMART-Net-2D-resnet \
--memo=None
END

: <<'END'
CUDA_VISIBLE_DEVICES=1 python -W ignore test.py \
--dataset=coreline_dataset_test \
--test-batch-size=10 \
--test-num_workers=0 \
--model=SMART-Net-2D \
--backbone=efficientnet-b7 \
--roi_size=256 \
--sw_batch_size=32 \
--use_skip=True \
--pool_type=gem \
--use_consist=True \
--multi-gpu=Single \
--resume=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240920-SMART-Net-2D-efficient/weights/epoch_293_checkpoint.pth \
--save-dir=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240920-SMART-Net-2D-efficient \
--memo=None
END

: <<'END'
CUDA_VISIBLE_DEVICES=2 python -W ignore test.py \
--dataset=coreline_dataset_test \
--test-batch-size=10 \
--test-num_workers=0 \
--model=SMART-Net-2D \
--backbone=maxvit-small \
--roi_size=256 \
--sw_batch_size=32 \
--use_skip=True \
--pool_type=gem \
--use_consist=True \
--multi-gpu=Single \
--resume=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240922-SMART-Net-2D-maxvit-small/weights/epoch_327_checkpoint.pth \
--save-dir=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240922-SMART-Net-2D-maxvit-small \
--memo=None
END

: <<'END'
CUDA_VISIBLE_DEVICES=3 python -W ignore test.py \
--dataset=coreline_dataset_test \
--test-batch-size=10 \
--test-num_workers=0 \
--model=SMART-Net-2D \
--backbone=maxvit-xlarge \
--roi_size=256 \
--sw_batch_size=32 \
--use_skip=True \
--pool_type=gem \
--use_consist=True \
--multi-gpu=Single \
--resume=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240921-SMART-Net-2D-maxvit-xlarge/weights/epoch_143_checkpoint.pth \
--save-dir=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240921-SMART-Net-2D-maxvit-xlarge \
--memo=None
END






# 3D - 2D transfer
# ResNet
: <<'END'
for e in 4 3 12 6
do
    CUDA_VISIBLE_DEVICES=1 python -W ignore test.py \
    --dataset=coreline_dataset_test_3d_cls \
    --test-batch-size=50 \
    --test-num_workers=0 \
    --model=SMART-Net-3D-CLS \
    --backbone=resnet-50 \
    --roi_size=256 \
    --sw_batch_size=32 \
    --use_skip=True \
    --operator_3d=bert \
    --pool_type=gem \
    --use_consist=False \
    --multi-gpu=Single \
    --resume=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240924-SMART-Net-3D-resnet-TR-freeze-encoder/weights/epoch_${e}_checkpoint.pth \
    --save-dir=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240924-SMART-Net-3D-resnet-TR-freeze-encoder \
    --memo=None
done
END


: <<'END'
for e in 77 200 232 93
do
    CUDA_VISIBLE_DEVICES=1 python -W ignore test.py \
    --dataset=coreline_dataset_test_3d_cls \
    --test-batch-size=50 \
    --test-num_workers=0 \
    --model=SMART-Net-3D-CLS \
    --backbone=resnet-50 \
    --roi_size=256 \
    --sw_batch_size=32 \
    --use_skip=True \
    --operator_3d=lstm \
    --pool_type=gem \
    --use_consist=False \
    --multi-gpu=Single \
    --resume=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240924-SMART-Net-3D-resnet-LSTM-freeze-encoder/weights/epoch_${e}_checkpoint.pth \
    --save-dir=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240924-SMART-Net-3D-resnet-LSTM-freeze-encoder \
    --memo=None
done
END


# EfficientNet
: <<'END'
for e in 10 11 4 65 56 66 8 9 21 12 38 39 20 45 14 36 46
do
    CUDA_VISIBLE_DEVICES=0 python -W ignore test.py \
    --dataset=coreline_dataset_test_3d_cls \
    --test-batch-size=50 \
    --test-num_workers=0 \
    --model=SMART-Net-3D-CLS \
    --backbone=efficientnet-b7 \
    --roi_size=256 \
    --sw_batch_size=32 \
    --use_skip=True \
    --operator_3d=bert \
    --pool_type=gem \
    --use_consist=False \
    --multi-gpu=Single \
    --resume=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240924-SMART-Net-3D-efficientnet-TR-freeze-encoder/weights/epoch_${e}_checkpoint.pth \
    --save-dir=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240924-SMART-Net-3D-efficientnet-TR-freeze-encoder \
    --memo=None
done
END

: <<'END'
for e in 55 44 15 54 31 36 43 53 52 14 30 29 45 35 17 51 28 32
do
    CUDA_VISIBLE_DEVICES=1 python -W ignore test.py \
    --dataset=coreline_dataset_test_3d_cls \
    --test-batch-size=50 \
    --test-num_workers=0 \
    --model=SMART-Net-3D-CLS \
    --backbone=efficientnet-b7 \
    --roi_size=256 \
    --sw_batch_size=32 \
    --use_skip=True \
    --operator_3d=lstm \
    --pool_type=gem \
    --use_consist=False \
    --multi-gpu=Single \
    --resume=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240924-SMART-Net-3D-efficientnet-LSTM-freeze-encoder/weights/epoch_${e}_checkpoint.pth \
    --save-dir=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240924-SMART-Net-3D-efficientnet-LSTM-freeze-encoder \
    --memo=None
done
END


# MaxViT-small
: <<'END'
for e in 36 26 14 31 156 27 30 154 37 24 155 23 22 29 8 184 13
do
    CUDA_VISIBLE_DEVICES=2 python -W ignore test.py \
    --dataset=coreline_dataset_test_3d_cls \
    --test-batch-size=50 \
    --test-num_workers=0 \
    --model=SMART-Net-3D-CLS \
    --backbone=maxvit-small \
    --roi_size=256 \
    --sw_batch_size=32 \
    --use_skip=True \
    --operator_3d=bert \
    --pool_type=gem \
    --use_consist=False \
    --multi-gpu=Single \
    --resume=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240924-SMART-Net-3D-maxvit-small-TR-freeze-encoder/weights/epoch_${e}_checkpoint.pth \
    --save-dir=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240924-SMART-Net-3D-maxvit-small-TR-freeze-encoder \
    --memo=None
done
END

: <<'END'
for e in 54 53 52 56 45 84 106 44 158 156 157 128 101 63 103 241 100
do
    CUDA_VISIBLE_DEVICES=3 python -W ignore test.py \
    --dataset=coreline_dataset_test_3d_cls \
    --test-batch-size=50 \
    --test-num_workers=0 \
    --model=SMART-Net-3D-CLS \
    --backbone=maxvit-small \
    --roi_size=256 \
    --sw_batch_size=32 \
    --use_skip=True \
    --operator_3d=lstm \
    --pool_type=gem \
    --use_consist=False \
    --multi-gpu=Single \
    --resume=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240924-SMART-Net-3D-maxvit-small-LSTM-freeze-encoder/weights/epoch_${e}_checkpoint.pth \
    --save-dir=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240924-SMART-Net-3D-maxvit-small-LSTM-freeze-encoder \
    --memo=None
done
END

# MaxViT-xlarge
: <<'END'
for e in 5 6 9 13 15 14 22 16 21 20 27 18 3 24 2
do
    CUDA_VISIBLE_DEVICES=0 python -W ignore test.py \
    --dataset=coreline_dataset_test_3d_cls \
    --test-batch-size=50 \
    --test-num_workers=0 \
    --model=SMART-Net-3D-CLS \
    --backbone=maxvit-xlarge \
    --roi_size=256 \
    --sw_batch_size=32 \
    --use_skip=True \
    --operator_3d=bert \
    --pool_type=gem \
    --use_consist=False \
    --multi-gpu=Single \
    --resume=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240924-SMART-Net-3D-maxvit-xlarge-TR-freeze-encoder/weights/epoch_${e}_checkpoint.pth \
    --save-dir=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240924-SMART-Net-3D-maxvit-xlarge-TR-freeze-encoder \
    --memo=None
done
END

: <<'END'
for e in 36 21 34 37 38 31 23 24 33 28 29 27 18 17 16 13 30 58
do
    CUDA_VISIBLE_DEVICES=1 python -W ignore test.py \
    --dataset=coreline_dataset_test_3d_cls \
    --test-batch-size=50 \
    --test-num_workers=0 \
    --model=SMART-Net-3D-CLS \
    --backbone=maxvit-xlarge \
    --roi_size=256 \
    --sw_batch_size=32 \
    --use_skip=True \
    --operator_3d=lstm \
    --pool_type=gem \
    --use_consist=False \
    --multi-gpu=Single \
    --resume=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240924-SMART-Net-3D-maxvit-xlarge-LSTM-freeze-encoder/weights/epoch_${e}_checkpoint.pth \
    --save-dir=/workspace/1.Hemorrhage/SMART-Net/checkpoints/20240924-SMART-Net-3D-maxvit-xlarge-LSTM-freeze-encoder \
    --memo=None
done
END




# Temperature
: <<'END'
for t in {2..30..1}
do
    CUDA_VISIBLE_DEVICES=2 python -W ignore test.py \
    --dataset 'coreline_dataset_3d_2dtransfer' \
    --test-batch-size 96 \
    --test-num_workers 48 \
    --model 'MaxViT_LSTM' \
    --resume '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/checkpoints/231229_MaxViT_LSTM/epoch_55_checkpoint.pth' \
    --checkpoint-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/checkpoints/[TEST]231229_MaxViT_LSTM' \
    --save-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/predictions/[TEST]231229_MaxViT_LSTM' \
    --memo 'image 512x512, 231229_MaxViT_LSTM' \
    --epoch 58 \
    --T ${t}
done
END