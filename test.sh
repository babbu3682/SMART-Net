#!/bin/sh

# 3D - 2D transfer
: <<'END'
CUDA_VISIBLE_DEVICES=2 python -W ignore test.py \
--dataset 'coreline_dataset_3d_2dtransfer' \
--test-batch-size 96 \
--test-num_workers 48 \
--model 'EfficientNetB7_LSTM' \
--resume '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/checkpoints/231117_EfficientNetB7_LSTM/epoch_82_checkpoint.pth' \
--checkpoint-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/checkpoints/[TEST]231117_EfficientNetB7_LSTM' \
--save-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/predictions/[TEST]231117_EfficientNetB7_LSTM' \
--memo 'image 512x512, 231117_EfficientNetB7_LSTM' \
--epoch 82
END


: <<'END'
CUDA_VISIBLE_DEVICES=2 python -W ignore test.py \
--dataset 'coreline_dataset_3d_2dtransfer' \
--test-batch-size 96 \
--test-num_workers 48 \
--model 'MaxViT_LSTM' \
--resume '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/checkpoints/231229_MaxViT_LSTM_epoch_55_checkpoint.pth' \
--checkpoint-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/checkpoints/[TEST]231229_MaxViT_LSTM' \
--save-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/predictions/[TEST]231229_MaxViT_LSTM' \
--memo 'image 512x512, 231229_MaxViT_LSTM' \
--epoch 55
END


: <<'END'
CUDA_VISIBLE_DEVICES=1 python -W ignore test.py \
--dataset 'coreline_dataset_3d_2dtransfer' \
--test-batch-size 96 \
--test-num_workers 48 \
--model 'EfficientNetB7_LSTM' \
--resume '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/checkpoints/240105_EfficientNetB7_LSTM/epoch_28_checkpoint.pth' \
--checkpoint-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/checkpoints/[TEST]240105_EfficientNetB7_LSTM' \
--save-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/predictions/[TEST]240105_EfficientNetB7_LSTM' \
--memo 'image 512x512, 240105_EfficientNetB7_LSTM' \
--epoch 28
END

: <<'END'
CUDA_VISIBLE_DEVICES=1 python -W ignore test.py \
--dataset 'coreline_dataset_3d_2dtransfer' \
--test-batch-size 96 \
--test-num_workers 48 \
--model 'EfficientNetB7_LSTM' \
--resume '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/checkpoints/240105_EfficientNetB7_LSTM/epoch_47_checkpoint.pth' \
--checkpoint-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/checkpoints/[TEST]240105_EfficientNetB7_LSTM' \
--save-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/predictions/[TEST]240105_EfficientNetB7_LSTM' \
--memo 'image 512x512, 240105_EfficientNetB7_LSTM' \
--epoch 47
END



: <<'END'
CUDA_VISIBLE_DEVICES=7 python -W ignore test.py \
--dataset 'coreline_dataset_3d_2dtransfer' \
--test-batch-size 96 \
--test-num_workers 48 \
--model 'EfficientNetB7_LSTM' \
--resume '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/checkpoints/240106_EfficientNetB7_LSTM_only_BCE/epoch_34_checkpoint.pth' \
--checkpoint-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/checkpoints/[TEST]240106_EfficientNetB7_LSTM_only_BCE' \
--save-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/predictions/[TEST]240106_EfficientNetB7_LSTM_only_BCE' \
--memo 'image 512x512, 240106_EfficientNetB7_LSTM_only_BCE' \
--epoch 34
END



: <<'END'
CUDA_VISIBLE_DEVICES=7 python -W ignore test.py \
--dataset 'coreline_dataset_3d_2dtransfer' \
--test-batch-size 96 \
--test-num_workers 48 \
--model 'EfficientNetB7_LSTM' \
--resume '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/checkpoints/240107_EfficientNetB7_LSTM_only_BCE/epoch_45_checkpoint.pth' \
--checkpoint-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/checkpoints/[TEST]240107_EfficientNetB7_LSTM_only_BCE' \
--save-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/predictions/[TEST]240107_EfficientNetB7_LSTM_only_BCE' \
--memo 'image 512x512, 240107_EfficientNetB7_LSTM_only_BCE' \
--epoch 45
END


: <<'END'
CUDA_VISIBLE_DEVICES=2 python -W ignore test.py \
--dataset 'coreline_dataset_3d_2dtransfer' \
--test-batch-size 96 \
--test-num_workers 48 \
--model 'EfficientNetB7_LSTM' \
--resume '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/checkpoints/240107_EfficientNetB7_LSTM_only_BCE/epoch_81_checkpoint.pth' \
--checkpoint-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/checkpoints/[TEST]240107_EfficientNetB7_LSTM_only_BCE' \
--save-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/predictions/[TEST]240107_EfficientNetB7_LSTM_only_BCE' \
--memo 'image 512x512, 240107_EfficientNetB7_LSTM_only_BCE' \
--epoch 81
END


: <<'END'
CUDA_VISIBLE_DEVICES=0 python -W ignore test.py \
--dataset 'coreline_dataset_3d_2dtransfer' \
--test-batch-size 96 \
--test-num_workers 48 \
--model 'EfficientNetB7_LSTM' \
--resume '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/checkpoints/240111_EfficientNetB7_LSTM_OnlyBCE_RandomAug/epoch_59_checkpoint.pth' \
--checkpoint-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/checkpoints/[TEST]240111_EfficientNetB7_LSTM_OnlyBCE_RandomAug' \
--save-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/predictions/[TEST]240111_EfficientNetB7_LSTM_OnlyBCE_RandomAug' \
--memo 'image 512x512, 240107_EfficientNetB7_LSTM_only_BCE' \
--epoch 59
END

: <<'END'
CUDA_VISIBLE_DEVICES=0 python -W ignore test.py \
--dataset 'coreline_dataset_3d_2dtransfer' \
--test-batch-size 96 \
--test-num_workers 48 \
--model 'EfficientNetB7_LSTM' \
--resume '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/checkpoints/240111_EfficientNetB7_LSTM_OnlyBCE_RandomAug/epoch_35_checkpoint.pth' \
--checkpoint-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/checkpoints/[TEST]240111_EfficientNetB7_LSTM_OnlyBCE_RandomAug' \
--save-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/predictions/[TEST]240111_EfficientNetB7_LSTM_OnlyBCE_RandomAug' \
--memo 'image 512x512, 240107_EfficientNetB7_LSTM_only_BCE' \
--epoch 35
END


: <<'END'
CUDA_VISIBLE_DEVICES=1 python -W ignore test.py \
--dataset 'coreline_dataset_3d_2dtransfer' \
--test-batch-size 96 \
--test-num_workers 48 \
--model 'MaxViT_LSTM' \
--resume '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/checkpoints/240111_MaxViT_LSTM_LSTM_OnlyBCE_RandomAug/epoch_12_checkpoint.pth' \
--checkpoint-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/checkpoints/[TEST]240111_MaxViT_LSTM_LSTM_OnlyBCE_RandomAug' \
--save-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/predictions/[TEST]240111_MaxViT_LSTM_LSTM_OnlyBCE_RandomAug' \
--memo 'image 512x512, 240111_MaxViT_LSTM_LSTM_OnlyBCE_RandomAug' \
--epoch 12
END

: <<'END'
CUDA_VISIBLE_DEVICES=1 python -W ignore test.py \
--dataset 'coreline_dataset_3d_2dtransfer' \
--test-batch-size 96 \
--test-num_workers 48 \
--model 'MaxViT_LSTM' \
--resume '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/checkpoints/240111_MaxViT_LSTM_LSTM_OnlyBCE_RandomAug/epoch_9_checkpoint.pth' \
--checkpoint-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/checkpoints/[TEST]240111_MaxViT_LSTM_LSTM_OnlyBCE_RandomAug' \
--save-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/predictions/[TEST]240111_MaxViT_LSTM_LSTM_OnlyBCE_RandomAug' \
--memo 'image 512x512, 240111_MaxViT_LSTM_LSTM_OnlyBCE_RandomAug' \
--epoch 9
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