#!/bin/sh

# 2D
: <<'END'
CUDA_VISIBLE_DEVICES=0,1 python -W ignore train.py \
--dataset=coreline_dataset \
--train-batch-size=2 \
--valid-batch-size=2 \
--train-num-workers=10 \
--valid-num-workers=10 \
--model=SMART-Net-2D \
--backbone=efficientnet-b7 \
--roi_size=256 \
--use_skip=True \
--pool_type=gem \
--use_consist=True \
--loss=MTL_Loss \
--optimizer=adamw \
--scheduler=poly_lr \
--epochs=1000 \
--lr=1e-4 \
--multi-gpu-mode=DataParallel \
--save-dir=/workspace/1.Hemorrhage/SMART-Net-Last/checkpoints/20240910-SMART-Net-2D-efficient \
--memo=None
END




# 3D - 2D transfer
: <<'END'
CUDA_VISIBLE_DEVICES=0,1 python -W ignore train.py \
--dataset 'coreline_dataset' \
--train-batch-size 2 \
--valid-batch-size 2 \
--train-num-workers 10 \
--valid-num-workers 10 \
--model 'SMART-Net-3D-CLS' \
--backbone 'efficientnet-b7' \
--roi_size 256 \
--use_skip True \
--pool_type 'gem' \
--operator_3d 'LSTM' \
--use_consist True \
--loss 'CLS_Loss' \
--optimizer 'adamw' \
--scheduler 'poly_lr' \
--epochs 1000 \
--lr 1e-4 \
--multi-gpu-mode 'DataParallel' \
--encoder-pretrained '/workspace/1.Hemorrhage/SMART-Net-Last/checkpoints/20240910-SMART-Net-2D-efficient/weights/epoch_3_checkpoint.pth' \
--save-dir '/workspace/1.Hemorrhage/SMART-Net-Last/checkpoints/20240910-SMART-Net-3D-efficient' \
--memo 'None'
END

: <<'END'
CUDA_VISIBLE_DEVICES=6,7 python -W ignore train.py \
--dataset 'coreline_dataset_3d_2dtransfer_crop' \
--train-batch-size 96 \
--valid-batch-size=96 \
--train-num_workers 48 \
--valid-num_workers 48 \
--model 'EfficientNetB7_LSTM' \
--loss 'CLS_Loss' \
--optimizer 'adamw' \
--scheduler 'poly_lr' \
--epochs 500 \
--lr 1e-4 \
--multi-gpu-mode 'DataParallel' \
--checkpoint-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/checkpoints/240106_EfficientNetB7_LSTM_only_BCE' \
--save-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/predictions/240106_EfficientNetB7_LSTM_only_BCE' \
--memo 'image 320x320 train and 512x512, 240106_EfficientNetB7_LSTM_only_BCE, Random Aug like V1'
END


: <<'END'
CUDA_VISIBLE_DEVICES=6,7 python -W ignore train.py \
--dataset 'coreline_dataset_3d_2dtransfer' \
--train-batch-size 96 \
--valid-batch-size 96 \
--train-num_workers 48 \
--valid-num_workers 48 \
--model 'EfficientNetB7_LSTM' \
--loss 'CLS_Loss' \
--optimizer 'adamw' \
--scheduler 'poly_lr' \
--epochs 500 \
--lr 1e-4 \
--multi-gpu-mode 'DataParallel' \
--checkpoint-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/checkpoints/240107_EfficientNetB7_LSTM_only_BCE' \
--save-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/predictions/240107_EfficientNetB7_LSTM_only_BCE' \
--memo 'image 320x320 train and 512x512, 240107_EfficientNetB7_LSTM_only_BCE, No Random'
END


: <<'END'
CUDA_VISIBLE_DEVICES=2,3 python -W ignore train.py \
--dataset 'coreline_dataset_3d_2dtransfer' \
--train-batch-size 96 \
--valid-batch-size 96 \
--train-num_workers 48 \
--valid-num_workers 48 \
--model 'MaxViT_LSTM' \
--loss 'CLS_Loss' \
--optimizer 'adamw' \
--scheduler 'poly_lr' \
--epochs 500 \
--lr 1e-4 \
--multi-gpu-mode 'DataParallel' \
--checkpoint-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/checkpoints/231229_MaxViT_LSTM' \
--save-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/predictions/231229_MaxViT_LSTM' \
--memo 'image 512x512, 231229_MaxViT_LSTM'
END


: <<'END'
CUDA_VISIBLE_DEVICES=6,7 python -W ignore train.py \
--dataset 'coreline_dataset_3d_2dtransfer_crop' \
--train-batch-size 96 \
--valid-batch-size 96 \
--train-num_workers 48 \
--valid-num_workers 48 \
--model 'EfficientNetB7_LSTM' \
--loss 'CLS_Loss' \
--optimizer 'adamw' \
--scheduler 'poly_lr' \
--epochs 500 \
--lr 1e-4 \
--multi-gpu-mode 'DataParallel' \
--checkpoint-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/checkpoints/240111_EfficientNetB7_LSTM_OnlyBCE_RandomAug' \
--save-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/predictions/240111_EfficientNetB7_LSTM_OnlyBCE_RandomAug' \
--memo 'image 320x320 train and 512x512, 240111_EfficientNetB7_LSTM_OnlyBCE_RandomAug, Random Aug like V1'
END



: <<'END'
CUDA_VISIBLE_DEVICES=0,1 python -W ignore train.py \
--dataset 'coreline_dataset_3d_2dtransfer_crop' \
--train-batch-size 96 \
--valid-batch-size 96 \
--train-num_workers 48 \
--valid-num_workers 48 \
--model 'MaxViT_LSTM' \
--loss 'CLS_Loss' \
--optimizer 'adamw' \
--scheduler 'poly_lr' \
--epochs 500 \
--lr 1e-4 \
--multi-gpu-mode 'DataParallel' \
--checkpoint-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/checkpoints/240111_MaxViT_LSTM_LSTM_OnlyBCE_RandomAug' \
--save-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net-Last/predictions/240111_MaxViT_LSTM_LSTM_OnlyBCE_RandomAug' \
--memo 'image 320x320 train and 512x512, 240111_MaxViT_LSTM_LSTM_OnlyBCE_RandomAug, Random Aug like V1'
END


    # 2D: encoder with MTL
    if args.model_name == 'SMART-Net-2D-w/ResNet-50':
        model = SMART_Net_2D(backbone=args.backbone, use_skip=args.use_skip, pool_type=args.pool_type)

    elif args.model_name == 'SMART-Net-2D-w/EfficientNet-B7':
        model = SMART_Net_2D(backbone='efficientnet-b7', use_skip=True, pool_type='gem')

    elif args.model_name == 'SMART-Net-2D-w/MaxViT-XLarge':
        model = SMART_Net_2D(backbone='maxvit-xlarge', use_skip=True, pool_type='gem')