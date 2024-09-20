from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
import numpy as np
import torch
import matplotlib.pyplot as plt

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def sensitivity(y_true, y_pred):
    return recall_score(y_true, y_pred)

def f1_metric(y_true, y_pred):
    return f1_score(y_true, y_pred)

def specificity(y_true, y_pred):
    true_negative = np.sum((y_true == 0) & (y_pred == 0))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    return true_negative / (true_negative + false_positive)

def binary_dice_score(y_true, y_pred, smooth=0.0, eps=1e-7):
    # batch size (B)
    bs = y_true.shape[0]
    
    # Reshaping y_true and y_pred to (B, 1, H * W) 형태로 변환
    y_true = y_true.reshape(bs, 1, -1)
    y_pred = y_pred.reshape(bs, 1, -1)

    # Intersection and cardinality calculations
    intersection = np.sum(y_true * y_pred, axis=(0, 2))
    cardinality = np.sum(y_true + y_pred, axis=(0, 2))

    # Dice score calculation
    dice_score = (2. * intersection + smooth) / (np.maximum(cardinality + smooth, eps))
    return np.mean(dice_score)
    


# for Trainer
def compute_mtl_metrics(eval_preds, paths, epoch, step, output_dir):
    gt_cls = eval_preds.inputs['labels']
    gt_seg = eval_preds.inputs['masks']
    gt_img = eval_preds.inputs['images']

    pred_cls = eval_preds.predictions['cls_logits']
    pred_seg = eval_preds.predictions['seg_logits']
    pred_rec = eval_preds.predictions['rec_logits']

    assert gt_cls.shape == pred_cls.shape, f"{gt_cls.shape} != {pred_cls.shape}"
    assert gt_seg.shape == pred_seg.shape, f"{gt_seg.shape} != {pred_seg.shape}"

    # CLS
    auc     = roc_auc_score(y_true=gt_cls, y_score=pred_cls)
    f1      = f1_metric(y_true=gt_cls, y_pred=pred_cls.round())
    acc     = accuracy(y_true=gt_cls, y_pred=pred_cls.round())
    sen     = sensitivity(y_true=gt_cls, y_pred=pred_cls.round())
    spe     = specificity(y_true=gt_cls, y_pred=pred_cls.round())

    # SEG    
    dice_score = binary_dice_score(y_pred=pred_seg.round(), y_true=gt_seg, smooth=0.0).item()   # pred_seg must be round() !!  

    # REC
    mae = np.mean(np.abs(pred_rec - gt_img)).item()

    # SEG
    image_png = gt_img[0].squeeze()
    plt.imsave(output_dir+'/predictions/epoch_'+str(epoch)+'_gt_input.png', image_png, cmap='gray')
    gt_seg_png = gt_seg[0].squeeze()
    plt.imsave(output_dir+'/predictions/epoch_'+str(epoch)+'_gt_mask.png', gt_seg_png, cmap='gray')
    pred_seg_png = pred_seg[0].squeeze()
    plt.imsave(output_dir+'/predictions/epoch_'+str(epoch)+'_pred_seg.png', pred_seg_png.round(), cmap='gray')
    pred_rec_png = pred_rec[0].squeeze()
    plt.imsave(output_dir+'/predictions/epoch_'+str(epoch)+'_pred_rec.png', pred_rec_png, cmap='gray')

    best_score = auc + dice_score
    return {"best_score": best_score, "cls_auc": auc, "cls_f1": f1, "cls_acc": acc, "cls_sen": sen, "cls_spe": spe, "seg_dice": dice_score, "rec_mae": mae}
