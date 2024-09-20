import os
import argparse
import datetime
import numpy as np
import time
import torch
import json
import random
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import transformers

import utils
from dataset import get_dataset
from models import get_model

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from arch.trainer import MyTrainer
from metrics import compute_mtl_metrics
import logging

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class ModelArguments:
    model: Optional[str] = field(default="microsoft/Phi-3-mini-4k-instruct", metadata={"help": "Path to the LLM or MLLM."})
    transfer_pretrained: Optional[str] = field(default=None, metadata={"help": "loss name"})
    use_pretrained_encoder: Optional[str] = field(default=None, metadata={"help": "loss name"})
    use_pretrained_decoder: Optional[str] = field(default=None, metadata={"help": "loss name"})
    freeze_encoder: Optional[bool] = field(default=False, metadata={"help": "loss name"})
    freeze_decoder: Optional[bool] = field(default=False, metadata={"help": "loss name"})
    from_pretrained: Optional[str] = field(default=None, metadata={"help": "loss name"})
    resume: Optional[str] = field(default=None, metadata={"help": "loss name"})
    sw_batch_size: Optional[int] = field(default=1, metadata={"help": "loss name"})
    backbone: Optional[str] = field(default="resnet-50", metadata={"help": "loss name"})
    use_skip: Optional[bool] = field(default=True, metadata={"help": "loss name"})
    use_consist: Optional[bool] = field(default=True, metadata={"help": "loss name"})
    pool_type: Optional[str] = field(default="gem", metadata={"help": "loss name"})
    operator_3d: Optional[str] = field(default="LSTM", metadata={"help": "loss name"})

@dataclass
class DataArguments:
    dataset: Optional[str] = field(default="hemorrhage_dataset", metadata={"help": "dataset name"})
    roi_size: Optional[int] = field(default=256, metadata={"help": "loss name"})
    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: Optional[str] = field(default="microsoft/Phi-3-mini-4k-instruct", metadata={"help": "Directory to save model checkpoints and outputs."})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Directory to store the pre-trained models."})
    overwrite_output_dir: Optional[bool] = field(default=True, metadata={"help": "Whether to overwrite the output directory if it exists."})
    eval_strategy: Optional[str] = field(default="epoch", metadata={"help": "Evaluation strategy to use: 'no', 'steps', or 'epoch'."})
    eval_steps: Optional[int] = field(default=1, metadata={"help": "Number of update steps between evaluations."})
    per_device_train_batch_size: Optional[int] = field(default=1, metadata={"help": "Batch size per GPU/TPU core for training."})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "Batch size per GPU/TPU core for evaluation."})
    gradient_accumulation_steps: Optional[int] = field(default=1, metadata={"help": "Number of gradient accumulation steps before updating the model."})
    eval_accumulation_steps: Optional[int] = field(default=1, metadata={"help": "Number of accumulation steps for evaluation."})
    learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "Initial learning rate."})
    weight_decay: Optional[float] = field(default=0.0, metadata={"help": "Weight decay for regularization."})
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "Total number of training epochs."})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "If set, overrides num_train_epochs to stop training after a set number of steps."})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "Type of learning rate scheduler."})
    warmup_ratio: Optional[float] = field(default=0.03, metadata={"help": "Ratio of steps for the learning rate warmup."})
    warmup_steps: Optional[int] = field(default=0, metadata={"help": "Number of steps for the learning rate warmup."})
    log_level: Optional[str] = field(default="debug", metadata={"help": "Logging level (e.g., 'info', 'warning', 'error')."})
    logging_dir: Optional[str] = field(default="microsoft/Phi-3-mini-4k-instruct", metadata={"help": "Directory to save the logs."})
    logging_strategy: Optional[str] = field(default="steps", metadata={"help": "Logging strategy to use: 'steps' or 'epoch'."})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "Number of steps between logging events."})
    save_strategy: Optional[str] = field(default="epoch", metadata={"help": "Saving strategy to use: 'steps' or 'epoch'."})
    save_steps: Optional[int] = field(default=1, metadata={"help": "Number of update steps between saves."})
    save_total_limit: Optional[int] = field(default=3, metadata={"help": "Limit the total number of checkpoints to save."})
    bf16: Optional[bool] = field(default=False, metadata={"help": "Whether to use bf16 (16-bit floating point precision) during training."})
    fp16: Optional[bool] = field(default=True, metadata={"help": "Whether to use bf16 (16-bit floating point precision) during training."})    
    dataloader_pin_memory: Optional[bool] = field(default=True, metadata={"help": "Whether to pin memory in data loaders."})
    dataloader_num_workers: Optional[int] = field(default=0, metadata={"help": "Number of subprocesses to use for data loading."})
    remove_unused_columns: Optional[bool] = field(default=False, metadata={"help": "Remove columns that are unused by the model."})
    load_best_model_at_end: Optional[bool] = field(default=True, metadata={"help": "Whether to load the best model at the end of training."})
    report_to: Optional[str] = field(default="tensorboard", metadata={"help": "Where to report the training progress (e.g., 'tensorboard')."})
    metric_for_best_model: Optional[str] = field(default="best_score", metadata={"help": "The metric to use to compare models for 'best model' selection."})
    greater_is_better: Optional[bool] = field(default=True, metadata={"help": "Whether a higher metric value indicates a better model."})
    gradient_checkpointing: Optional[bool] = field(default=False, metadata={"help": "Enable gradient checkpointing to save memory."})
    resume_from_checkpoint: Optional[str] = field(default=None, metadata={"help": "Path to a checkpoint to resume training from."})
    eval_on_start: Optional[bool] = field(default=False, metadata={"help": "Whether to evaluate the model at the beginning of training."})
    ddp_backend: Optional[str] = field(default="nccl", metadata={"help": "Backend to use for distributed data parallel (DDP) training."})
    ddp_find_unused_parameters: Optional[bool] = field(default=True, metadata={"help": "Whether to find unused parameters during DDP training."})
    optim: Optional[str] = field(default="adamw_torch", metadata={"help": "Optimizer to use during training (e.g., 'adamw_torch')."})
    eval_on_start: Optional[bool] = field(default=False, metadata={"help": "Whether to evaluate the model at the beginning of training."})
    include_inputs_for_metrics: Optional[bool] = field(default=True, metadata={"help": "Whether to include the inputs in the evaluation metrics."})
    
logger = logging.getLogger(__name__)

# fix random seeds for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
np.random.seed(random_seed)
random.seed(random_seed)
import torch


# MAIN
def main():
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    rank0_print("="*20 + " STRAT Training " + "="*20)

    if local_rank == 0:
        utils.print_args(model_args, data_args, training_args)

        # Make folder if not exist
        os.makedirs(training_args.output_dir, exist_ok =True)
        os.makedirs(training_args.output_dir + "/args", exist_ok =True)
        os.makedirs(training_args.output_dir + "/weights", exist_ok =True)
        os.makedirs(training_args.output_dir + "/predictions", exist_ok =True)

        # Save args to json
        the_time = datetime.datetime.now().strftime("%y%m%d_%H%M")
        if not os.path.isfile(training_args.output_dir + "/args/args_" + the_time + ".json"):
            with open(training_args.output_dir + "/args/args_" + the_time + ".json", "w") as f:
                json.dump(training_args.to_json_string(), f, indent=2)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Dataset
    train_dataset, train_collator = get_dataset(name=data_args.dataset, mode='train', roi_size=data_args.roi_size, operator_3d=model_args.operator_3d)
    valid_dataset, valid_collator = get_dataset(name=data_args.dataset, mode='valid', roi_size=data_args.roi_size, operator_3d=model_args.operator_3d)

    # Model
    model = get_model(model_args)

    # Pretrained
    if model_args.from_pretrained is not None:
        print("Loading... Pretrained")
        checkpoint = torch.load(model_args.from_pretrained)
        model.load_state_dict(checkpoint['model_state_dict']) 

    # Trainer
    rank0_print("="*20 + " Training " + "="*20)
    trainer = MyTrainer(model=model,
                        args=training_args,
                        train_data_collator=train_collator,
                        eval_data_collator=valid_collator,
                        train_dataset=train_dataset,
                        eval_dataset=valid_dataset,
                        compute_metrics=compute_mtl_metrics
                        )

    # Etc traing setting
    print(f"Start training")
    start_time = time.time()

    if model_args.resume is not None:
        train_result = trainer.train(resume_from_checkpoint=model_args.resume)
    else:
        train_result = trainer.train()

    # Save model
    rank0_print("="*20 + " Save model " + "="*20)    
    torch.cuda.synchronize()
    trainer.save_model(training_args.output_dir + '/weights/best_checkpoint.bin')
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
                                      

        # # Save checkpoint
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model.module.state_dict() if args.multi_gpu_mode == 'DataParallel' else model.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        #     'scheduler': scheduler.state_dict(),
        # }, args.output_dir + '/weights/epoch_' + str(epoch) + '_checkpoint.pth')

        # # Log text
        # log_stats = {**{f'{k}': v for k, v in train_stats.items()}, 
        #             **{f'{k}': v for k, v in valid_stats.items()}, 
        #             'epoch': epoch,
        #             'lr': optimizer.param_groups[0]['lr']}

        # with open(args.output_dir + "/log.txt", "a") as f:
        #     f.write(json.dumps(log_stats) + "\n")

    # Finish
    total_time_str = str(datetime.timedelta(seconds=int(time.time()-start_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':       
    main()