import functools
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import FSDPStrategy
from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from pytorch_lightning.strategies import DeepSpeedStrategy, DDPStrategy

from transformers.models.llama.modeling_llama import LlamaDecoderLayer

def get_trainer(config, callbacks=None):

    if config.mode == 'eval':
        trainer = pl.Trainer(
        accelerator=config.device,
        devices=config.world_size,
        strategy=config.trainer.strategy,
        max_epochs=1, # config.experiment.epoch,
        precision=config.trainer.precision,
        callbacks=[pl.callbacks.ModelSummary(max_depth=2)],
        )
        return trainer

    tb_logger = pl.loggers.TensorBoardLogger(save_dir = config.save_path, name = config.exp_name)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=tb_logger.log_dir,
        filename="{step:06d}",
        save_top_k=-1, # save all ckpt corresponding to saving interval              
        every_n_train_steps=config.trainer.save_steps, 
        # save_last=True
    )

    base_callbacks = [
        pl.callbacks.ModelSummary(max_depth=2),
        pl.callbacks.ModelCheckpoint(
        dirpath=tb_logger.log_dir,
        filename="{step:06d}",
        save_top_k=-1, # save all ckpt corresponding to saving interval              
        every_n_train_steps=config.trainer.save_steps, 
        save_last=True
        )
    ]

    if callbacks:
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        base_callbacks.extend(callbacks)

    precision = config.trainer.precision

    strategy = get_strategy(config)

    if config.trainer.strategy == 'fsdp':
        trainer = pl.Trainer(
            devices=config.world_size,
            accelerator=config.device,
            logger=tb_logger,
            default_root_dir=config.save_path,
            callbacks=base_callbacks, 
            strategy=strategy,
            log_every_n_steps=config.trainer.log_steps,
            gradient_clip_val=None,
            enable_checkpointing=config.trainer.enable_checkpointing,
            accumulate_grad_batches=config.trainer.gradient_accumulation_steps,
            precision=precision, 
            max_steps=config.trainer.max_training_steps, # or max_epochs   
            check_val_every_n_epoch=None, # no validation
            val_check_interval=config.trainer.val_steps * config.trainer.gradient_accumulation_steps, 
            num_sanity_val_steps=1,
        )
    else:
        trainer = pl.Trainer(
            devices=config.world_size,
            accelerator=config.device,
            logger=tb_logger,
            default_root_dir=config.save_path,
            callbacks=base_callbacks, 
            strategy=strategy,
            log_every_n_steps=config.trainer.log_steps,
            gradient_clip_val=config.trainer.gradient_clip_val, 
            enable_checkpointing=config.trainer.enable_checkpointing,
            accumulate_grad_batches=config.trainer.gradient_accumulation_steps,
            precision=precision, 
            max_steps=config.trainer.max_training_steps, # or max_epochs   
            check_val_every_n_epoch=None, # no validation
            val_check_interval=config.trainer.val_steps * config.trainer.gradient_accumulation_steps, 
            num_sanity_val_steps=0,
        )

    return trainer
        
def get_strategy(config):
    if config.trainer.strategy == 'fsdp':
        my_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={LlamaDecoderLayer},
        )

        strategy = FSDPStrategy(
            auto_wrap_policy=my_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            use_orig_params=True,
            limit_all_gathers=True,
        )

    elif config.trainer.strategy == 'ddp':
        strategy=DDPStrategy(                                          
            find_unused_parameters=True # allow unused
        )
    elif config.trainer.strategy == 'deepspeed':
        strategy=DeepSpeedStrategy(
            **config.trainer.deepspeed_config
        )
    else:
        raise ValueError(f"Unknown strategy: {config.trainer.strategy}")

    return strategy
    
