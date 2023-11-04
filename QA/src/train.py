import os
import hydra
import numpy as np
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch

from src.model import InstructionModel
from src.dataset import prepare_dataloaders


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "QA/conf"),
    config_name=os.environ.get("HYDRA_CONFIG_NAME", "config"),
)
def run_training(cfg: DictConfig):
    print(cfg)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # initialize model
    pl.seed_everything(cfg.training_args.seed)
    rank = int(os.environ.get("RANK") or 0)
    model = InstructionModel(
        model_name=cfg.model_name,
        load_in_4_bit=cfg.bnb_config.load_in_4_bit,
        bnb_4bit_use_double_quant=cfg.bnb_config.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=cfg.bnb_config.bnb_4bit_quant_type,
        max_memory=cfg.training_args.max_memory,
        max_length=cfg.training_args.max_length,
        lora_target_modules=cfg.lora_config.lora_target_modules,
        lora_rank=cfg.lora_config.lora_rank,
        lora_alpha=cfg.lora_config.lora_alpha,
        lora_dropout=cfg.lora_config.lora_dropout,
        lora_bias=cfg.lora_config.lora_bias,
        lora_task_type=cfg.lora_config.lora_task_type,
        learning_rate=cfg.training_args.learning_rate,
    )

    # setup hardware
    torch.set_float32_matmul_precision(cfg.training_args.matmul_precision)
    devices = torch.cuda.device_count()
    print(f"{devices} GPUs are available")
    if cfg.training_args.device_num != "auto":
        devices = cfg.training_args.device_num
    print("Finally number of devices", devices)

    if "cuda" in cfg.training_args.device:
        accelerator = "gpu"
        if devices > 1:
            # when you use DDP over N gpu’s, your effective batch_size is (N x batch size).
            # After summing the gradients from each gpu DDP divides the gradients by N.
            # So in order to preserve gradient variance you need to assign learning_rate*sqrt(N)
            # for AdamW
            model.learning_rate = model.learning_rate * np.sqrt(devices)
            strategy = pl.strategies.DDPStrategy(gradient_as_bucket_view=True)
            print(f"Using {strategy} strategy")
        else:
            strategy = "auto"
    else:
        accelerator = "cpu"
        devices = 0
        strategy = "auto"

    print(f"Using {accelerator} accelerator\n")
    print("strategy", strategy)
    print("devices", devices)

    logger = pl.loggers.CSVLogger(save_dir=cfg.training_args.log_path, name="")

    # setup callbacks
    checkfolder = f"{cfg.training_args.save_path}/checkpoints/"
    best_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=cfg.training_args.saving_checkpoint_monitor,  # save by stat value
        mode=cfg.training_args.saving_checkpoint_mode,
        verbose=False,
        dirpath=os.path.join(".", checkfolder),
        filename="BEST-{epoch}-{train_loss:.3f}-{val_loss:.3f}-{f1_macro:.3f}-{f1_weighted:.3f}",
    )
    eoe_callback = pl.callbacks.ModelCheckpoint(
        every_n_epochs=1,  # on the end of each epoch
        save_on_train_epoch_end=True,
        save_top_k=-1,
        verbose=False,
        dirpath=os.path.join(".", checkfolder),
        filename="EOE-{epoch}-{train_loss:.3f}-{val_loss:.3f}-{f1_macro:.3f}-{f1_weighted:.3f}",
    )
    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        best_checkpoint_callback,
        eoe_callback,
    ]

    if cfg.training_args.early_stopper.use_early_stopper:
        patience = cfg.training_args.early_stopper.patience
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor=cfg.training_args.early_stopper.monitor,
            min_delta=cfg.training_args.early_stopper.early_stop_min_delta,
            patience=patience,
            verbose=False,
            mode=cfg.training_args.early_stopper.early_stop_mode,
        )
        callbacks.append(early_stop_callback)

    # setup dataset
    train_loader, test_loader = prepare_dataloaders(
        tokenizer=model.tokenizer,
        max_length=model.max_length,
        dataset_type=cfg.dataset_type,
        batch_size=cfg.training_args.batch_size,
    )

    trainer = pl.Trainer(
        strategy=strategy,
        accelerator=accelerator,
        devices=devices,
        num_nodes=1,
        logger=logger,
        accumulate_grad_batches=cfg.training_args.gradient_accumulation_steps,
        enable_checkpointing=False,  # checkpointing still works with _save_model(); 'False' means lightning will not save automatically
        callbacks=callbacks,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        max_epochs=cfg.training_args.epochs,
        use_distributed_sampler=True,  # same as replace_sampler_ddp = True. Trainer adds DistributedSampler on top of dataloader
        # plugins=[SLURMEnvironment(auto_requeue=False)]
    )
    trainer.fit(model, train_loader, val_dataloaders=test_loader)
    model.save_model(cfg.training_args.save_path)


if __name__ == "__main__":
    run_training()
