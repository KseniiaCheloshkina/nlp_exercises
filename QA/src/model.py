""" PyTorch Lightning model definition for training"""
from typing import Union, List, Tuple
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import pytorch_lightning as pl
import torch
from torch.optim import AdamW
import torchmetrics
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM


class InstructionModel(pl.LightningModule):
    def __init__(
        self,
        model_name,
        load_in_4_bit: bool = True,
        bnb_4bit_use_double_quant: bool = True,
        bnb_4bit_quant_type: str = "nf4",
        max_memory: str = "15000MB",
        max_length: Union[None, int] = None,
        lora_target_modules: Union[None, List[str]] = None,
        lora_rank: int = 16,
        lora_alpha: int = 64,
        lora_dropout: float = 0.1,
        lora_bias: str = "none",
        lora_task_type: str = "CAUSAL_LM",
        learning_rate: float = 1e-5,
        log_every_steps: int = 50
    ):
        super(InstructionModel, self).__init__()
        self.save_hyperparameters()
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.learning_rate = learning_rate
        self.log_every_steps = log_every_steps

        # BNB config to load model with lower precision
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4_bit,  # True = Load model in 4-bit precision mode
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,  # True = Nested quantization for 4-bit model
            bnb_4bit_quant_type=bnb_4bit_quant_type,  # Quantization data type for 4-bit model
            bnb_4bit_compute_dtype=torch.bfloat16,  # Computation data type for 4-bit model
        )
        # Load model
        n_gpus = torch.cuda.device_count()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",  # dispatch the model efficiently on the available resources
            max_memory={i: max_memory for i in range(n_gpus)},
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        # Set padding token as EOS token because it is not set by default
        #  but we need it because we would like train the model in batches
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # get max length for model
        if max_length is None:
            pot_max_lengths = [
                getattr(model.config, length_setting, None)
                for length_setting in [
                    "n_positions",
                    "max_position_embeddings",
                    "seq_length",
                ]
            ]
            self.max_length = [
                max_length for max_length in pot_max_lengths if max_length is not None
            ][0]
        else:
            self.max_length = max_length
        print(self.max_length)


        # Get linear module names to add LoRa adapters for them
        if lora_target_modules is None:
            lora_target_modules = self.find_all_linear_names(model)
        peft_config = LoraConfig(
            r=lora_rank,  # LoRA attention dimension
            lora_alpha=lora_alpha,  # Alpha parameter for LoRA scaling
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,  # Dropout probability for LoRA layers
            bias=lora_bias,
            task_type=lora_task_type,
        )
        self.model = self.convert_model_to_peft(model, peft_config)
        self.model.config.use_cache = False # do not use cached keys/values as we perform fine-tuning

    @staticmethod
    def convert_model_to_peft(model, peft_config):
        model.gradient_checkpointing_enable()
        # Prepare the model for training: set precision of LM head and LayerNorm to fp32
        model = prepare_model_for_kbit_training(model)
        # convert to PeftModel using config
        model = get_peft_model(model, peft_config)
        # Print information about the percentage of trainable parameters
        InstructionModel.print_trainable_parameters(model)
        return model

    @staticmethod
    def find_all_linear_names(model) -> List[str]:
        """
        Find modules to apply LoRA to.

        :param model: PEFT model
        """
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, bnb.nn.Linear4bit):
                names = name.split(".")
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if "lm_head" in lora_module_names:
            lora_module_names.remove("lm_head")
        print(f"LoRA module names: {list(lora_module_names)}")
        return list(lora_module_names)

    @staticmethod
    def print_trainable_parameters(model, use_4bit=False) -> None:
        """
        Prints the number of trainable parameters in the model.

        :param model: PEFT model
        """

        trainable_params = 0
        all_param = 0

        for _, param in model.named_parameters():
            num_params = param.numel()
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        if use_4bit:
            trainable_params /= 2

        print(
            f"All Parameters: {all_param:,d} || Trainable Parameters: {trainable_params:,d} || Trainable Parameters %: {100 * trainable_params / all_param}"
        )

    def save_model(self, folder: str = "model") -> None:
        torch.save(self.model.state_dict(), folder)

    def load_model(self, saved_folder: str = "model"):
        self.model.load_state_dict(torch.load(saved_folder))

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch["input_ids"], batch["labels"]
        output = self.model.forward(input_ids=x, labels=y)
        loss = output.loss
        self.train_loss.update(loss.detach().cpu())

        if batch_idx % self.log_every_steps == 0:
            rank = self.global_rank
            self.log(f"train/loss_worker_{rank}", loss.item())
        return loss

    @torch.no_grad()
    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x, y = batch["input_ids"], batch["labels"]
        output = self.model.forward(input_ids=x, labels=y)
        self.val_loss.update(output.loss.cpu())
        return output.loss.cpu()

    def on_validation_epoch_end(self) -> None:
        val_loss = self.val_loss.compute()
        self.log("val/loss", val_loss)
        self.val_loss.reset()

    def on_train_epoch_end(self) -> None:
        train_loss = self.train_loss.compute()
        self.log("train/loss_epoch", train_loss)
        self.train_loss.reset()

    def configure_optimizers(self):
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        return {"optimizer": self.optimizer}

    @torch.no_grad()
    def sample(
        self,
        context: str,
        num_return_sequences: int = 5,
        temp: float = 0.2,
        top_p: float = 0.95,
        max_length_sample: int = 512,
        max_length: int = None,
    ):
        if max_length is None:
            max_length = self.max_length
        input_ids = self.tokenizer(
            context,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        ).input_ids

        input_ids_len = input_ids.shape[1]
        assert input_ids_len <= max_length

        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")

        print(f"MODEL IS ON GPU: {next(self.model.parameters()).is_cuda}")
        print(f"INPUT IDS ARE ON GPU: {input_ids.is_cuda}")

        model_gen = self.model
        tokens = model_gen.generate(
            input_ids=input_ids,
            do_sample=True,
            num_beams=num_return_sequences,
            num_return_sequences=num_return_sequences,
            temperature=temp,
            max_length=input_ids_len + max_length_sample,
            top_p=top_p,
            use_cache=True,
            no_repeat_ngram_size=3,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        output = self.tokenizer.batch_decode(tokens[:, input_ids_len:, ...])

        eos_idx = [
            out.find(self.tokenizer.eos_token)
            if out.find(self.tokenizer.eos_token) != -1
            else None
            for out in output
        ]
        output = [out[:eos_id] for out, eos_id in zip(output, eos_idx)]
        output = self.remove_special_tokens(output)

        return output
