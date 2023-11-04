from datasets import load_dataset
import torch
from torch.utils.data import Dataset, Dataloader
from typing import Tuple, Union
from transformers import AutoTokenizer


class InstructionDataset(Dataset):
    def __init__(self, tokenizer: Union[AutoTokenizer, None] = None, max_length: Union[int, None] = None) -> None: 
        self.dataset = load_dataset()
        self.max_length = max_length
        self.tokenizer = tokenizer    
        self.instruction = """
        You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information ans answer `I do not know`.
        """
    
    @staticmethod
    def load_dataset():
        dataset = load_dataset("NebulaByte/E-Commerce_FAQs")
        dataset = dataset["train"]  # this dataset has only train subset
        # Shuffle dataset
        dataset = dataset.shuffle(seed=42)
        return dataset 
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx],
        }
        
    @staticmethod
    def create_prompt(instruction: str, input: str, output: Union[None, str]) -> str:
        """
        Creates a formatted prompt template for a prompt in the instruction dataset
        If output is not None it will be instruction + input_context, otherwise it will be
        instruction + input_context + output
        """

        # Initialize static strings for the prompt template
        INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
        INSTRUCTION_KEY = "### Instruction:"
        INPUT_KEY = "Input:"
        RESPONSE_KEY = "### Response:"
        END_KEY = "### End"

        # Combine a prompt with the static strings
        blurb = f"{INTRO_BLURB}"
        instruction = f"{INSTRUCTION_KEY}\n{instruction}"
        input_context = f"{INPUT_KEY}\n{input}"
        response = f"{RESPONSE_KEY}\n{output}"
        end = f"{END_KEY}"

        # Create a list of prompt template elements
        if output:
            parts = [
                part for part in [blurb, instruction, input_context, response, end] if part
            ]
        else:
            parts = [
                part for part in [blurb, instruction, input_context, response] if part
            ]        

        # Join prompt template elements into a single string to create the prompt template
        formatted_prompt = "\n\n".join(parts)
        return {"formatted_prompt": formatted_prompt}


class InstructionDataset_GPT(InstructionDataset):
    """ This dataset class should be used with AutoModelForCausalLM (decoder-only models)
    as it prepares the instruction dataset as input = output = instruction + question + answer
    """
    def __init__(self, tokenizer: Union[AutoTokenizer, None] = None, max_length: Union[int, None] = None) -> None: 
        super(InstructionDataset_GPT, self).__init__(tokenizer, max_length)
        self.dataset = self.dataset.map(
            lambda x: self.create_prompt(
                instruction=self.instruction, input=x["question"], output=x["answer"]
            )
        )
        self.data = [text + tokenizer.eos_token for text in self.dataset["formatted_prompt"]]
        (
            self.input_ids,
            self.attention_masks,
            self.labels,
        ) = self.tokenize_and_mask()

    def tokenize_and_mask(self):
        encodings = self.tokenizer(
            self.data,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids, attention_masks = encodings.input_ids, encodings.attention_mask
        
        # for decoder-only models copy input_ids to labels
        labels = input_ids.clone()
        # set PAD tokens equal to `-100` to not take them into account during loss calculation
        input_ids_lens = [tokenized.ne(self.tokenizer.pad_token_id).sum().item() for tokenized in input_ids]
        for label, source_len in zip(labels, input_ids_lens):
            label[source_len:] = -100
        return input_ids, attention_masks, labels   
        

class InstructionDataset_T5(InstructionDataset):
    """ This dataset class should be used with AutoModelForSeq2SeqLM (encoder-decoder models)
    as it prepares the instruction dataset as input = instruction + question, output = answer
    """
    def __init__(self, tokenizer: Union[AutoTokenizer, None] = None, max_length: Union[int, None] = None) -> None: 
        super(InstructionDataset_T5, self).__init__(tokenizer, max_length)
        self.dataset = self.dataset.map(
            lambda x: self.create_prompt(
                instruction=self.instruction, input=x["question"], output=None
            )
        )
        self.labels = [text for text in self.dataset['answer']]
        self.data = [text for text in self.dataset["formatted_prompt"]]
        (
            self.input_ids,
            self.attention_masks,
            self.labels,
        ) = self.tokenize_and_mask()

    def tokenize_and_mask(self):
        encodings = self.tokenizer(
            self.data,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids, attention_masks = encodings.input_ids, encodings.attention_mask
        
        # for encoder-decoder models labels are the expected answers only
        labels = self.tokenizer(
            self.labels,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).input_ids
        # set PAD tokens equal to `-100` to not take them into account during loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100
        return input_ids, attention_masks, labels


def prepare_dataloaders(tokenizer: AutoTokenizer, max_length: int, dataset_type: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    if dataset_type == "decoder":
        prepared_datasets = {
            "train": InstructionDataset_GPT(tokenizer, max_length=max_length),
            # "test": InstructionDataset_GPT(tokenizer, max_length=max_length),
        }
    elif dataset_type == "encoder-decoder":
        prepared_datasets = {
            "train": InstructionDataset_T5(tokenizer, max_length),
        }
    else:
        raise NotImplementedError("Only `decoder` and `encoder-decoder` dataset_type's are supported")
    
    train_loader = DataLoader(
        prepared_datasets["train"],
        batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = None
    if 'test' in prepared_datasets:
        test_loader = DataLoader(
            prepared_datasets["test"],
            batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )
    return train_loader, test_loader
