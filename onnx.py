""" ONNX conversion and inference for BERT text classification """
import argparse
import os
import sys
import tqdm
from typing import List, Tuple, Dict
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from onnxruntime import (
  GraphOptimizationLevel, 
  InferenceSession,
  SessionOptions, 
  get_all_providers
)


def bert_model_to_onnx(model_path: str, out_onnx_path: str) -> None:
    """Конвертация модели из формата PyTorch в формат ONNX.
    Args:
        model_path (str): путь к текущей модели
        out_onnx_path (str): путь сохранения модели в формате ONNX
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, config=os.path.join(model_path, "config.json")
    )
    model.eval()

    dummy_model_input = tokenizer(
        "hi!",
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to("cpu")

    torch.onnx.export(
        model,
        (
            dummy_model_input["input_ids"],
            dummy_model_input["attention_mask"],
            dummy_model_input["token_type_ids"],
        ),
        out_onnx_path,
        opset_version=12,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["output_logit"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_len"},
            "attention_mask": {0: "batch_size", 1: "sequence_len"},
            "token_type_ids": {0: "batch_size", 1: "sequence_len"},
            "output_logit": {0: "batch_size", 1: "sequence_len", 2: "num_classes"},
        },
    )

    main_onnx_folder = "/".join(out_onnx_path.split("/")[:-1])
    tokenizer.save_pretrained(main_onnx_folder)
    if "config.json" in os.listdir(model_path):
        os.system(
            f"cp {os.path.join(model_path, 'config.json')} {os.path.join(main_onnx_folder, 'config.json')}"
        )
    print(f"Saved at {out_onnx_path}")


class BaseDataset(Dataset):
    """ Class of torch Dataset for simple iteration through texts """
    def __init__(self, text: List[str]) -> None:
        self.text = text

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, idx: int) -> str:
        return self.text[idx]


class InferenceBertONNXModel(object):
    """Base class for model inference with common data processing pipeline"""

    def __init__(
        self, model_checkpoint_path: str, device: int
    ) -> None:
        """Base initialization (common for both transformers and ONNX models)

        Args:
            model_checkpoint_path (str): path to folder with model and tokenizer
            reference_path (str): path to json-file with id2label mapping
            device (int): -1 if run on CPU otherwise CUDA device identifier
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_path)
        self.map_device_name = {
            -1: {"provider": "CPUExecutionProvider", "device_name": "cpu"},
            0: {"provider": "CUDAExecutionProvider", "device_name": "cuda:0"},
        }
        self.device = device
        self.device_name = self.map_device_name[self.device]["device_name"]

        full_model_path = os.path.join(model_checkpoint_path, model_file_name)
        provider = self.map_device_name[self.device]["provider"]
        self.model = self.create_model_for_provider(full_model_path, provider)

    @staticmethod
    def create_model_for_provider(model_path: str, provider: str) -> InferenceSession:
        """Method to load ONNX model on specified device

        Args:
            model_path (str): model.onnx file path
            provider (str): one of ["CPUExecutionProvider", "CUDAExecutionProvider"]

        Returns:
            InferenceSession: onnx session to run with inputs
        """
        assert (
            provider in get_all_providers()
        ), f"provider {provider} not found, {get_all_providers()}"
        # Few properties that might have an impact on performances (provided by MS)
        options = SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        # Load the model as a graph and prepare the CPU backend
        session = InferenceSession(model_path, options, providers=[provider])
        session.disable_fallback()
        return session


    def infer(self, batch: Dict[str, List[int]]) -> torch.Tensor:
        """Forward call to the model
        Arguments:
            batch (Dict[str, List[int]]): dictionary with all needed elements for the model
        Returns:
            torch.tensor: tensor of logits
        """
        inputs_onnx = {k: v.cpu().detach().numpy() for k, v in batch.items()}
        output = self.model.run(None, inputs_onnx)
        return torch.tensor(output[0])


    def predict(
        self, texts: List[str], batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """General functionfor prediction on a list of texts

        Args:
            texts (List[str]): texts for prediction
            batch_size (int): batch size

        Returns:
            Tuple[torch.Tensor, torch.Tensor, List[str]]: probability for a predicted class,
        id of predicted class, name of predicted class
        """
        dataset = BaseDataset(texts)
        dl = DataLoader(dataset, batch_size=batch_size)
        all_probas = []
        for batch in tqdm.tqdm(dl):
            encoded_text = self.tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            )
            logits = self.infer(encoded_text)
            probas = torch.softmax(logits, dim=-1)
            all_probas.append(probas)
        all_probas = torch.cat(all_probas)
        max_proba, predicted_label = torch.max(all_probas, axis=1)
        return max_proba, predicted_label
