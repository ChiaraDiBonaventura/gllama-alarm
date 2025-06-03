import os

import torch
import datasets
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.trainer_utils import enable_full_determinism
from transformers.pipelines.pt_utils import KeyDataset

import pandas as pd
import inspect
import format as format_module

MODELS_DIR = "."
DATA_DIR = 'data'

available_formats = {obj.format_name: obj
                     for name, obj in inspect.getmembers(format_module, inspect.isclass)
                     if issubclass(obj, format_module.Format) and not obj == format_module.Format}


def inference_single_example(parameters: dict):
    model_id = parameters['new_model_name']
    save_directory = MODELS_DIR
    model_path = os.path.join(save_directory, model_id)

    max_seq_length = parameters.get("max_seq_length", None)
    pretrained_parameters = parameters.get("pretrained_parameters", {})

    # dataset_path = parameters['dataset_path']
    # dataset_extension = dataset_path.split('.')[-1]

    # if dataset_extension == "jsonl":
    #     dataset_extension = "json"

    training_parameters = parameters.get("training_arguments", None)

    if training_parameters:
        print("Training parameters detected! Retrieving seed")
        seed = training_parameters.get("seed", 42)
        enable_full_determinism(seed)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=None,
        device_map="auto",
        **pretrained_parameters
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # if based on which dataset
    instruction = "Classify the input text as 'hate speech', 'offensive' or 'neutral'."
    context = "jewish harvard professor wants to abolish white race with wr"

    prompt = f"Below is an instruction that describes a task, paired with an input text. " \
                f"Write a response that appropriately completes the instruction.\n\n" \
                f"### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Answer: "

    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, device_map="auto", return_full_text=False, max_new_tokens=512)

    result = pipe(prompt) # pass list of prompts 

    print("\n\nRESULT:\n\n")
    print(result[0]['generated_text'])



def inference(parameters: dict, dataset_path:str):
    model_id = parameters['new_model_name']
    save_directory = MODELS_DIR
    model_path = os.path.join(save_directory, model_id)

    max_seq_length = parameters.get("max_seq_length", None)
    pretrained_parameters = parameters.get("pretrained_parameters", {})

    dataset_extension = dataset_path.split('.')[-1]

    if dataset_extension == "jsonl":
        dataset_extension = "json"

    datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True
    dataset = load_dataset(dataset_extension, data_files=os.path.join(DATA_DIR, dataset_path))['train']

    training_parameters = parameters.get("training_arguments", None)

    if training_parameters:
        print("Training parameters detected! Retrieving seed")
        seed = training_parameters.get("seed", 42)
        enable_full_determinism(seed)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=None,
        device_map="auto",
        torch_dtype=torch.bfloat16, 
        **pretrained_parameters
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    instruction_format = parameters['instruction_format']
    instruction_format_parameters = parameters.get('instruction_format_parameters', {})

    instruction_formatter = available_formats[instruction_format](**instruction_format_parameters)
    dataset = dataset.map(lambda x: {"input": instruction_formatter.get_prompt(x, False)})
    print(dataset['input'][0])

    results = []
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, device_map="auto", return_full_text=False, max_new_tokens=25)
    for x in dataset['input']:
        out = pipe(x)
        results.append(out)
        print(out)

    # results = []
    # for out in pipe(KeyDataset(dataset, "input"), batch_size=8):
    #    print(out)
    #    results.extend(out)

    # saving -- change the name
    pd.DataFrame.from_records({"results":results}).to_csv('/datadrive/disk2/chiaradb/predictions/llama_7b_kg_implicit_hate.csv')

