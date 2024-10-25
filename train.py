#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import wandb
import copy
import logging
import tqdm
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

import torch
import transformers
import utils
from torch.utils.data import Dataset
from transformers import Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_paths: List[str] = field(default_factory=lambda: [], metadata={"help": "Path to the training data."})
    len_segment: int = field(default=2)
    len_offset: int = field(default=1)
    block_size: int = field(default=1024)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    enable_lora: bool = field(default=False)
    lora_rank: Optional[int] = field(default=None)
    load_in_4bit: bool = field(default=False)
    load_in_8bit: bool = field(default=False)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_paths: List[str], tokenizer: transformers.PreTrainedTokenizer, len_segment: int, len_offset: int, block_size: int, model_max_length: int):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = sum((utils.jload(one_file_path) for one_file_path in data_paths), start=[])

        logging.warning("Formatting and tokenizing inputs...")
        # Formatting for Bamboo-style timeline-reorder and tokenizing
        self.input_ids = []
        self.labels = []
        len_segment = len_segment * block_size
        len_offset = len_offset * block_size
        for example in tqdm.tqdm(list_data_dict, desc="Tokenize"):
            context_ids = tokenizer(example['content'] + tokenizer.bos_token, add_special_tokens=False, return_tensors='pt').input_ids.flatten()
            self.input_ids += [context_ids[s:s+len_segment] for s in range(0, context_ids.shape[-1], len_offset)]
            self.labels += [context_ids[s:s+len_segment] for s in range(0, context_ids.shape[-1], len_offset)]
            if 'qa_pairs' in example:
                qa_pairs = example['qa_pairs']
            else:
                qa_pairs = [example]
            for qa in qa_pairs:
                prompts = [
                    "Please sort the given events in the order of their appearance in the following long texts, from first to last.",
                    example['content'],
                    "Please sort the given events in the order of their appearance in the long texts, from first to last. The given events are:",
                ]
                prompts += [f"[{i + 1}]: {summary}" for i, summary in enumerate(qa['summaries'])]
                prompts += ["For example, a valid answer is [2] < [3] < [1] < [4] < [5]."]
                messages_in = [
                    {'role': 'system', 'content': "You are a helpful assistant."},
                    {'role': 'user', 'content': '\n'.join(prompts)},
                ]
                response = ' < '.join(f'[{i}]' for i in qa['answer'])
                input_ids = tokenizer.apply_chat_template(messages_in, return_tensors='pt', add_generation_prompt=True, return_dict=True)['input_ids'].flatten()
                output_ids = tokenizer(response, return_tensors='pt', add_special_tokens=True).input_ids.flatten()
                max_input_length = model_max_length - output_ids.shape[-1]
                if input_ids.shape[-1] > max_input_length:
                    input_ids = torch.concat((input_ids[:max_input_length//2], input_ids[-max_input_length//2:]), dim=-1)
                self.input_ids.append(torch.concat((input_ids, output_ids), dim=-1))
                self.labels.append(torch.concat((torch.full_like(input_ids, IGNORE_INDEX, dtype=torch.long), output_ids), dim=-1))
        print('!' * 100, f"#Training samples = {len(self.input_ids)}")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments, model_max_length: int) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_paths=data_args.data_paths, len_segment=data_args.len_segment, len_offset=data_args.len_offset, block_size=data_args.block_size, model_max_length=model_max_length)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    wandb.login(key=os.environ.get('WANDB_API_KEY'))
    if int(os.environ.get('LOCAL_RANK', 0)) == 0:
        wandb.init(
            project="Long-Context-SFT",
            name=training_args.run_name,
            config={
                'learning_rate': training_args.learning_rate,
                'batch_size': training_args.per_device_train_batch_size * torch.cuda.device_count() * training_args.gradient_accumulation_steps
            }
        )

    if training_args.load_in_4bit:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
        )
    elif training_args.load_in_8bit:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
            )
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype=torch.bfloat16,
        )
    if training_args.enable_lora:
        lora_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=2 * training_args.lora_rank,
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        print(model)
        model.print_trainable_parameters()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, model_max_length=training_args.model_max_length)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
