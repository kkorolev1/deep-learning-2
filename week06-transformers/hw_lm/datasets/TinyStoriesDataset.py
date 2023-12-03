import json
from glob import glob
import os
import numpy as np
from tqdm.auto import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
from hw_lm.tokenizer import Tokenizer


class TinyStoriesDataset:
    def __init__(self,
                 is_test=False,
                 config_parser=None):
        self.data_dir = config_parser["dataset"].get("data_dir", None)
        self.limit = config_parser["dataset"].get("limit", None)
        self.padding_mode = config_parser["dataset"].get("padding_mode", "truncate")
        assert self.padding_mode == "truncate" or self.padding_mode == "padding", "Padding mode should be either `truncate` or `padding`"
        self.seq_len = config_parser["dataset"].get("seq_len", 256)
        tokenizer_model_path = config_parser["dataset"].get("tokenizer_model_path", None)
        self.tokenizer = Tokenizer(model_path=tokenizer_model_path)
        self.story_list = np.load(os.path.join(self.data_dir, "TinyStoriesTokenized.npy"), allow_pickle=True)
        self.val_size = config_parser["dataset"].get("val_size", 0.05)
        if self.limit is not None:
            self.story_list = self.story_list[:self.limit]
        train_size = int(len(self.story_list) * (1 - self.val_size))
        if not is_test:
            self.story_list = self.story_list[:train_size]
        else:
            self.story_list = self.story_list[train_size:]
        self.is_test = is_test
        self.config_parser = config_parser

    def __getitem__(self, index):
        return {
            "input_ids": torch.tensor(self.story_list[index])
        }
    
    def __len__(self):
        return len(self.story_list)

    def _add_bos_eos(self, input_ids):
        return torch.cat((
            torch.tensor([self.tokenizer.processor.bos_id()]),
            input_ids,
            torch.tensor([self.tokenizer.processor.eos_id()])
        ))

    def get_collate_fn(self):
        pad_id = self.tokenizer.processor.pad_id()
        def collate_fn(dataset_items):
            """
            Collate and pad fields in dataset items
            """
            result_batch = {}
            if self.padding_mode == "padding":
                result_batch["input_ids"] = pad_sequence(
                    [self._add_bos_eos(item["input_ids"]) for item in dataset_items], 
                    batch_first=True,
                    padding_value=pad_id
                )
            else:
                result_batch["input_ids"] = pad_sequence(
                    [self._add_bos_eos(item["input_ids"][:self.seq_len]) for item in dataset_items], 
                    batch_first=True,
                    padding_value=pad_id
                )
            result_batch["padding_mask"] = (result_batch["input_ids"] == pad_id)
            return result_batch
        return collate_fn


def copy_stories_to_dir(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    json_files = sorted(glob(os.path.join(data_dir, "*.json")))
    for file in tqdm(json_files, "Processing json files"):
        with open(file) as f_in:
            item_list = json.load(f_in)
            output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file))[0]}.txt")
            with open(output_path, "w") as f_out:
                f_out.write("\n".join([item["story"] for item in item_list]))
            print(f"Wrote to {output_path}")


def save_tokenized_dataset(data_dir, output_dir, tokenizer):
    os.makedirs(output_dir, exist_ok=True)
    json_files = glob(os.path.join(data_dir, "*.json"))
    story_list = []
    for file in tqdm(json_files, "Processing json files"):
        with open(file) as f:
            item_list = json.load(f)
            for item in item_list:
                tokenized_story = np.array(tokenizer.encode(item["story"]))
                story_list.append(tokenized_story)
    story_array = np.array(story_list, dtype=object)
    np.save(os.path.join(output_dir, "TinyStoriesTokenized.npy"), story_array, allow_pickle=True)