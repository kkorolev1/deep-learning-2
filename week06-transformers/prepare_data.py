import logging
import argparse
from glob import glob
import os
from hw_lm.datasets.TinyStoriesDataset import copy_stories_to_dir, save_tokenized_dataset
from hw_lm.tokenizer import Tokenizer

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

def main(args):
    logger.info("Creating txt stories for tokenizer training")
    copy_stories_to_dir(args.data_dir, args.txt_dir)
    tokenizer = Tokenizer(args.tokenizer_path)
    logger.info("Training a tokenizer")
    tokenizer.train(
        input_file=",".join(glob(os.path.join(args.txt_dir, "*.txt"))),
        vocab_size=args.vocab_size,
        model_type="bpe"
    )
    tokenizer.load_processor()
    logger.info("Tokenizing a dataset")
    save_tokenized_dataset(args.data_dir, args.out_dir, tokenizer)
    logger.info(f"Dataset is saved in {args.out_dir}")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Prepares dataset for training")
    args.add_argument(
        "-d",
        "--data-dir",
        default=None,
        type=str,
        required=True,
        help="Directory of TinyStories JSON files",
    )
    args.add_argument(
        "-t",
        "--txt-dir",
        default=None,
        type=str,
        required=True,
        help="Directory to write txt files with stories for tokenizer training",
    )
    args.add_argument(
        "-v",
        "--vocab-size",
        default=32000,
        type=int,
        help="Vocab size for tokenizer",
    )
    args.add_argument(
        "-m",
        "--tokenizer-path",
        default=None,
        type=str,
        required=True,
        help="Tokenizer model path",
    )
    args.add_argument(
        "-o",
        "--out-dir",
        default=None,
        type=str,
        required=True,
        help="Directory of tokenized stories",
    )
    main(args.parse_args())
