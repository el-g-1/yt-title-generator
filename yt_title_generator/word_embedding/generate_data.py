import collections
import yt_title_generator.word_embedding.inputs as inputs
import yt_title_generator.utils.sharded_writer as sharded_writer
import yt_title_generator.utils.example as example_utils
import yt_title_generator.utils.vocabulary as vocabulary_utils
import yt_title_generator.utils.run as run_utils
import yt_title_generator.context as context
import argparse
import os
import json
import concurrent
import glob
import threading
from nltk.stem.snowball import SnowballStemmer


def process_text(text, stemmer, writer):
    """Processes individual comment"""
    vocab = collections.defaultdict(int)

    tokens = inputs.tokenize(text, stemmer)

    if not tokens:
        return vocab

    writer.write(
        example_utils.make_example({"tok": example_utils.make_string_feature(tokens)})
    )

    for tok in tokens:
        vocab[tok] += 1

    return vocab


def process_db_video(filepath, stemmer, writer):
    """Processes individual video file"""
    vocab = collections.defaultdict(int)

    with open(os.path.join(filepath), "r") as fdata:
        data = json.load(fdata)
        if "comments_threads" not in data:
            return vocab
        for item in data["comments_threads"]:
            vocab_part = process_text(item.get("text", ""), stemmer, writer)
            vocabulary_utils.merge_sum(vocab, vocab_part)
    return vocab


def generate_examples(context, args):
    """Generates examples for word embedding"""
    files_in_db_dir = glob.glob(os.path.join(context.db.video_dir, "*"))
    filepaths = [os.path.join(context.db.video_dir, f) for f in files_in_db_dir]

    lock = threading.Lock()

    if os.path.exists(context.embedding.data_dir):
        raise FileExistsError(f"{context.embedding.data_dir} already exists")

    os.makedirs(context.embedding.data_dir)

    stemmer = SnowballStemmer("russian")
    writer = sharded_writer.ShardedWriter(output_dir=context.embedding.data_dir)

    vocab = collections.defaultdict(int)

    def work(filepath):
        vocab_part = process_db_video(filepath, stemmer, writer)
        with lock:
            vocabulary_utils.merge_sum(vocab, vocab_part)

    exceptions = run_utils.run_with_progress(
        work,
        filepaths,
        num_workers=context.embedding.num_workers,
        progress_text="[Embedding] Generate examples",
    )

    vocabulary_utils.save_vocabulary_tsv(vocab, context.embedding.word_frequencies_path)

    return exceptions


if __name__ == "__main__":
    run_utils.run_main(
        generate_examples,
        argparse.ArgumentParser(description="Generate data for word embedding."),
    )
