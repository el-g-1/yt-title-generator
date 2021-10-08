import yt_title_generator.utils.run as run_utils
import yt_title_generator.utils.sharded_writer as sharded_writer_utils
import yt_title_generator.utils.vocabulary as vocabulary_utils
import yt_title_generator.utils.example as example_utils
from yt_title_generator.word_embedding.inputs import tokenize
import os
import argparse
import glob
import numpy as np
from nltk.stem.snowball import SnowballStemmer
import json


def process_features(
    script_lines,
    max_script_tokens,
    latent_dim,
    num_latent_examples,
    vocab,
    stemmer,
    writer,
):
    '''Processes all video data'''
    script_features = []
    for line in script_lines:
        toks = tokenize(line, stemmer)
        for tok in toks:
            script_features.append(vocab.get(tok, 0))
    script_features = example_utils.pad_sequence(script_features, max_script_tokens)
    # Save to file pairs of script and random vectors for generator training
    for _ in range(num_latent_examples):
        latent_features = [int(x) for x in np.random.randn(latent_dim)]

        example = example_utils.make_example(
            {
                "latent": example_utils.make_int_feature(latent_features),
                "script": example_utils.make_int_feature(script_features),
            }
        )

        writer.write(example)


def process_db_video(context, filename, vocab, stemmer, writer):
    '''Fetches script path from video data and processes features'''
    with open(filename, "r") as fdata:
        video_data = json.load(fdata)
        video_id = video_data["id"]

        script_path = os.path.join(context.db.script_dir, video_id)
        with open(script_path, "r") as scrdata:
            process_features(
                scrdata.readlines(),
                context.gan.discriminator.max_script_tokens,
                context.gan.generator.latent_dim,
                context.gan.generator.num_latent_examples,
                vocab,
                stemmer,
                writer,
            )


def generate_examples(context, args):
    files_in_db_dir = glob.glob(os.path.join(context.db.video_dir, "*"))
    filepaths = [os.path.join(context.db.video_dir, f) for f in files_in_db_dir]

    if os.path.exists(context.gan.generator.data_dir):
        raise FileExistsError(f"{context.gan.generator.data_dir} already exists")

    os.makedirs(context.gan.generator.data_dir)

    stemmer = SnowballStemmer("russian")
    writer = sharded_writer_utils.ShardedWriter(
        output_dir=context.gan.generator.data_dir
    )

    vocab = vocabulary_utils.load_vocabulary_tsv(context.embedding.vocabulary_path)

    def work(filepath):
        process_db_video(context, filepath, vocab, stemmer, writer)

    return run_utils.run_with_progress(
        work,
        filepaths,
        num_workers=context.gan.generator.num_workers,
        progress_text="[Generator] Generate examples",
    )


if __name__ == "__main__":
    run_utils.run_main(
        generate_examples, argparse.ArgumentParser(description="Generate data for GAN.")
    )
