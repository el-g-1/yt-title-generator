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
    title,
    max_title_tokens,
    script_lines,
    max_script_tokens,
    view_count,
    subscriber_count,
    num_fake_examples,
    vocab,
    stemmer,
    writer,
):
    '''Processes all video-related data'''
    title_tokens = tokenize(title, stemmer)
    if not title_tokens:
        return
    title_features = example_utils.pad_sequence(
        [vocab.get(t, 0) for t in title_tokens], max_title_tokens
    )

    script_features = []
    for line in script_lines:
        toks = tokenize(line, stemmer)
        for tok in toks:
            script_features.append(vocab.get(tok, 0))
    script_features = example_utils.pad_sequence(script_features, max_script_tokens)

    ratio = view_count / subscriber_count if subscriber_count != 0 else 0.0

    view_labels = [ratio > threshold for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]

    example = example_utils.make_example(
        {
            "label_view": example_utils.make_int_feature(view_labels),
            "label_real": example_utils.make_int_feature([1]),
            "title": example_utils.make_int_feature(title_features),
            "script": example_utils.make_int_feature(script_features),
        }
    )

    writer.write(example)

    # Generate fake examples.
    for _ in range(num_fake_examples):
        fake = example_utils.make_example(
            {
                "label_view": example_utils.make_int_feature(
                    [0 for _ in range(len(view_labels))]
                ),
                "label_real": example_utils.make_int_feature([0]),
                "title": example_utils.make_int_feature(
                    np.random.random_integers(0, len(vocab), len(title_features))
                ),
                "script": example_utils.make_int_feature(script_features),
            }
        )
        writer.write(fake)


def process_db_video(context, filename, vocab, stemmer, writer):
    '''Processes individual video file'''
    with open(filename, "r") as fdata:
        video_data = json.load(fdata)
        video_id = video_data["id"]

        title = video_data["info"]["snippet"]["title"]
        channel_id = video_data["info"]["snippet"]["channelId"]

        subscriber_count = 0

        channel_path = os.path.join(context.db.channel_dir, channel_id)
        with open(channel_path, "r") as chdata:
            channel_data = json.load(chdata)
            subscriber_count = int(
                channel_data["info"]["statistics"]["subscriberCount"]
            )

        view_count = int(video_data["info"]["statistics"]["viewCount"])
        view_count = int(view_count)

        script_path = os.path.join(context.db.script_dir, video_id)
        with open(script_path, "r") as scrdata:
            process_features(
                title,
                context.generator.max_title_tokens,
                scrdata.readlines(),
                context.generator.max_script_tokens,
                view_count,
                subscriber_count,
                context.generator.num_fake_examples,
                vocab,
                stemmer,
                writer,
            )


def generate_examples(context, args):
    '''Generates examples for discriminator model'''
    files_in_db_dir = glob.glob(os.path.join(context.db.video_dir, "*"))
    filepaths = [os.path.join(context.db.video_dir, f) for f in files_in_db_dir]

    if os.path.exists(context.generator.data_dir):
        raise FileExistsError(f"{context.generator.data_dir} already exists")

    os.makedirs(context.generator.data_dir)

    stemmer = SnowballStemmer("russian")
    writer = sharded_writer_utils.ShardedWriter(output_dir=context.generator.data_dir)

    vocab = vocabulary_utils.load_vocabulary_tsv(context.embedding.vocabulary_path)

    def work(filepath):
        process_db_video(context, filepath, vocab, stemmer, writer)

    return run_utils.run_with_progress(
        work,
        filepaths,
        num_workers=context.generator.num_workers,
        progress_text="[Discriminator] Generate examples",
    )


if __name__ == "__main__":
    run_utils.run_main(
        generate_examples, argparse.ArgumentParser(description="Generate data for GAN.")
    )
