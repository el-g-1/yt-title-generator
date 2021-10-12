import tensorflow as tf
import tensorflow.keras.backend as K
import yt_title_generator.generator.transform_to_generator_inputs as transform_to_generator_inputs
import yt_title_generator.generator.train_discriminator_model as train_discriminator_model
import yt_title_generator.utils.run as run_utils
import yt_title_generator.utils.vocabulary as vocabulary_utils
import yt_title_generator.utils.example as example_utils
import tensorflow.keras.backend as K
import argparse
from yt_title_generator.word_embedding.inputs import tokenize
import os
import argparse
import glob
import numpy as np
from nltk.stem.snowball import SnowballStemmer
import os
import numpy as np


def cosine_similarity(v1, v2):
    """Computes cosine similarity metric"""
    return tf.matmul(v1, tf.transpose(v2, [1, 0]))


def load_model(checkpoint_path):
    return tf.keras.models.load_model(checkpoint_path, compile=False)


def set_cpu_session():
    """Disables GPU"""
    tf.config.set_visible_devices([], "GPU")
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != "GPU"


def eval_model_main(context, args):
    """Main function for model prediction"""

    if args.cpu:
        set_cpu_session()

    dataset = transform_to_generator_inputs.transform_to_inputs(context)

    model = load_model(context.gan.generator.model_dir)

    print(model.summary())

    vocab = vocabulary_utils.load_vocabulary_tsv(context.embedding.vocabulary_path)
    reversed_vocab = {v: k for k, v in vocab.items()}

    with open(args.script, "r") as scrdata:
        stemmer = SnowballStemmer("russian")
        script_lines = scrdata.readlines()
        script_features = []
        for line in script_lines:
            toks = tokenize(line, stemmer)
            for tok in toks:
                script_features.append(vocab.get(tok, 0))
        script_features = example_utils.pad_sequence(
            script_features, context.gan.discriminator.max_script_tokens
        )
        latent_features = [
            int(x) for x in np.random.randn(context.gan.generator.latent_dim)
        ]

        script = np.array(script_features).reshape((1, 1, len(script_features)))
        latent = np.array(latent_features).reshape((1, 1, len(latent_features)))

        dataset = tf.data.Dataset.from_tensor_slices(
            {"script": script, "latent": latent}
        )

        prediction = model.predict(dataset)
        print(f"Raw prediction ({prediction.shape}):", prediction)

        embedding = (
            load_model(context.embedding.model_dir)
            .get_layer("embedding")
            .get_weights()[0]
        )

        top_10_cosine = tf.nn.top_k(
            cosine_similarity(
                tf.nn.l2_normalize(prediction, dim=1),
                tf.nn.l2_normalize(embedding, dim=1),
            ),
            k=10,
        )
        sim, indices = top_10_cosine[0].numpy(), top_10_cosine[1].numpy()

        print("Top 10 (cosine):", indices, sim)
        print("Top 10 (cosine) words:")
        for i, val_sim in enumerate(zip(indices[0], sim[0])):
            words = []
            for val, sim in zip(val_sim[0], val_sim[1]):
                words.append(
                    "*" if sim < (args.maxdist or 0) else reversed_vocab.get(val, "%")
                )
            print(f"{i}:\t", ", ".join(words))


if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description="Eval generator model.")
    argparse.add_argument("--script", type=str, required=True)
    argparse.add_argument("--cpu", action="store_true")
    argparse.add_argument("--maxdist", type=int)
    run_utils.run_main(eval_model_main, argparse)
