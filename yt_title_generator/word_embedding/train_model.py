import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.utils as utils
import yt_title_generator.utils.run as run_utils
import yt_title_generator.utils.vocabulary as vocabulary_utils
import yt_title_generator.word_embedding.transform_to_inputs as transform_to_inputs
import argparse
import os


def load_model(checkpoint_path):
    '''Loads model by path'''
    return tf.keras.models.load_model(checkpoint_path, compile=False)


def define_model(window_size, vocab_size, embedding_dim):
    '''Defines architecture of word embedding model'''
    inp = tf.keras.layers.Input(shape=(window_size * 2))
    embed = tf.keras.layers.Embedding(
        input_dim=vocab_size, output_dim=embedding_dim, input_length=window_size * 2
    )(inp)
    mean = tf.keras.layers.Lambda(
        lambda x: K.mean(x, axis=1), output_shape=(embedding_dim,)
    )(embed)
    soft_max = tf.keras.layers.Dense(vocab_size, activation="softmax")(mean)
    return tf.keras.Model(inputs=inp, outputs=soft_max)


def train_model(model, dataset, checkpoint_path):
    '''Trains model'''
    model.compile(loss="categorical_crossentropy", optimizer="Adam")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path)

    model.fit(
        dataset.batch(1024).prefetch(16), epochs=100, callbacks=[checkpoint_callback]
    )


def train_model_main(context, args):
    '''Main function for model training'''
    dataset, vocab = transform_to_inputs.transform_to_inputs(context)

    model = define_model(
        context.embedding.window_size, len(vocab), context.embedding.embedding_dim
    )

    if args.print:
        print(model.summary())
        print([x for x in dataset.take(1).as_numpy_iterator()])
        return

    if args.plot_model:
        utils.plot_model(model, to_file=args.plot_model, show_shapes=True)
        return

    if os.path.exists(context.embedding.model_dir):
        if not args.resume:
            raise FileExistsError(
                f"{context.embedding.model_dir} already exists and 'resume' is not set"
            )
        load_model(context.embedding.model_dir)
    else:
        os.makedirs(context.embedding.model_dir)

    train_model(model, dataset, context.embedding.model_dir)


if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description="Train model for word embedding.")
    argparse.add_argument("--resume", action="store_true")
    argparse.add_argument("--print", action="store_true")
    argparse.add_argument(
        "--plot_model", type=str, help="Output png file with model structure."
    )
    run_utils.run_main(train_model_main, argparse)
