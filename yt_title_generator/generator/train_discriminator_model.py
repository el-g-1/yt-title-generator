import tensorflow as tf
import tensorflow.keras.backend as K
import yt_title_generator.generator.transform_to_inputs as transform_to_inputs
import yt_title_generator.utils.run as run_utils
import yt_title_generator.utils.vocabulary as vocabulary_utils
import argparse
import os


def load_model(checkpoint_path):
    return tf.keras.models.load_model(checkpoint_path, compile=False)


def define_model(context, embedding_title=None):
    '''Defines architecture of discriminator model'''
    embedding_model = load_model(context.embedding.model_dir)

    vocab = vocabulary_utils.load_vocabulary_tsv(context.embedding.vocabulary_path)
    vocab_size = len(vocab)

    embedding_dim = context.embedding.embedding_dim

    input_script = tf.keras.layers.Input(
        shape=(context.gan.discriminator.max_script_tokens,), name="script"
    )
    embedding_script = tf.keras.layers.Embedding(
        vocab_size,
        embedding_dim,
        input_length=1,
        weights=embedding_model.get_layer("embedding").get_weights(),
        trainable=False,
        name="embedding",
    )(input_script)
    lstm_script = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            64,
            input_shape=(embedding_dim,),
            dropout=0.1,
            recurrent_dropout=0.1,
            return_sequences=True,
        ),
        name="bidirectional",
    )(embedding_script)

    input_title = embedding_title
    if embedding_title is None:
        input_title = tf.keras.layers.Input(shape=(15,), name="title")
        embedding_title = tf.keras.layers.Embedding(
            vocab_size,
            embedding_dim,
            input_length=15,
            weights=embedding_model.get_layer("embedding").get_weights(),
            trainable=False,
            name="embedding_1",
        )(input_title)

    lstm_title = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            64,
            input_shape=(embedding_dim,),
            dropout=0.1,
            recurrent_dropout=0.1,
            return_sequences=True,
        ),
        name="bidirectional_1",
    )(embedding_title)
    concat = tf.keras.layers.Concatenate(axis=1, name="concatenate")(
        [lstm_script, lstm_title]
    )
    flat = tf.keras.layers.Flatten(name="flatten")(concat)

    dense = tf.keras.layers.Dense(2, activation="sigmoid", name="dense")(flat)

    return tf.keras.Model(inputs=[input_script, input_title], outputs=dense)


def train_model(model, dataset, checkpoint_path):
    '''Trains discriminator model'''
    model.compile(loss="binary_crossentropy", optimizer="Adam")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path)

    model.fit(
        dataset.batch(128).prefetch(64), epochs=100, callbacks=[checkpoint_callback]
    )


def train_model_main(context, args):
    '''Main function for model training'''
    dataset = transform_to_inputs.transform_to_inputs(context)

    model = define_model(context)

    if args.print:
        print(model.summary())
        print([x for x in dataset.take(1).as_numpy_iterator()])
        return

    if os.path.exists(context.gan.discriminator.model_dir):
        if not args.resume:
            raise FileExistsError(
                f"{context.gan.discriminator.model_dir} already exists and 'resume' is not set"
            )
        load_model(context.gan.discriminator.model_dir)
    else:
        os.makedirs(context.gan.discriminator.model_dir)

    train_model(model, dataset, context.gan.discriminator.model_dir)


if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description="Train model for word embedding.")
    argparse.add_argument("--resume", action="store_true")
    argparse.add_argument("--print", action="store_true")
    run_utils.run_main(train_model_main, argparse)
