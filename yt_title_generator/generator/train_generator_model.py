import tensorflow as tf
import yt_title_generator.generator.transform_to_generator_inputs as transform_to_generator_inputs
import yt_title_generator.generator.train_discriminator_model as train_discriminator_model
import yt_title_generator.utils.run as run_utils
import yt_title_generator.utils.vocabulary as vocabulary_utils
import argparse
import os


def load_model(checkpoint_path):
    return tf.keras.models.load_model(checkpoint_path, compile=False)


def define_generator_model(context):
    """Defines architecture for generator model"""
    input_script = tf.keras.layers.Input(
        shape=(context.gan.discriminator.max_script_tokens,), name="script"
    )
    input_latent = tf.keras.layers.Input(
        shape=(context.gan.generator.latent_dim,), name="latent"
    )

    embedding_model = load_model(context.embedding.model_dir)

    vocab = vocabulary_utils.load_vocabulary_tsv(context.embedding.vocabulary_path)
    vocab_size = len(vocab)

    embedding_dim = context.embedding.embedding_dim

    embedding = tf.keras.layers.Embedding(
        vocab_size,
        embedding_dim,
        input_length=1,
        weights=embedding_model.get_layer("embedding").get_weights(),
        trainable=False,
    )(input_script)

    lstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            128,
            input_shape=(embedding_dim,),
            dropout=0.3,
            recurrent_dropout=0.3,
            return_sequences=True,
        )
    )(embedding)

    dense_lstm = tf.keras.layers.Dense(32, activation="relu")(lstm)

    flat = tf.keras.layers.Flatten()(dense_lstm)

    concat = tf.keras.layers.Concatenate(axis=-1)([input_latent, flat])

    dense = tf.keras.layers.Flatten()(
        tf.keras.layers.Dense(
            context.embedding.embedding_dim
            * context.gan.discriminator.max_title_tokens,
            activation="relu",
        )(concat)
    )
    out = tf.keras.layers.Reshape(
        (context.gan.discriminator.max_title_tokens, context.embedding.embedding_dim)
    )(dense)
    return tf.keras.Model(
        inputs=[input_script, input_latent], outputs=out, name="generator_model"
    )


def load_discriminator_model(context):
    """Loads discriminator model without embedding layer for titles"""

    input_after_embed = tf.keras.layers.Input(
        shape=(
            context.gan.discriminator.max_title_tokens,
            context.embedding.embedding_dim,
        ),
        name="title",
    )

    model = train_discriminator_model.define_model(
        context, embedding_title=input_after_embed
    )

    full_model = load_model(context.gan.discriminator.model_dir)

    for l in model.layers:
        l.set_weights(full_model.get_layer(l.name).get_weights())

    return model


def define_gan_model(context):
    """Defines architecture for GAN model"""

    input_script = tf.keras.layers.Input(
        shape=(context.gan.discriminator.max_script_tokens,), name="script"
    )
    input_latent = tf.keras.layers.Input(
        shape=(context.gan.generator.latent_dim,), name="latent"
    )

    generator_model = define_generator_model(context)
    generator_out = generator_model([input_script, input_latent])

    discriminator_model = load_discriminator_model(context)

    out = discriminator_model([input_script, generator_out])
    out.trainable = False

    return (
        tf.keras.Model(inputs=[input_script, input_latent], outputs=out),
        generator_model,
    )


def train_model(context, model, generator, dataset):
    """Trains GAN model"""
    model.compile(loss="binary_crossentropy", optimizer="Adam")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=context.gan.generator.full_model_dir
    )

    model.fit(
        dataset.batch(10).prefetch(10),
        epochs=100,
        callbacks=[
            checkpoint_callback,
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda batch, logs: generator.save(
                    context.gan.generator.model_dir
                )
            ),
        ],
    )


def set_cpu_session():
    """Disables GPU"""
    tf.config.set_visible_devices([], "GPU")
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != "GPU"


def train_model_main(context, args):
    """Main function for GAN model training"""

    if args.cpu:
        set_cpu_session()

    dataset = transform_to_generator_inputs.transform_to_inputs(context)

    model, generator = define_gan_model(context)

    if args.print:
        print(model.summary())
        print([x for x in dataset.take(1).as_numpy_iterator()])
        return

    if os.path.exists(context.gan.generator.full_model_dir):
        if not args.resume:
            raise FileExistsError(
                f"{context.gan.generator.full_model_dir} already exists and 'resume' is not set"
            )
        load_model(context.gan.generator.full_model_dir)
    else:
        os.makedirs(context.gan.generator.full_model_dir)

    train_model(context, model, generator, dataset)


if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description="Train model for word embedding.")
    argparse.add_argument("--resume", action="store_true")
    argparse.add_argument("--print", action="store_true")
    argparse.add_argument("--cpu", action="store_true")
    run_utils.run_main(train_model_main, argparse)
