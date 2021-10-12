import yt_title_generator.utils.sharded_dataset as sharded_dataset_utils
import yt_title_generator.utils.vocabulary as vocabulary_utils
import os
import tensorflow as tf


def transform_to_inputs(context):
    """Transforms input examples into CBOW examples"""
    vocab = vocabulary_utils.load_vocabulary_tsv(
        context.embedding.word_frequencies_path,
        freq_min=context.embedding.frequency_min,
        freq_max=context.embedding.frequency_max,
    )
    vocab_size = len(vocab)
    vocabulary_utils.save_vocabulary_tsv(
        vocab, context.embedding.vocabulary_path, save_frequencies=False
    )

    window_size = context.embedding.window_size

    def example_to_input(example):
        words = example.features.feature["tok"].bytes_list.value
        for i in range(window_size, len(words) - window_size):
            context_words = []
            target_word = words[i]
            for j in range(-window_size, window_size + 1):
                if j != 0:
                    context_words.append(words[i + j])
            x = [vocab.get(w.decode("utf-8"), 0) for w in context_words]
            y = tf.keras.utils.to_categorical(
                vocab.get(target_word.decode("utf-8"), 0), vocab_size
            )
            yield (x, y)

    dataset = sharded_dataset_utils.load_dataset(
        os.path.join(context.embedding.data_dir, "*"),
        example_to_input,
        output_types=(tf.int32, tf.int32),
        output_shapes=((window_size * 2), (vocab_size)),
    )

    return (dataset, vocab)
