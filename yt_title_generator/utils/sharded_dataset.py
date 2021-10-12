import tensorflow as tf


def load_dataset(glob_path, example_to_input_fn, output_types=None, output_shapes=None):
    """Loads files into a dataset and executes example_to_input_fn for each element"""
    file_dataset = tf.data.Dataset.list_files(glob_path)

    record_dataset = file_dataset.interleave(
        lambda filename: tf.data.TFRecordDataset([filename])
    )

    def generator():
        for raw_record in record_dataset:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            for x in example_to_input_fn(example):
                yield x
        return generator

    return tf.data.Dataset.from_generator(
        generator, output_types=output_types, output_shapes=output_shapes,
    )
