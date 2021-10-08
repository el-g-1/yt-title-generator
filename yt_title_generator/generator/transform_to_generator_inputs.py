import yt_title_generator.utils.sharded_dataset as sharded_dataset_utils
import os
import tensorflow as tf


def transform_to_inputs(context):
    '''Loads and transforms data set with GAN model inputs'''
    def example_to_input(example):
        script = example.features.feature["script"].int64_list.value
        latent = example.features.feature["latent"].int64_list.value
        yield ({"script": script, "latent": latent}, [1, 1])

    dataset = sharded_dataset_utils.load_dataset(
        os.path.join(context.gan.generator.data_dir, "*"),
        example_to_input,
        output_types=({"script": tf.int32, "latent": tf.int32}, tf.int32),
        output_shapes=None,
    )

    return dataset
