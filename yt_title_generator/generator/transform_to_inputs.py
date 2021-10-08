import yt_title_generator.utils.sharded_dataset as sharded_dataset_utils
import os
import tensorflow as tf


def transform_to_inputs(context):
    '''Loads and transforms data set with discriminator model inputs'''
    def example_to_input(example):
        label_view = example.features.feature["label_view"].int64_list.value[1]
        label_real = example.features.feature["label_real"].int64_list.value[0]

        title = example.features.feature["title"].int64_list.value
        script = example.features.feature["script"].int64_list.value
        yield ({"script": script, "title": title}, [label_view, label_real])

    dataset = sharded_dataset_utils.load_dataset(
        os.path.join(context.generator.data_dir, "*"),
        example_to_input,
        output_types=({"script": tf.int32, "title": tf.int32}, tf.int32),
        # output_shapes=((context.generator.max_script_tokens, context.generator.max_title_tokens), 2)
        output_shapes=None,
    )

    return dataset
