from setuptools import setup

setup(
    name="train_and_infer",
    version="0.1.0",
    py_modules=[
        "cmottd_compare_multiple_outputs_to_training_data",
        "create_padded_channel_stacks_np_u16",
        "graph_progress",
        "make_a_channel_stack_u16_from_local_file_pathed_annotation",
        "make_preprocessed_channel_stacks_u16",
        "train_multiple_output_model",
        "train_multiple_output_model_caller",
        "training_loop_for_multiple_outputs",
    ],
    license="MIT",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
    ],
    entry_points={
        "console_scripts": [
        ],
    },
)
