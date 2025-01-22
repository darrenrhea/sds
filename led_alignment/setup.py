from setuptools import setup

setup(
    name="led_alignment",
    version="0.1.0",
    py_modules=[
        "a2dc_annotate_2d_correspondences",
        "elaq_evaluate_led_alignment_quality",
    ],
    license="MIT",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
    ],
    entry_points={
        "console_scripts": [
            "a2dc_annotate_2d_correspondences = a2dc_annotate_2d_correspondences_cli_tool:a2dc_annotate_2d_correspondences_cli_tool",
            "elaq_evaluate_led_alignment_quality = elaq_evaluate_led_alignment_quality_cli_tool:elaq_evaluate_led_alignment_quality_cli_tool",
        ],
    },
)
