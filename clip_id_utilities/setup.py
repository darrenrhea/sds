from setuptools import setup

setup(
    name="clip_id_utilities",
    version="0.1.0",
    py_modules=[
        "load_all_clips",
        "validate_a_clip_entity",
    ],
    license="MIT",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
    ],
    entry_points={
        "console_scripts": [
            "mci_migrate_clip_id = mci_migrate_clip_id_cli_tool:mci_migrate_clip_id_cli_tool",
            "nci_new_clip_id = nci_new_clip_id_cli_tool:nci_new_clip_id_cli_tool",
            "vaci_validate_all_clip_ids = vaci_validate_all_clip_ids_cli_tool:vaci_validate_all_clip_ids_cli_tool",
        ],
    },
)
