from setuptools import setup

setup(
    name="postgres_utils",
    version="0.1.0",
    py_modules=[
        "get_psycopg2_connection",
        "insert_progress_message",
        "insert_run_id",
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
