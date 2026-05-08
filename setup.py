"""
Legacy shim for users running ``python setup.py ...``.

The real, modern configuration lives in ``pyproject.toml`` and the project
is meant to be built via a PEP 517 frontend, e.g.:

    pip install build
    python -m build

You can also install in editable mode with:

    pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="G2PInsight",
    packages=find_packages(),
    package_dir={"": "."},
    include_package_data=True,
    package_data={
        "G2PInsight": ["bin/software/*"],
    },
    entry_points={
        "console_scripts": [
            "G2PInsight=G2PInsight.main:main",
        ],
    },
)