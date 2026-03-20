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
    name="assog2p",
    packages=find_packages(),
    package_dir={"": "."},
    include_package_data=True,
    package_data={
        "assoG2P": ["bin/software/*"],
    },
    entry_points={
        "console_scripts": [
            "assog2p=assoG2P.main:main",
        ],
    },
)