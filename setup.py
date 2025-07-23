from setuptools import setup, find_packages

setup(
    name="assoG2P",
    version="1.0.0",
    author = "chenrf",
    author_email = "12024128035@stu.ynu.edu.cn",
    description="Genome-wide association analysis toolkit",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    
    packages=find_packages(),
    package_dir={"": "."},
    include_package_data=True,
    
    install_requires=[
        "numpy>=1.20,<2.2",
        "pandas>=1.3",
        "scikit-learn>=1.0",
        "matplotlib>=3.5",
        "numba>=0.57",
        "lightgbm>=3.3",
        "xgboost>=1.6",
        "catboost>=1.0",
        "shap>=0.40",
        "plotly>=5.0",
        "kaleido>=0.2",
        "seaborn>=0.11",
    ],
    
    entry_points={
        "console_scripts": [
            "association=assoG2P.main:main",
        ],
    },
    
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)

# python setup.py sdist bdist_wheel