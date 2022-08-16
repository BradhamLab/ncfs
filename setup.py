from setuptools import setup

with open("readme.md", "r") as f:
    long_description = f.read()

setup(
    name="ncfs",
    version="0.1.2",
    description="Python implementation of Neighborhood Component Feature Selection (NCFS)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BradhamLab/ncfs",
    author="Dakota Y. Hawkins",
    author_email="dyh0110@bu.edu",
    license="BSD",
    packages=["ncfs"],
    install_requires=["numpy", "scipy", "numba", "scikit-learn"],
    classifiers=["Programming Language :: Python :: 3"],
)
