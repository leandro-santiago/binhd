"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages

setup(
    name="binhd",
    version="1.0.0-alpha",
    author="Leandro Santiago de Araújo",
    description="BinHD is a Python implementation based on A Binary Learning Framework for Hyperdimensional Computing",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/leandro-santiago/binhd",
    license="MIT",
    install_requires=[
        "torch-hd",
        "ucimlrepo",
        "pandas",                
    ],    
    packages=find_packages(exclude=["examples"]),
)
