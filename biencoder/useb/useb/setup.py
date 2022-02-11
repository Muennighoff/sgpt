from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    readme = fh.read()

setup(
    name="useb",
    version="0.0.1",
    author="Kexin Wang",
    author_email="kexin.wang.2049@gmail.com",
    description="Heterogenous, Task- and Domain-Specific Benchmark for Unsupervised Sentence Embeddings used in the TSDAE paper.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/kwang2049/useb",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        'sentence-transformers>=1.2.0',
        'pytrec_eval'
    ],
)