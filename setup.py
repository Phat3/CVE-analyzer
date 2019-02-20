import setuptools
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'Readme.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name="cve_analyzer",
    version="0.0.4",
    author="Sebastiano Mariani",
    author_email="mariani.sebastiano@gmail.com",
    description="Simple package that given a CVE desription tries to extract useful semantics from it using NLP",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/Phat3/CVE-analyzer",
    packages=setuptools.find_packages(),
    install_requires=[
        'spacy==2.0.18',
    ],
    include_package_data=True
)
