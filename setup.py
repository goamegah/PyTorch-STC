# Copyright (c) UPC.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from setuptools import setup, find_packages


NAME = 'torchSTC'
DESCRIPTION = 'PyTorch Self Training approch for Short Text Clustering'

URL = 'https://github.com/goamegah/torchSTC'
AUTHOR = 'Godwin AMEGAH'
EMAIL = 'komlan.godwin.amegah@gmail.com'
REQUIRES_PYTHON = '>=3.6'

for line in open('torchclust/__init__.py'):
    line = line.strip()
    if '__version__' in line:
        context = {}
        exec(line, context)
        VERSION = context['__version__']

HERE = Path(__file__).parent

try:
    with open(HERE / "README.md", encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Handle the case when requirements.txt does not exist
# requirements_file = HERE / 'requirements.txt'
# if requirements_file.is_file():
#     with open(requirements_file) as f:
#         REQUIRED = f.read().splitlines()
# else:
#     REQUIRED = []

REQUIRED = [i.strip() for i in open(HERE / 'requirements.txt') if not i.startswith('#')]


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author_email=EMAIL,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    url=URL,
    python_requires=REQUIRES_PYTHON,
    install_requires=REQUIRED,
    extras_require={
        'dev': ['coverage', 'flake8', 'pytest', 'torchinfo'],
        'vis': ['umap-learn', 'matplotlib']
    },
    packages=[p for p in find_packages() if p.startswith('torchSTC')],
    # package_data={'torchclust': ['py.typed']},
    include_package_data=True,
    license='MIT License',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Topic :: Text :: Short Text',
        'Topic :: Scientific/Engineering :: Artificial Intelligence, Deep Learning, Representation Learning, Clustering',
    ],
)