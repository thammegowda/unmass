import setuptools
from pathlib import Path
import re

long_description = Path('README.md').read_text(encoding='utf-8', errors='ignore')

vpat = re.compile(r"""__version__\s*=\s*['"]([^'"]*)['"]""")
__version__ = None
for line in Path('unmass/__init__.py').read_text().splitlines():
    line = line.strip()
    if vpat.match(line):
        __version__ = vpat.match(line)[1]

assert __version__, 'Could not find __version__ in __init__.py'

setuptools.setup(
    name='unmass',
    version=__version__,
    author="Thamme Gowda",
    author_email="tgowdan@gmail.com",
    description="UNMASS - Unsupervised NMT with Masked Sequence-to-Sequence training",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/thammegowda/unmass",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        'License :: OSI Approved :: MIT License',
        "Operating System :: OS Independent",
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    platforms=['any'],
    install_requires=[
        'ruamel.yaml >= 0.16.10',
        'sacrebleu >= 1.4.6',
        'sacremoses >= 0.0.43',
        'nlcodec >= 0.2.3',
        'torch >= 1.4'
    ],
    python_requires='>=3.7',
    scripts=['scripts/unmass-prep'],
    entry_points={
        'console_scripts': [
            'unmass-train=unmass.train:cli',
            'unmass-translate=unmass.translate:cli',
        ],
    }
)
