from setuptools import setup, find_packages

# Read the contents of requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='eswa_difficulty',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
    author='Pedro Ramoneda',
    author_email='pedro.ramoneda@upf.edu',
    description='Predicting difficulty from musicxml piano scores.',
    long_description=open('readme.md').read(),
    long_description_content_type='text/markdown',
    license='AGPL-3.0',
    keywords='predict estimating difficulty musicxml piano scores',
    url='https://github.com/pramoneda/difficulty-prediction-CIPI',
    package_data={
        'eswa_difficulty.piano_fingering': ['models/*.pth'],
        'eswa_difficulty': ['eswa_models/*.pth'],
    },
    classifiers=[
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.10',
    ],
)
