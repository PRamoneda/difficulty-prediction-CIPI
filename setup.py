from setuptools import setup, find_packages

# Read the contents of requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='difficulty-prediction-CIPI',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
    author='Your Name',
    author_email='pedro.ramoneda@upf.edu',
    description='A short description of the project.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='AGPL-3.0',
    keywords='some keywords to describe your project',
    url='https://github.com/yourusername/mypackage',
    classifiers=[
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
