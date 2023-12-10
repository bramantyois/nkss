from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read() 

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='nkss_simulator',
    version='0.0.1',
    author='Bramantyo Supriyatno',
    url='https://github.com/bramantyois/nkss',
    packages=find_packages(),  # Automatically find and include all packages
    install_requires=requirements,
    long_description=long_description,
)
