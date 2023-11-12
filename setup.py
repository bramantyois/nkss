from setuptools import setup, find_packages

setup(
    name='nkss-simulator',
    version='0.1',
    packages=find_packages(),  # Automatically find and include all packages
    install_requires=[
        'numpy',
        'tqdm'
    ],
)
