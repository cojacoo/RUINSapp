import importlib
from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


def requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()


def version():
    ruins = importlib.import_module('ruins')
    return ruins.__version__


setup(
    name='ruins-app',
    description='Climate change and uncertainty explorer',
    long_description=readme(),
    version=version(),
    author='Conrad Jackisch, Mirko Mälicke',
    author_email='Conrad.Jackisch@tbt.tu-freiberg.de, mirko@hydrocode.de',
    install_requires=requirements(),
    packages=find_packages()
)

