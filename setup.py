from setuptools import setup, find_packages

long_description = open("Readme.md").read()

setup(
    name='trickster',
    version='0.1.0',
    packages=find_packages(),
    url='https://github.com/csxeba/trickster.git',
    license='MIT',
    author='csxeba',
    author_email='csxeba@gmail.com',
    description='Deep Reinforcement Learning with TensorFlow 2',
    long_description=long_description,
    long_description_content_type='text/markdown'
)
