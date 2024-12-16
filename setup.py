from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="transit",
    version='0.1',
    description="A package transit experiments",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Ivan Oleksiyuk",
    packages=find_packages(exclude=["tests"]),
    license="MIT",
    dependency_links=[],
)
