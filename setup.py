import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tinyBenchmarks",
    version="1.0.0",
    author="Felipe Maia Polo / Lucas Weber",
    author_email="felipemaiapolo@gmail.com",
    description="tinyBenchmarks: evaluating LLMs with fewer examples",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/felipemaiapolo/tinyBenchmarks',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy','scipy','requests'],
) 
