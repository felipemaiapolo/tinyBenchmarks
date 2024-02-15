# tinyBenchmarks tutorials

This repository contains Jupyter notebooks that guide you through the process of training your own IRT models, finding anchor points, and estimating the performance of Large Language Models (LLMs) using Python 3.11.7.

## Getting Started

To get started with these tutorials, you need to first clone this repository to your local machine. Use the following command in your terminal:

```shell
git clone https://github.com/felipemaiapolo/tinyBenchmarks.git
```

Once the repository is cloned, navigate to the repository's directory:

```shell
cd tinyBenchmarks/tutorials
```

### Prerequisites

Ensure that you have Python 3.11.7 installed on your machine. You can download it from the [official Python website](https://www.python.org/downloads/).

After ensuring you have the correct Python version, install all the required packages listed in `requirements.txt` using `pip`. Run the following command:

```shell
pip install -r requirements.txt
```

If you are running `training_irt.ipynb`, install `py_irt` with our modifications. Start installing [Poetry](https://python-poetry.org/docs/#installation) and then

```shell
git clone https://github.com/felipemaiapolo/py-irt.git
cd py-irt
poetry install
```


## Tutorials Overview

This repository contains three Jupyter notebooks that serve as tutorials:

1. **Training IRT Models (`training_irt.ipynb`):**
   - This notebook demonstrates how to train your own Item Response Theory (IRT) models. It covers the setup, training process, and evaluation of the models.

2. **Finding Anchor Points (`anchor_points.ipynb`):**
   - In this tutorial, we show how to identify anchor points from your training set. These anchor points are crucial for estimating the performance of new models on the test set.

3. **Estimating LLM Performance (`estimating_performance.ipynb`):**
   - This notebook guides you through the process of obtaining performance estimates for LLMs by combining the concepts of anchor points and IRT.

### Important Notes

- The `.py` files included in this repository are called inside the Jupyter notebooks and are not intended to be run separately. Make sure to follow the notebooks for the full tutorial experience.

## Contribution

Feel free to fork this repository and submit pull requests to contribute to this project. If you encounter any issues or have suggestions, please open an issue in this repository.

## License

This project is licensed under the MIT License - see the LICENSE file for details.


