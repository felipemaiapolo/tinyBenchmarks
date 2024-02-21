# tinyBenchmarks: evaluating LLMs with fewer examples

Welcome to the tinyBenchmarks GitHub repository! Here you will find more information about tiny datasets, how to estimate LLM performance using them (using our Python package), and tutorials on how to obtain your tiny datasets and make cheap model evaluating using the ideas presented in

[reference goes here](https://arxiv.org). 

If you use any material from this repository in your academic work, please cite

    @article{abcde,
      title={tinyBenchmarks: evaluating LLMs with fewer examples},
      author={our names},
      journal={journal},
      pages={pages},
      year={year},
      publisher={publisher}
    }

--------------

## Datasets

Please check our [HuggingFace collection](https://huggingface.co/collections/felipemaiapolo/tinybenchmarks-65d40353d37914c4c8afc6e4) with tiny datasets, each one containing 100 examples. In that collection, you will find tiny versions of 
- From the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard): TruthfulQA, GSM8K, Winogrande, ARC, HellaSwag, and MMLU;
- From [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval): AlpacaEval 2.0;
- From [HELM Lite](https://crfm.stanford.edu/helm/lite): to be added.

All the datasets were obtained from our work "tinyBenchmarks: evaluating LLMs with fewer examples" by
1. Finding anchor points using IRT embeddings;
2. Choosing, over the five used random seeds, the version of the dataset that performs better on average in the test set considering the `IRT` estimate (only using anchor points). We consider the random split setup when building the tiny datasets.

## Estimating the performance of a new LLM using our package

You can install our package by running the following commands on the terminal

``` :sh
$ pip install git+https://github.com/felipemaiapolo/tinyBenchmarks
```

Now, you can estimate the performance of a specific LLM on any tiny dataset or benchmark in our [HuggingFace collection](https://huggingface.co/collections/felipemaiapolo/tinybenchmarks-65d40353d37914c4c8afc6e4).

```python
import numpy as np
import tinyBenchmarks as tb

### Parameters
benchmark = 'lb' # choose from possible benchmarks in
                 # ['lb','mmlu','alpaca','helm_lite','truthfulqa',
                 #  'gsm8k', 'winogrande', 'arc', 'hellaswag']

y = np.random.binomial(1,.5, 600) # dummy data (unidimensional numpy array)
                                  # In this example, y has dimension 600 because we
                                  # observe 100 examples from each Open LLM Leaderboard scenario)

### Evaluation
tb.evaluate(y, benchmark)
```

    {'harness_truthfulqa_mc_0': {'irt': 0.5483476132190942,
      'pirt': 0.5216756041366227,
      'gpirt': 0.5350116086778585},
     'gsm8k': {'irt': 0.5132676269901439,
      'pirt': 0.5328183759663551,
      'gpirt': 0.5230430014782494},
     'winogrande': {'irt': 0.4301499605367009,
      'pirt': 0.4792754277690377,
      'gpirt': 0.4547126941528693},
     'arc': {'irt': 0.5520477815699659,
      'pirt': 0.5066457168990404,
      'gpirt': 0.5293467492345032},
     'hellaswag': {'irt': 0.5338577972515436,
      'pirt': 0.5108037778592825,
      'gpirt': 0.5223307875554131},
     'mmlu': {'irt': 0.5377958382081949,
      'pirt': 0.5393624918280722,
      'gpirt': 0.5385791650181335}}

