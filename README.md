# tinyBenchmarks: evaluating LLMs with fewer examples

This package is based on the ideas presented in

[Citation goes here](https://arxiv.org). 

Please cite us in the following way

    @article{abcde,
      title={tinyBenchmarks: evaluating LLMs with fewer examples},
      author={our names},
      journal={journal},
      pages={pages},
      year={year},
      publisher={publisher}
    }

--------------


## Installing package 

You can install our package by running the following commands on the terminal

``` :sh
$ pip install git+https://github.com/felipemaiapolo/tinyBenchmarks
```


## Estimating the performance of a new LLM

```python
import numpy as np
import tinyBenchmarks as tb

### Parameters
benchmark = 'lb' # choose from possible benchmarks in ['lb', 'mmlu', 'helm_lite', 'alpaca']
y = np.random.binomial(1,.5, 600) # dummy data (unidimensional numpy array. In this example, y has dimension 600 because we observe 100 examples from each scenario)

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

