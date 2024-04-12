# *tinyBenchmarks*: evaluating LLMs with fewer examples

Welcome to the [*tinyBenchmarks* GitHub repository](https://github.com/felipemaiapolo/tinyBenchmarks)! Here you will find more information about tiny datasets, how to estimate Large Language Model (LLM) performance using them using our Python package, a [demo](https://github.com/felipemaiapolo/tinyBenchmarks/blob/main/tinyBenchmarks_MMLU_demo.ipynb) in which we test our methods in MMLU, and [tutorials](https://github.com/felipemaiapolo/tinyBenchmarks/tree/main/tutorials) on how to obtain your own tiny datasets and make cheap model evaluation using the ideas presented in 

[Maia Polo, Felipe, Lucas Weber, Leshem Choshen, Yuekai Sun, Gongjun Xu, and Mikhail Yurochkin. "tinyBenchmarks: evaluating LLMs with fewer examples." arXiv preprint arXiv:2402.14992 (2024)](https://arxiv.org/abs/2402.14992) 


**Table of contents**
1. [ Datasets ](#1)
2. [ Estimating the performance of a new LLM using our package  ](#2)
3. [ Performance  ](#3)
4. [ Computing performances for all scenarios at once ](#4)
5. [ Citing ](#5)
   

--------------

<a name="1"></a>
## Datasets

Please check our [HuggingFace community](https://huggingface.co/tinyBenchmarks) with tiny datasets, each one containing 100 examples. In that collection, you will find tiny versions of 
- From the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard): TruthfulQA, GSM8K, Winogrande, ARC, HellaSwag, and MMLU;
- From [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval): AlpacaEval 2.0;
- From [HELM Lite](https://crfm.stanford.edu/helm/lite): to be added.

All the datasets were obtained from our work "*tinyBenchmarks*: evaluating LLMs with fewer examples" by
1. Finding anchor points using IRT embeddings;
2. Choosing, over the five used random seeds, the version of the dataset that performs better on average in the test set considering the `IRT` estimate (only using anchor points). We consider the random split setup when building the tiny datasets. For HELM Lite and AlpacaEval 2.0, in which we use K-fold cross-validation in the paper, we consider the first fold as the test set.

<a name="2"></a>
## Estimating the performance of a new LLM using our package

You can install our package by running the following commands on the terminal

``` :sh
$ pip install git+https://github.com/felipemaiapolo/tinyBenchmarks
```

Now, you can estimate the performance of a specific LLM on any tiny dataset or benchmark in our [HuggingFace community](https://huggingface.co/tinyBenchmarks) using the code below. If you want to evaluate a new LLM on the whole Open LLM Leaderboard or HELM Lite benchmarks, please estimate the performance of individual scenarios separately and then properly combine the scores. For example, for the Open LLM Leaderboard, the final score should be the average score across scenarios. For all scenarios the dimension of `y` below should be 100.

```python
import numpy as np
import tinyBenchmarks as tb

### Parameters
benchmark = 'mmlu' # choose from possible benchmarks in (HELM scenarios will be added)
                   # ['mmlu','alpaca','truthfulqa',
                   #  'gsm8k', 'winogrande', 'arc', 'hellaswag']

y = np.random.binomial(1,.5, 100) # dummy data (unidimensional numpy array)
                                 

### Evaluation
tb.evaluate(y, benchmark)
```

    {'mmlu': {'irt': 0.5377958382081949,
      'pirt': 0.5393624918280722,
      'gpirt': 0.5385791650181335}}

**Disclaimer**
For convenience, we advise users (see datasets cards on our [HuggingFace community](https://huggingface.co/tinyBenchmarks)) to use a more modern version of [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), which can create divergences with the numbers presented in the leaderboard. We expect *tinyBenchmarks* to still work very well in that case, but use our tools at your own risk.


<a name="3"></a>
## Performance 

We report in the following tables the average estimation error in the test set (using data from the paper) and standard deviation across LLMs.

#### Open LLM Leaderboard

|| IRT | p-IRT | gp-IRT |
|--|--|--|--|
| TruthfulQA | 0.013 (0.010) | 0.010 (0.009) | 0.011 (0.009) |
| GSM8K | 0.022 (0.017) | 0.029 (0.022) | 0.020 (0.017) |
| Winogrande | 0.022 (0.017) | 0.016 (0.014) | 0.015 (0.013) |
| ARC | 0.022 (0.018) | 0.017 (0.014) | 0.017 (0.013) |
| HellaSwag | 0.013 (0.016) | 0.015 (0.012) | 0.015 (0.012) |
| MMLU | 0.024 (0.017) | 0.016 (0.015) | 0.016 (0.015) |


#### AlpacaEval 2.0
|| IRT | p-IRT | gp-IRT |
|--|--|--|--|
| AlpacaEval 2.0 | 0.012 (0.015) | 0.020 (0.021) | 0.016 (0.016) |

#### Helm Lite

As we conduct 11-fold cross-validation (CV) with HELM Lite, the test set only contains three models, making the error estimate not meaningful. Please check Appendix D.2 of our paper for CV error estimates.

<a name="4"></a>
## Computing performances for all scenarios at once

In our paper, we comment that is beneficial to estimate the ability parameter $\theta$ for a LLM of interest using data from all scenarios. Moreover, this is what we do for the Open LLM Leaderboard in our tutorials. Unfortunately, the current version of the tinyBenchmarks datasets on HuggingFace does not permit this feature, because tinyMMLU was generated separately. In the next version of tinyBenchmarks, we intend to unify tinyMMLU with other tiny datasets to enable this feature.
    
<a name="5"></a>
## Citing
If you use any material from this repository in your academic work, please cite

    @article{polo2024tinybenchmarks,
      title={tinyBenchmarks: evaluating LLMs with fewer examples},
      author={Maia Polo, Felipe and Weber, Lucas and Choshen, Leshem and Sun, Yuekai and Xu, Gongjun and Yurochkin, Mikhail},
      journal={arXiv preprint arXiv:2402.14992},
      year={2024}
    }

## Contribution and License

Feel free to fork this repository and submit pull requests to contribute to this project. If you encounter any issues or have suggestions, please open an issue in this repository.

This project is licensed under the MIT License - see the LICENSE file for details.
