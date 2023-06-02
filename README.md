[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
## Sketches for Time-Dependant Machine Learning

This repository contains source code supporting a series of experiments on how streaming (in particular sketches) 
techniques can aid in the modelling of time series. The results can be found in the 
[paper](https://arxiv.org/abs/2108.11923).

**Installation**
```
pip install -i https://test.pypi.org/simple/ skcm
```

These are the sketches, deep learning models and functionalities included:

- **Exponential Histogram**, capable of keeping track of the following statistics:
    - Binary Counter [[1]](#1)
    - Sum
        - Positive integers [[1]](#1)
        - Extension over positive real numbers (own)
    - Mean (positive real and real, the former more space efficient)
    - Variance (real) [[2]](#2)

- **EHRNN**
    A modified Elmann Network (RNN) that efficiently keeps track of hidden state statistics across multiple time resolutions via Exponential Histograms.        Implemented in PyTorch.

- **Other utils**
    - DataFrame sketch windower: returns a ``pandas.DataFrame`` with the results of applying a 
    summarizing sketch over a/some windows (Exponential Histograms). Useful for obtaining descriptive statistics and summarization of data trends across time resolutions.
    - Format converters
        - ``csv`` to ``arff`` and viceversa
        - ``pandas.DataFrame`` to ``arff``  
        (``arff`` is a data format used by ML frameworks such as [Weka](https://www.cs.waikato.ac.nz/ml/weka/) and 
        [MOA](https://moa.cms.waikato.ac.nz/))

## Citations

If you use this code in your research / application, please cite the current pre-print.

``` 
@misc{antonanzas2021sketches,
      title={Sketches for Time-Dependent Machine Learning}, 
      author={Jesus Antonanzas and Marta Arias and Albert Bifet},
      year={2021},
      eprint={2108.11923},
      archivePrefix={arXiv},
      URL={https://arxiv.org/abs/2108.11923},
      primaryClass={cs.LG}
} 
```

## References

Ideas from these references have been used in the software:

<a id="1">[1]</a> 
M. Datar et al. (2002). 
Maintaining Stream Statistics over Sliding Windows. 
Society for Industrial and Applied Mathematics, 31(6), 1794-1813.

<a id="1">[2]</a> 
B. Babcock et al. (2003). 
Maintaining Variance and k-Medians over Data Stream Windows. 
Proceedings of the ACM SIGACT-SIGMOD-SIGART Symposium on Principles of Database Systems, 22, 234-243.
