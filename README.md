[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
## Streaming techniques for learning 

This repository contains source code supporting a series of experiments on how streaming (in particular sketches) 
techniques can aid in the modelling of time series.

These are the sketches and functionalities included:

- **Exponential Histogram**, with the following uses:
    - Binary Counter [[1]](#1)
    - Sum
        - Positive integers [[1]](#1)
        - Extension over positive real numbers (own)
    - Mean (positive real and real, the former more space efficient)
    - Variance (real) [[2]](#2)

- **Other utils**
    - DataFrame sketch windower: returns a ``pandas.DataFrame`` with the results of applying a 
    summarizing sketch over a/some windows (only Exponential Histogram for now).
    - Format converters
        - ``csv`` to ``arff`` and viceversa
        - ``pandas.DataFrame`` to ``arff``  
        (``arff`` is a data format used by ML frameworks such as [Weka](https://www.cs.waikato.ac.nz/ml/weka/) and 
        [MOA](https://moa.cms.waikato.ac.nz/))
        
## References
<a id="1">[1]</a> 
M. Datar et al. (2002). 
Maintaining Stream Statistics over Sliding Windows. 
Society for Industrial and Applied Mathematics, 31(6), 1794-1813.

<a id="1">[2]</a> 
B. Babcock et al. (2003). 
Maintaining Variance and k-Medians over Data Stream Windows. 
Proceedings of the ACM SIGACT-SIGMOD-SIGART Symposium on Principles of Database Systems, 22, 234-243.