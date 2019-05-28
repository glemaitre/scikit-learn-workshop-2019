# Annual workshop of the scikit-learn foundation @ Inria
## 28 May 2019

### New feaures tutorial

<a href="https://mybinder.org/v2/gh/glemaitre/scikit-learn-workshop-2019/master">
  <img src="https://mybinder.org/badge.svg" />
</a>

This tutorial requires Python 3.5+ as well as,
 - scikit-learn >=0.21.2
 - matplotlib
 - pandas

The tutorial notebook can be found in 
[new-features-tutorial/new-features-tutorial.ipynb](https://github.com/glemaitre/scikit-learn-workshop-2019/blob/master/new-features-tutorial/new-features-tutorial.ipynb).

### Interpretability tutorial

This interpretability tutorial uses some features from scikit-learn which are under-development:

* DataFrame handling with OpenML datasets:
  https://github.com/scikit-learn/scikit-learn/pull/13902
* Fast partial dependence plot for Gradient Boosting Decsision Trees:
  https://github.com/scikit-learn/scikit-learn/pull/13769
* Permutation feature importance:
  https://github.com/scikit-learn/scikit-learn/pull/13146

These features have been combined into a scikit-learn branch in the following
repository: https://github.com/glemaitre/scikit-learn/tree/workshop

You can refer to the following documentation to install scikit-learn from such
source:
https://scikit-learn.org/stable/developers/advanced_installation.html#install-bleeding-edge
