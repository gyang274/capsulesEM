# Capsule

A Tensorflow Implementation of Hinton's __[Matrix Capsules with EM Routing](https://openreview.net/pdf?id=HJWLfGWRb)__.

## Quick Start

```
$ git clone https://github.com/gyang274/capsulesEM.git

$ cd src

$ python train.py

# open a new terminal (ctrl + alt + t)

$ python tests.py
```

## TODO

1. Add documentation of understanding, implementation, and optimization.

2. Run train.py/tests.py on MNIST.

3. Add train.py/tests.py on smallNORB.

## Questions

1. $\lambda$ schedule is never mentioned in paper.

2. $\beta_a$ and $\beta_v$: should be one global parameter, or one copy each convolutional layer.

## GitHub Page

This [`gh-pages`](https://gyang274.github.io/capsulesEM/) includes all notes.

## GitHub Repository

This [github repository](https://github.com/gyang274/capsulesEM) includes all source codes.
