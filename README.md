# Capsule

A Tensorflow Implementation of Hinton's __[Matrix Capsules with EM Routing](https://openreview.net/pdf?id=HJWLfGWRb)__.

## Quick Start

```
$ git clone https://github.com/gyang274/capsulesEM.git && cd capsulesEM

$ cd src

$ python train.py

# open a new terminal (ctrl + alt + t)

$ python tests.py
```

## TODO

1. Run train.py/tests.py on MNIST.

1. Add train.py/tests.py on smallNORB.

## Questions

1. $$\lambda$$ schedule is never mentioned in paper.

1. The place encode in lower level and rate encode in higher level is not discussed, other than a coordinate addition in last layer.

## GitHub Page

This [`gh-pages`](https://gyang274.github.io/capsulesEM/) includes all notes.

## GitHub Repository

This [github repository](https://github.com/gyang274/capsulesEM) includes all source codes.
