# Capsule

A Tensorflow Implementation of Hinton's `Matrix Capsules with EM Routing.`

Paper is under open review.

## MNIST

```
$ python train.py

# open a new terminal (ctrl + alt + t)

$ python tests.py
```

## TODO:

1. Add documentation of the implementation.

2. Add train/test on smallNORB


## Questions:

1. $\lambda$ schedule is never mentioned in paper.

2. $\beta_a$ and $\beta_v$: should be one global parameter, or one copy each convolutional layer.

