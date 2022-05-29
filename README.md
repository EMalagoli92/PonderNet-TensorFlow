# PonderNet-TensorFlow

This is an Unofficial Implementation of the paper: [PonderNet: Learning to Ponder](https://arxiv.org/abs/2107.05407) in TensorFlow.

Official PyTorch Implementation: [Link](https://nn.labml.ai/adaptive_computation/ponder_net/index.html)

PonderNet adapts the computation based on the input. It changes the number of steps to take on a recurrent network based on the input. PonderNet learns this with end-to-end gradient descent.

## Installation
Installing necessary packages: `pip install -r requirements.txt`

## Citations
```bibtex
@misc{banino2021pondernet,
      title={PonderNet: Learning to Ponder}, 
      author={Andrea Banino and Jan Balaguer and Charles Blundell},
      year={2021},
      eprint={2107.05407},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## License

[MIT](https://github.com/EMalagoli92/PonderNet-TensorFlow/blob/main/LICENSE)
