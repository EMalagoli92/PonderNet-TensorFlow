# PonderNet - TensorFlow

This is an Unofficial Implementation of the paper: [PonderNet: Learning to Ponder](https://arxiv.org/abs/2107.05407) in TensorFlow 2.x.


In standard neural networks the amount of computation used grows with the size of the inputs, but not with the complexity of the problem being learnt. To overcome this limitation we introduce PonderNet, a new algorithm that learns to adapt the amount of computation based on the complexity of the problem at hand. PonderNet learns end-to-end the number of computational steps to achieve an effective compromise between training prediction accuracy, computational cost and generalization. On a complex synthetic problem, PonderNet dramatically improves performance over previous adaptive computation methods and additionally succeeds at extrapolation tests where traditional neural networks fail. Also, our method matched the current state of the art results on a real world question and answering dataset, but using less compute. Finally, PonderNet reached state of the art results on a complex task designed to test the reasoning capabilities of neural networks.

## Installation
Installing necessary packages: `pip install -r requirements.txt`

## Experiment  on Parity Task
Train a PonderNet on Parity Task: `python experiment.py`

![My Remote Image](https://production-media.paperswithcode.com/methods/99d96967-3912-44fe-9930-b0ea32ed42a3.png)



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

Official PyTorch Implementation: [Link](https://nn.labml.ai/adaptive_computation/ponder_net/index.html)

## License

[MIT](https://github.com/EMalagoli92/PonderNet-TensorFlow/blob/main/LICENSE)
