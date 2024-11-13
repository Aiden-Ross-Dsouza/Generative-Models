# Generative Adversarial Networks

Generative Adversarial Networks (GANs) marked a transformative shift in generative modeling, introducing a novel framework where two neural networks, the generator and the discriminator, engage in a dynamic adversarial process. Prior to the advent of GANs, generative tasks were predominantly tackled using statistical models or rule-based approaches, which often struggled with scalability, complexity, and generating high-quality outputs. GANs, by contrast, harness the power of deep learning to model intricate data distributions, enabling the generation of incredibly realistic images, audio, and text. This section delves into implementations of seminal papers that introduced breakthrough GAN architectures, illustrating their far-reaching impact on the generative landscape and ongoing research advancements.

All the models were trained on the [CIFAR10](https://www.kaggle.com/c/cifar-10/) dataset which consists of 60,000 32x32 color images containing one of 10 object classes, with 6000 images per class.

Below is a table addressing some common data and optimization related parameters.

| Parameter                |          DCGAN        |              WGAN             |
| ------------------------ |:---------------------:|:-----------------------------:|
| Latent Dimension (z)     |           100         |              100              |
| Batch Size               |           128         |               64              |
| Learning Rate            |           2e-4        |              5e-5             |
| Discriminator Iterations |            1          |                5              |
| Loss Function            |   Cross Entropy Loss  |       Wasserstein loss        |
| Optimizer                | Adam(β1=0.5, β2=0.999)|             RMSprop           |
| Noise Distribution       |        Gaussian       |             Gaussian          |
| Activation Functions     |Generator: ReLU (intermediate layers), Tanh (output); Discriminator: Leaky ReLU (hidden layers), Sigmoid (output layer)|Generator: ReLU (intermediate layers), Tanh (output); Discriminator: Leaky ReLU (hidden layers)|

## Architectures

### 1. [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://github.com/Aiden-Ross-Dsouza/Generative-Models/blob/a2d87c38e41ebc374f4e9c9a6deda99fb91a8384/Generative%20Adversarial%20Networks/notebooks/DCGAN.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OfWn0-nDdxi_66ZS-qPD4Eii1A2Jd1Rt?usp=sharing)
This paper advances GANs by incorporating deep convolutional architectures for both the generator and discriminator. Key innovations include the use of strided convolutions instead of pooling, batch normalization to stabilize training, and Leaky ReLU activations to address vanishing gradients. The DCGAN framework effectively generates realistic images and learns useful representations, although challenges like mode collapse and training stability persist.

### 2. [Wasserstein GAN](https://github.com/Aiden-Ross-Dsouza/Generative-Models/blob/a2d87c38e41ebc374f4e9c9a6deda99fb91a8384/Generative%20Adversarial%20Networks/notebooks/WGAN.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Fz-QJKFOcdCCX61LR05dTzQYZuO2IQ59?usp=sharing)
This paper introduces the Wasserstein distance as an alternative to the Jensen-Shannon divergence used in traditional GANs, improving training stability and convergence. It employs the Earth Mover’s Distance (EMD) to measure the discrepancy between real and generated data distributions, which addresses issues like mode collapse and vanishing gradients. The paper also proposes the use of a critic network with weight clipping to enforce the Lipschitz constraint required by the Wasserstein distance.

## Summary
Below is a table, summarising the number of parameters and the BLEU scores achieved by each architecture.

| Architecture                        | No. of Trainable Parameters Generator | No. of Trainable Parameters Discriminator | FID Score  |
| ----------------------------------- |:-------------------------------------:|:-----------------------------------------:|:----------:|
| DCGAN                               |               12,658,435              |                  2,765,633                |   270.69   |

<ins>**Note:**</ins>
1. The above FID scores may vary slightly upon training the models (even with fixed SEED).

### Reference(s):
* [PyTorch GAN by Aladdin Persson](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/558557c7989f0b10fee6e8d8f953d7269ae43d4f/ML/Pytorch/GANs)
* [Blog on DCGAN by Chris](https://medium.com/@kyang3200/deep-learning-dcgan-deep-convolutional-generative-adversarial-network-882624fdefe3)
