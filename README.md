# **Deep Convolutional GANs for Image Generation**

## **Project Overview**

This project demonstrates the implementation of Deep Convolutional Generative Adversarial Networks (DCGANs) to generate realistic images. We focus on two different datasets:
1. **Face Images**: Using a dataset of faces to generate high-quality human face images.<img width="521" alt="image" src="https://github.com/user-attachments/assets/3c8cc752-2e1f-4164-81dd-3cdf3ab6a697">

2. **Fashion MNIST**: Using the Fashion MNIST dataset to generate clothing item images.<img width="653" alt="image" src="https://github.com/user-attachments/assets/27c40535-a62e-4748-bf63-63ac779f37fd">


Generative Adversarial Networks (GANs) are a powerful class of neural networks used for generative modeling. DCGAN is a variant of GANs that leverages convolutional layers to generate high-quality images by capturing complex data distributions. This project aims to provide a step-by-step guide on training DCGANs, visualizing results, and understanding the architecture of the generator and discriminator networks.

## **Table of Contents**
- [Project Structure](#project-structure)
- [DCGAN Architecture](#dcgan-architecture)
  - [Generator](#generator)
  - [Discriminator](#discriminator)
- [Techniques Used](#techniques-used)
  - [Convolutional Layers](#convolutional-layers)
  - [Batch Normalization](#batch-normalization)
  - [Activation Functions](#activation-functions)
- [Results](#results)
- [Dependencies](#dependencies)
- [Usage](#usage)

## **Project Structure**
The repository contains the following files:
- `Dcgan_faces_tutorial.ipynb`: Jupyter notebook detailing the implementation of DCGAN for generating face images.
- `DCGAN_FMNIST.ipynb`: Jupyter notebook detailing the implementation of DCGAN for generating images of clothing items from the Fashion MNIST dataset.

## **DCGAN Architecture**
DCGANs consist of two primary components that work in tandem: the Generator and the Discriminator. The Generator aims to create realistic images, while the Discriminator tries to distinguish between real and generated images. Both networks are convolutional in nature, making DCGANs highly effective at handling image data.

### **Generator**
The Generator is a neural network designed to create images from random noise. It consists of several key components:
- **Transpose Convolution Layers**: Also known as deconvolutional layers, these layers help upsample the random noise to a higher-dimensional output, ultimately forming the generated image.
- **Batch Normalization**: Applied after each layer to stabilize learning and allow for faster convergence.
- **Activation Function**: LeakyReLU is used for all layers except the output, where a Tanh function is employed to map pixel values between -1 and 1.

The architecture of the Generator can be summarized as follows:
1. Input: Random noise vector (typically sampled from a normal distribution).
2. Series of transpose convolutional layers with increasing resolution.
3. Batch normalization and LeakyReLU applied after each layer.
4. Output layer with Tanh activation to produce an image.
<img width="736" alt="image" src="https://github.com/user-attachments/assets/89a6b84b-68ed-406d-b90a-755e46e4cf55">

### **Discriminator**
The Discriminator is a binary classifier that attempts to distinguish between real images from the dataset and fake images produced by the Generator. It uses:
- **Convolutional Layers**: These layers extract features from the input images. Strided convolutions are used to downsample the input.
- **Batch Normalization**: Similar to the Generator, this helps in stabilizing training.
- **Activation Function**: LeakyReLU is used for all layers to prevent vanishing gradient issues, and a Sigmoid function is used in the final layer to output a probability score.

The Discriminator is structured as follows:
1. Input: Image (either real or generated).
2. Series of convolutional layers with decreasing resolution.
3. Batch normalization and LeakyReLU applied after each layer.
4. Output layer with Sigmoid activation to produce a probability.

## **Techniques Used**
### **Convolutional Layers**
- DCGANs rely heavily on convolutional and transpose convolutional layers to downsample and upsample images, respectively.
- The use of **stride** in convolutional layers helps reduce the dimensionality, while **stride** in transpose convolutions increases dimensionality.

### **Batch Normalization**
- Batch normalization is used extensively in both the Generator and Discriminator to stabilize learning.
- It normalizes the output of each layer, reducing the chances of exploding/vanishing gradients, which makes the network training more stable and faster.

### **Activation Functions**
- **LeakyReLU**: Used in both Generator and Discriminator, providing a small gradient for negative values to address the dying ReLU problem.
- **Tanh**: Used in the Generator’s output layer, ensuring that pixel values are scaled between -1 and 1.
- **Sigmoid**: Used in the Discriminator’s final layer to output a probability score indicating real or fake.

## **Results**
The DCGANs were trained on:
- **Face Dataset**: Successfully generated realistic face images after sufficient training. The model learned the distribution of features, like facial structure and expressions, to create new and diverse faces.
- **Fashion MNIST**: The Generator learned to produce various types of clothing items such as shirts, pants, and shoes. The generated images closely resemble the original dataset after a few epochs.
<img width="343" alt="image" src="https://github.com/user-attachments/assets/57a66baf-d6b0-40df-a018-6dc34b2f387d">


Visualizations in the notebooks illustrate the progress of training over epochs, showing how the generated images evolve from random noise to recognizable outputs.

## **Dependencies**
To run the project, you need the following Python libraries:
- `TensorFlow` or `PyTorch` (as per the provided notebooks)
- `NumPy`
- `Matplotlib`
- `Pandas`
- `Seaborn`

These dependencies can be installed using:
```sh
pip install tensorflow numpy matplotlib pandas seaborn
```

### **Note**:
- Training GANs is computationally intensive. It is recommended to use a GPU for faster training.

## **Conclusion**
DCGANs offer a powerful way to model complex data distributions, particularly for generating realistic images. This project covers both theoretical and practical aspects of implementing DCGANs for two different types of datasets. The generated results highlight the effectiveness of deep convolutional layers in capturing intricate details of image data.


---

Let me know if you'd like any changes or additions to this README file.
