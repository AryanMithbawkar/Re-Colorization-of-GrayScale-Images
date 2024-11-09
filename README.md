# Re-Colorization of GrayScale Images

This repository provides a deep learning model for colorizing grayscale images using a convolutional autoencoder. It aims to predict realistic colors for grayscale images, particularly landscapes, by leveraging a deep neural network trained on thousands of images.

## Overview
The model is trained on pairs of grayscale and color images, learning to map grayscale pixel values to their respective colors. This project showcases the architecture and training of the model, with visualizations of the colorization results.

### Project Structure
+ Data Preparation: Images are loaded, resized, and normalized for training.
+ Model Architecture: An encoder-decoder architecture based on convolutional layers is used for learning colorization patterns.
+ Training: The model is trained on a dataset of images with mean squared error as the loss function.
+ Evaluation: The colorized results are compared against the original color images using PSNR and SSIM metrics.
## Model Architecture
The model architecture consists of an encoder for downsampling grayscale images and a decoder for reconstructing the colorized image. Key components include:

1. Encoder:
  + Multiple convolutional layers for feature extraction with downsampling.
  + Batch normalization and Leaky ReLU activation.
2. Decoder:
  + Transposed convolution layers for upsampling.
  + Concatenation layers to retain spatial information from the encoder layers.
  + A final convolution layer with a sigmoid activation to output the colorized image.
Below is a sample architecture summary:

```yaml
Layer (type)           Output Shape           Param #
=================================================================
conv2d                 (None, 80, 80, 128)     3584
...
conv2d_transpose       (None, 160, 160, 3)     6915
=================================================================
Total params: 9,602,856
Trainable params: 9,600,296
Non-trainable params: 2,560
```
## Results
**Sample Outputs** 
The model's predictions are shown below. Each row represents:
+ Training Loss and Mae
  ![Training Loss and Mae](output.png)
+ Input Grayscale Image Predicted Colorized Image Ground Truth Color Image
  ![Predicted](output1.png)

## Evaluation
To measure the effectiveness of the colorization, we use:
+ Peak Signal-to-Noise Ratio (PSNR): Measures the quality of the predicted colorized image: PSNR: 22.81.
+ Structural Similarity Index (SSIM): Quantifies image similarity between the colorized and ground truth images: SSIM: 0.95.

## Installation and Usage
1. Clone the repository:
  ```bash
  git clone https://github.com/AryanMithbawkar/Re-Colorization-of-GrayScale-Images.git
  ```

## Gradio Interface
This project includes a Gradio interface for uploading grayscale images and receiving colorized outputs.

```python
import gradio as gr 
gr_interface = gr.Interface(
    fn=colorize_image,
    inputs=gr.Image(image_mode='L', label="Upload a grayscale image"),
    outputs=gr.Image(label="Colorized Image"),
    title="Image Recolorization",
    description="Upload a grayscale image to see it colorized by the trained model."
)
```
+ Gradio interface
  ![Ginterface](image.png)
## Acknowledgments
+ [Kaggle Landscape Image Colorization Dataset](https://github.com/AryanMithbawkar/Re-Colorization-of-GrayScale-Images)
+ Various open-source libraries: TensorFlow, Keras, OpenCV, Gradio.
