# Sartorius Cell Instance Segementation

## Detect single neuronal cells in microscopy images using Unet

![Alt text](/images/train_truth.png)


This task provides a 3x520x704 cell image and ground truth as shown in Figure 1. It provides 606 images and masks as a Train dataset and 3 images as a Test dataset.

First, we decided to solve it by applying the basic Unet structure. It's been a long time since Unet came out, and although it's not a current SOTA model, I'm used to it, and I think it's a basic model in Segmentation, so I chose Unet. First, it was decided to solve it with Unet implemented based on the paper, and to improve performance by applying various techniques or tuning.

Unet Paper:https://arxiv.org/pdf/1505.04597.pdf

### Model implementaion

![Alt text](/images/unet.png)

I tried to implement it as much as possible as I could in the paper. In this paper, a 1x572x572 size image is input, and the background and the semantic segmentation of the cell part are outputted.

### Data handling

In the paper, Unet receives 1x572x572 size image data as input, but our original data is 3x520x704 size images in RGB format. Therefore, we received a 3x572x572 size input from the input unit and resized the image to 3x572x572.

It was resized correctly, and the result was confirmed as shown below.

![Alt text](/images/resuze.png)

The Kaggle task provides a mask corresponding to ground truth by a non-loss compression method in the form of run-length encoding (RLE).

![Alt text](/images/rle.png)
RLE compression(source: http://stoimen.com/2012/05/03/computer-algorithms-lossy-image-compression-with-run-length-encoding/)