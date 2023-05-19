# Sartorius Cell Instance Segementation

## Detect single neuronal cells in microscopy images using Unet

![Alt text](/images/train_truth.png)

Detail description on my blog(Korean):
https://gomduribo.tistory.com/39

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

The RLE-type mask was decoded with the segmentation map of 520x704 and entered as a label value into the model. And the decoded segmentation map is as follows.
(source: https://www.kaggle.com/code/julian3833/sartorius-starter-torch-mask-r-cnn-lb-0-273)


![Alt text](/images/rledecode.png)

## Train
I conducted the learning on the Lenovo laptop with the Nvidia RTX 1050Ti 4GB model that I am currently using.
I was planned 20 epochs, but it didn't seem to converge, so I stopped learning in the middle.

<pre>
<code>
is_cuda = True
DEVICE = torch.device('cuda' if torch.cuda.is_available and is_cuda else 'cpu')
# DEVICE = torch.device('cpu')

BATCH_SIZE = 1
LEARNING_RATE = 1e-3
EPOCHS = 20
NUM_CLASSES = 2

train_data = CT_Dataset(phase='train', transformer=transformer)
trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle=True)
model = Unet()
model = model.to(DEVICE)
criterion = Unet_loss(num_classes=NUM_CLASSES)
optimizer = torch.optim.SGD(model.parameters(), lr= LEARNING_RATE, momentum=0.9)
</code>
</pre>


When I was learning on the company laptop with the Nvidia RTX 1050Ti 4GB model, the memory capacity (4GB..) was small, so I couldn't proceed with batch size 8 and proceeded with 1.

## Inference Result
Less than 5 Epoch passed during learning, so we found a trend that diverges past convergence, stopped learning, and checked the inference results.

![Alt text](/images/result.png)
Test image (left), inference result during 3Epoch learning (middle), inference result during 5Epoch learning (right)

Perhaps because there was little learning data, it was confirmed that the model who learned 3Epoch was better at reasoning than the model who learned 5Epoch.

![Alt text](/images/result2.png)
