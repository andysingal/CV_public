## Introduction
We present SegFormer, a simple, efficient yet powerful semantic segmentation framework which unifies Transformers with lightweight multilayer perception (MLP) decoders. 
SegFormer has two appealing features: 1) SegFormer comprises a novel hierarchically structured Transformer encoder which outputs multiscale features. It does not need positional encoding, 
thereby avoiding the interpolation of positional codes which leads to decreased performance when the testing resolution differs from training. 2) SegFormer avoids complex decoders. The proposed MLP 
decoder aggregates information from different layers, and thus combining both local attention and global attention to render powerful representations. We show that this simple and lightweight design is 
the key to efficient segmentation on Transformers. 

1. Hybrid transformer backbone
SegFormer leverages a hybrid transformer backbone to extract features from input images. This involves a convolutional layer to process the input image, followed by a transformer to capture the global context of the image.


2. Multi-Scale feature integration
To handle objects and features of varying scales in an image, SegFormer amalgamates multi-scale feature maps derived from different transformer layers. This multi-scale feature integration enables the model to recognize and accurately segment objects of different sizes and shapes.


3. MLA head
The Multi-Level Aggregation (MLA) head is a distinct component of SegFormer, which fuses feature maps from different levels, ensuring that the segmentation model can effectively utilize features from all scales. This is crucial for maintaining high-resolution details and recognizing small objects, enhancing the model's segmentation performance.

<img width="594" alt="Screenshot 2024-05-01 at 8 51 31 AM" src="https://github.com/andysingal/CV_public/assets/20493493/310cfb42-c9a2-457d-bf68-d39870feaa2e">

## Usage tips
SegFormer consists of a hierarchical Transformer encoder, and a lightweight all-MLP decoder head. SegformerModel is the hierarchical Transformer encoder (which in the paper is also referred to as Mix Transformer or MiT). SegformerForSemanticSegmentation adds the all-MLP decoder head on top to perform semantic segmentation of images. In addition, there’s SegformerForImageClassification which can be used to - you guessed it - classify images. The authors of SegFormer first pre-trained the Transformer encoder on ImageNet-1k to classify images. Next, they throw away the classification head, and replace it by the all-MLP decode head. Next, they fine-tune the model altogether on ADE20K, Cityscapes and COCO-stuff, which are important benchmarks for semantic segmentation. 

<img width="743" alt="Screenshot 2024-05-01 at 9 07 51 AM" src="https://github.com/andysingal/CV_public/assets/20493493/6460c3cf-64da-4ffe-a4cc-660ee75909e3">

## References
- [Understanding SegFormer](https://www.ikomia.ai/blog/master-segformer-advanced-semantic-segmentation)
- [An Efficient Transformers Design for Semantic Segmentation](https://pub.towardsai.net/segformer-an-efficient-transformers-design-for-semantic-segmentation-179d73590d0a)
  



Notebooks:
- [SegFormer](https://medium.com/geekculture/semantic-segmentation-with-segformer-2501543d2be4)
- [Finetune-SegFormer-Water-Detection](https://www.kaggle.com/code/ekaterinadranitsyna/segformer-water-segmentation-pytorch)
- [SegFormer-Roboflow](https://github.com/roboflow/notebooks/blob/main/notebooks/train-segformer-segmentation-on-custom-data.ipynb)
- [Binary Image Segmentation using Segformer](https://www.kaggle.com/code/niyarrbarman/binary-image-segmentation-using-segformer)
- https://debuggercafe.com/training-segformer-for-person-segmentation/ 
