SegGPT is a vision generalist on image segmentation, quite like GPT for computer vision 

<img width="1349" alt="Screenshot 2024-05-03 at 8 31 51 AM" src="https://github.com/andysingal/CV_public/assets/20493493/b0f1b80a-6ec0-42eb-a198-062707b07b82">

Definitions:

1. ICL: In-context learning (ICL) learns a new task from a small set of examples presented within the context (the prompt) at inference time. LLMs trained on sufficient data exhibit ICL, even though they are trained only with the objective of next token prediction.

## What is the core technology of SegGPT?
The core technologies of SegGPT are Vision Transformer (ViT) and in-context learning. ViT is an application of transformer technology to image recognition tasks, allowing us to segment images into patches and deeply understand their relationships. This technology can capture minute features and patterns in images, greatly improving segmentation accuracy.

In-context learning, on the other hand, is a method in which a model learns a task through given examples. In this approach, the model selects and performs an appropriate segmentation technique based on the specific context or situation. This flexibility is what sets SegGPT apart from other models.

<img width="926" alt="Screenshot 2024-05-03 at 8 52 31 AM" src="https://github.com/andysingal/CV_public/assets/20493493/2529c028-5274-43bd-a23a-53e789e73233">

Notebook
- [SegGPT](https://github.com/andysingal/CV_public/blob/main/SegGPT/Inference_with_SegGPT_for_one_shot_image_segmentation.ipynb)
