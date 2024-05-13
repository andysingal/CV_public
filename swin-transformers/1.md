## Swin Transformer
Introduced in the 2021 paper, Swin Transformer: Hierarchical Vision Transformer using Shifted Windows, the Swin Transformer architecture optimizes for latency and performance using a shifted window (as opposed to sliding window) approach which reduces the number of operations required. Swin is considered a hierarchical backbone for computer vision. Swin can be used for tasks like image classification. 

## Main Highlights
### Shifted windows
In the original ViT, attention is done between each patch and all other patches, which gets computationally intensive. Swin optimizes this process by reducing the normally quadratic complexity ViT into linear complexity (with respect to image size). Swin achieves this using a technique similar to CNN, where patches only attend to other patches in the same window, as opposed to all other patches, and then are gradually merged with neighboring patches. This is what makes Swin a hierarchical model. 

<img width="808" alt="Screenshot 2024-05-13 at 8 52 14 AM" src="https://github.com/andysingal/CV_public/assets/20493493/4807b94e-109e-4efd-a52a-c131d17d7b3f">



Resources:

- [Swin Transformer](https://huggingface.co/learn/computer-vision-course/en/unit3/vision-transformers/swin-transformer) 

