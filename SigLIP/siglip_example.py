import torch
import torch.nn.functional as F
from urllib.request import urlopen
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch==2.23.0

model, preprocess = create_model_from_pretrained('hf-hub:timm/ViT-B-16-SigLIP')
tokenizer = get_tokenizer('hf-hub:timm/ViT-B-16-SigLIP')

image = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))
image = preprocess(image).unsqueeze(0)

labels_list = ["a diagram", "a dog", "a cat", "a beignet"]
text = tokenizer(labels_list, context_length=model.context_length)

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[0., 0., 0., 1.0]]

zipped_list = list(zip(labels_list, text_probs.tolist()[0]))
print(zipped_list)
