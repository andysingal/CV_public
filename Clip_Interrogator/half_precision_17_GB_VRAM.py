from PIL import Image
import os
import time
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

# set your folder Path
folder_path = os.path.join(r"G:\test_video\test4")

print(f'selected path: {folder_path}')

def generate_caption(image_path, processor, model, device):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=75)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True, max_new_tokens=75, max_length=250)[0].strip()
    return generated_text

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ask user for model choice
    options = [
        ("Salesforce/blip2-opt-6.7b", "~16.5 GB VRAM"),
        ("Salesforce/blip2-flan-t5-xxl", "~24+ GB VRAM"),
        ("Salesforce/blip2-opt-6.7b-coco", "~16.5 GB VRAM")
    ]
    print("Please select a model:")
    for idx, (option, vram) in enumerate(options, 1):
        print(f"{idx}. {option} (Requires {vram})")
    choice = int(input("Enter choice (1/2/3): "))
    selected_model, _ = options[choice-1]
    print("Increase your virtual RAM if you don't have sufficient RAM!")
    print("Loading model into RAM first please wait...")

    processor = Blip2Processor.from_pretrained(selected_model)
    model = Blip2ForConditionalGeneration.from_pretrained(selected_model, torch_dtype=torch.float16, device_map="cuda:0")
    model.to(device)

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    total_images = len(image_files)
    captioned_images = 0

    start_time = time.time()

    for image_file in image_files:
        caption = generate_caption(os.path.join(folder_path, image_file), processor, model, device)
        
        # Save caption as .txt file
        txt_filename = os.path.splitext(image_file)[0] + ".txt"
        with open(os.path.join(folder_path, txt_filename), 'w') as f:
            f.write(caption)
        
        captioned_images += 1
        elapsed_time = time.time() - start_time
        speed = elapsed_time / captioned_images
        
        print(f"Captioned: {captioned_images}/{total_images}, Remaining: {total_images-captioned_images}, Speed: {speed:.2f} seconds per image")
