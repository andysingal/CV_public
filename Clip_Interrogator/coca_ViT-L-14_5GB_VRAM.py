import open_clip
import torch
from PIL import Image
import os
import time

# set your folder Path
directory_path = os.path.join(r"G:\test_video\test4")

print(f'selected path: {directory_path}')

def generate_caption(image_path, model, transform, device):
    im = Image.open(image_path).convert("RGB")
    im = transform(im).unsqueeze(0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        generated = model.generate(im)
    return open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")

def save_captions_in_directory(directory_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model and transforms
    model, _, transform = open_clip.create_model_and_transforms(
      model_name="coca_ViT-L-14",
      pretrained="mscoco_finetuned_laion2b_s13b_b90k"
    )
    model = model.to(device)

    all_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_files = len(all_files)
    
    print(f"Found {total_files} images in the directory.")

    # Loop through each file in the directory
    for idx, filename in enumerate(all_files, start=1):
        start_time = time.time()
        
        file_path = os.path.join(directory_path, filename)
        caption = generate_caption(file_path, model, transform, device)
        
        # Save caption to a .txt file with the same name as the image
        with open(os.path.join(directory_path, os.path.splitext(filename)[0] + '.txt'), 'w') as f:
            f.write(caption)

        time_taken = time.time() - start_time
        print(f"Processed {filename} ({idx}/{total_files}). Time taken: {time_taken:.2f} seconds. Remaining: {total_files - idx}.")

# Use the function
save_captions_in_directory(directory_path)
