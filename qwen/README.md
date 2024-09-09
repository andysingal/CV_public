```py
!pip install pillow
!python -m pip install git+https://github.com/huggingface/transformers

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
from IPython.display import 
display

# Load model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct"
)

# Load the Car Sales Dashboard image
image = Image.open("/content/Analytical-Dashboard-Drill-Down-2.webp")
# Prepare the image and create a message template
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
            },
            {
                "type": "text",
                "text": "Explain this image? Your analysis should help the User."
            }
        ]
    }
]
text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(
    text=[text_prompt],
    images=[image],
    padding=True,
    return_tensors="pt"
)
inputs = inputs.to("cuda")


output_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
]
output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
)

## Show the image and the generated text:

from PIL import Image
from IPython.display import display

image = Image.open("/content/Analytical-Dashboard-Drill-Down-2.webp")
# Display the image
display(image)

# Print the generated text
for i in output_text:
    print(i)

```
