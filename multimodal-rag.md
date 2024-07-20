Llava

```py
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

processor = LlavaNextProcessor.from_pretrained("liuhaotian/llava-next-v1.1-7b-chat")
model = LlavaNextForConditionalGeneration.from_pretrained("liuhaotian/llava-next-v1.1-7b-chat").to("cuda")

messages = [
    {"role": "user", "content": "Image of a red stop sign."},
    {"role": "assistant", "content": "A red octagon with the word STOP in white capital letters."},
    {"role": "user", "content": "More details please."},
]
prompt = processor.apply_chat_template(messages, return_tensors="pt").to("cuda")
image = processor(images=image, return_tensors="pt").to("cuda")

with torch.no_grad():
    generated_ids = model.generate(
        pixel_values=image["pixel_values"],
        input_ids=prompt["input_ids"],
        attention_mask=prompt["attention_mask"],
        max_new_tokens=100,
    )

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)
```
