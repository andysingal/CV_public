```py
pip install torch torchvision transformers sentencepiece protobuf accelerate diffusers optimum-quanto huggingface_hub
```

```py
import time

import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from optimum.quanto import freeze, qfloat8, quantize
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

dtype = torch.bfloat16

bfl_repo = "black-forest-labs/FLUX.1-schnell"

# モデルの読み込み
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    bfl_repo, subfolder="scheduler"
)
text_encoder = CLIPTextModel.from_pretrained(
    "openai/clip-vit-large-patch14", torch_dtype=dtype
)
tokenizer = CLIPTokenizer.from_pretrained(
    "openai/clip-vit-large-patch14", torch_dtype=dtype
)
text_encoder_2 = T5EncoderModel.from_pretrained(
    bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype
)
tokenizer_2 = T5TokenizerFast.from_pretrained(
    bfl_repo, subfolder="tokenizer_2", torch_dtype=dtype
)
vae = AutoencoderKL.from_pretrained(
    bfl_repo, subfolder="vae", torch_dtype=dtype
)
transformer = FluxTransformer2DModel.from_pretrained(
    bfl_repo, subfolder="transformer", torch_dtype=dtype
)

# 8bit量子化
quantize(transformer, weights=qfloat8)
freeze(transformer)

quantize(text_encoder_2, weights=qfloat8)
freeze(text_encoder_2)

# FluxPipelineの設定
pipe = FluxPipeline(
    scheduler=scheduler,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=None,
    tokenizer_2=tokenizer_2,
    vae=vae,
    transformer=None,
)
pipe.text_encoder_2 = text_encoder_2
pipe.transformer = transformer
pipe.enable_model_cpu_offload()

# シードの固定
generator = torch.Generator().manual_seed(0)

# 画像の生成
image = pipe(
    prompt="A cat holding a sign that says hello world",
    width=1024,
    height=1024,
    num_inference_steps=4,
    max_sequence_length=256,
    generator=generator,
    guidance_scale=0.0,
).images[0]
image.save("flux-dev.png")
```
<img width="708" alt="Screenshot 2024-08-12 at 8 52 01 AM" src="https://github.com/user-attachments/assets/b67359dd-501b-4efb-85e5-504adf3385e7">

Resources:
- [Flux](https://qiita.com/yamichi77/items/1f12bac65d1584900b35)
- [Black-forest-flux](https://github.com/black-forest-labs/flux?tab=readme-ov-file)
- [Flux's ControlNet](https://note.com/npaka/n/n8254fd8d4af1)
- [Fine-tune FLUX.1 with your own images](https://replicate.com/blog/fine-tune-flux)

