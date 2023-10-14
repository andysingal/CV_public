## 8bit_precision_10_GB_VRAM.py supporting below models

("Salesforce/blip2-opt-6.7b", "~16.5 GB VRAM"),
("Salesforce/blip2-flan-t5-xxl", "~24+ GB VRAM"),
("Salesforce/blip2-opt-6.7b-coco", "~16.5 GB VRAM")
half_precision_17_GB_VRAM.py supporting below models

("Salesforce/blip2-opt-6.7b", "~9.5 GB VRAM"),
("Salesforce/blip2-flan-t5-xxl", "~19.5 GB VRAM"),
("Salesforce/blip2-opt-6.7b-coco", "~9.5 GB VRAM")
I think 8 bit and half precision are almost same quality. I am still testing.

## How To Use Scripts On RunPod
RunPod referral link : https://bit.ly/RunPodIO

Select RunPod Fast Stable Diffusion template

Edit pod and expose HTTP ports and add 7861

If you wish to delete auto downloaded models run below code first (optional)
rm -r auto-models
Upload runpod_install.sh into workspace folder
Open a new terminal and execute below codes to install
export HF_HOME="/workspace"
chmod +x runpod_install.sh
./runpod_install.sh
After Install How To Run On RunPod
How to use RunPod and RunPodCTL tutorial >
https://youtu.be/QN1vdGhjcRc

Open a new terminal
Execute below code
export HF_HOME="/workspace"
source venv/bin/activate
This above code will activate installed venv and now you can use scripts
To start Clip Interrogator Graido Web UI execute below code
python Clip_Interrogator.py --share
Use --share on RunPod it is mandatory due to a bug : https://github.com/gradio-app/gradio/issues/5896
When you first time run it will download model and you may get error message
After download of model complete refresh and try again
You can watch the terminal you started for download process
To run other captioners edit their folder path with runpod version
E.g. edit half_precision_17_GB_VRAM.py and change path like below
/workspace/test1
