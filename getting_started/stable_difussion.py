import boto3
import json
import base64
import os

prompt = 'a persian cat with odd eye colour in a beach in 4k hd image'

bedrock = boto3.client(service_name='bedrock-runtime')

prompt_template = [{'text': prompt, 'weight':1}]

payload = {
    "text_prompts": prompt_template,
    "cfg_scale":10,
    "seed":0,
    "steps":50,
    "width":1024,
    "height":1024
}

body = json.dumps(payload)

response = bedrock.invoke_model(
    modelId="stability.stable-diffusion-xl-v1",
    contentType="application/json",
    accept="application/json",
    body = body
)

response_body = json.loads(response.get('body').read())
print(response_body)

artifact = response_body.get('artifacts')[0]

image_encoded = artifact.get('base64').encode('utf-8')

image_bytes = base64.b64decode(image_encoded)

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
file_name = f"{output_dir}/generated-img.png"
with open(file_name, "wb") as f:
    f.write(image_bytes)