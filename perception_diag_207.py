import time, json, faulthandler
from PIL import Image
import perception_module as pm
import torch

img_path='dataset/multi/WeatherBench/rain/test/input/207.jpg'
faulthandler.dump_traceback_later(120, repeat=True)

def p(*args):
    print(*args, flush=True)

t0=time.time(); p('[T] start')
p('[T] stage1 _get_model_and_processor begin')
model, processor, device = pm._get_model_and_processor()
p('[T] stage1 done', round(time.time()-t0,2), 'sec', 'device=', device)

p('[T] stage2 build prompt begin')
img=Image.open(img_path).convert('RGB')
prompt = pm._build_prompt(img)
p('[T] stage2 done', round(time.time()-t0,2), 'sec', 'prompt_len=', len(prompt))

p('[T] stage3 tensorize begin')
messages=[
    {'role':'system','content':'You are a precise image perception model for adverse weather analysis. Output valid JSON only.'},
    {'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]},
]
chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[chat_text], images=[img], padding=True, return_tensors='pt')
inputs = {k:(v.to(device) if hasattr(v,'to') else v) for k,v in inputs.items()}
p('[T] stage3 done', round(time.time()-t0,2), 'sec')

p('[T] stage4 generate begin')
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
p('[T] stage4 done', round(time.time()-t0,2), 'sec')

prompt_len = inputs['input_ids'].shape[1]
trimmed = out[:, prompt_len:]
text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
parsed = pm._normalize_result_json(pm._extract_json(text))
p('[T] raw_output_head', text[:500])
p('[T] parsed', json.dumps(parsed, ensure_ascii=False))
