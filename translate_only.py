import transformers
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, AutoTokenizer
import pysrt
import torch
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_id', type=str, default="facebook/m2m100-12B-last-ckpt")
parser.add_argument('--input', type=str)
args = parser.parse_args()

model_id = args.model_id
pipeline = transformers.pipeline("translation", model=model_id, src_lang="ja", tgt_lang="zh", device_map="auto", torch_dtype=torch.float16, max_length=1024)

cache = {}
def translate(text):
    if text in cache:
        return cache[text]
    else:
        cache[text] = pipeline(text)[0]['translation_text']
        return cache[text]

subs = pysrt.open(args.input, encoding='utf-8')
for s in tqdm(subs):
    s.text = translate(s.text)

subs.save(args.input, encoding='utf-8')
