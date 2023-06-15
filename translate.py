import transformers
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, AutoTokenizer
import pysrt
import torch
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_id', type=str, default="facebook/m2m100_1.2B")
parser.add_argument('--input', type=str)
parser.add_argument('--src_lang', type=str, default="en")
parser.add_argument('--tgt_lang', type=str, default="zh")
args = parser.parse_args()

model_id = args.model_id
pipeline = transformers.pipeline("translation", model=model_id, src_lang=args.src_lang, tgt_lang=args.tgt_lang, device_map="auto", torch_dtype=torch.float16, max_length=1024)

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
