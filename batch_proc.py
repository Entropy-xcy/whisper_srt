import os
import argparse
import whisper
import subprocess
import whisper
from whisper.utils import WriteSRT
import tempfile
import io
import transformers
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, AutoTokenizer
from tqdm import tqdm


def gen_srt(video_file, src_lang=None, tgt_lang="zh", cuda_device=0):
    with tempfile.TemporaryDirectory() as temp_dir:
        model = whisper.load_model("large-v2")

        print("Translating from {} to {}".format(src_lang, tgt_lang))
        results = model.transcribe(video_file, language=src_lang)
        del model

        ## Translate
        model_id = "facebook/m2m100_1.2B"
        pipeline = transformers.pipeline("translation", 
                                        model=model_id, 
                                        src_lang=src_lang, 
                                        tgt_lang=tgt_lang, 
                                        device=cuda_device, 
                                        torch_dtype=torch.float16, 
                                        max_length=1024)
        cache = {}
        def translate(text):
            if text in cache:
                return cache[text]
            else:
                cache[text] = pipeline(text)[0]['translation_text']
                return cache[text]

        for seg in tqdm(results['segments']):
            print("Original:", seg['text'])
            translated = translate(seg['text'])
            print("Translated:", translated)
        
        srt_content = io.StringIO()
        writer = WriteSRT(output_dir=temp_dir)
        writer.write_result(results, srt_content)
    return srt_content.getvalue()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # positional arguments after optional arguments
    parser.add_argument("--video_dir", help="directory of video files", default=None)
    parser.add_argument('--src_lang', type=str, default=None)
    parser.add_argument('--tgt_lang', type=str, default="zh")
    parser.add_argument('video_files', metavar='ARG', nargs='*', help="Video files to process")

    args = parser.parse_args()
    if args.video_dir is not None:
        video_files = [os.path.join(args.video_dir, f) for f in os.listdir(args.video_dir) if f.endswith(".mp4")]
    else:
        video_files = args.video_files
    
    result = gen_srt(video_files[0], src_lang=args.src_lang, tgt_lang=args.tgt_lang)
    print(result)
