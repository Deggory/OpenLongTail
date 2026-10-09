"""One-sentence scene captions of raw clips with Qwen2.5-VL, written to <out>/per_uuid/<uuid>.txt.

    python -m openlongtail.scripts.build_captions --input CLIPS --out-root CLIPS/captions
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from openlongtail.data.dashcam import collect_clips

DEFAULT_MODEL = 'Qwen/Qwen2.5-VL-7B-Instruct'
PROMPT = ('You are captioning a driving-scene dashcam video for a generative model. Look at these frames (all the same clip) and write '
          'ONE concise sentence describing the overall scene: road/location type, time of day, weather, and the most salient '
          'surroundings (buildings, vehicles, vegetation). Be factual and specific; no camera talk, no lists, no quotes. '
          'Respond with ONLY the sentence.')


def sample_frames(path: Path, n: int) -> list:
    """n frames evenly spaced over the video, as PIL images."""
    import imageio.v3 as iio
    from PIL import Image
    frames = iio.imread(path, plugin='pyav')
    n = max(1, min(n, len(frames)))
    idx = [len(frames) // 2] if n == 1 else [min(len(frames) - 1, round(i * (len(frames) - 1) / (n - 1))) for i in range(n)]
    return [Image.fromarray(frames[i]).convert('RGB') for i in idx]


def first_sentence(text: str) -> str:
    s = ' '.join(text.replace('```', ' ').split()).strip().strip('"\'').strip()
    for end in ('. ', '? ', '! '):
        if end in s:
            return s[:s.index(end) + 1]
    return s


class QwenCaptioner:

    def __init__(self, model: str, device: str = 'cuda') -> None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, torch_dtype=torch.bfloat16, device_map=device).eval()

    @torch.no_grad()
    def __call__(self, frames: list) -> str:
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': f} for f in frames] + [{'type': 'text', 'text': PROMPT}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = {k: v.to(self.model.device) for k, v in self.processor(text=[text], images=frames, padding=True, return_tensors='pt').items()}
        generated = self.model.generate(**inputs, max_new_tokens=80, do_sample=False)
        return first_sentence(self.processor.batch_decode(generated[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', type=Path, required=True, help='raw clip root')
    ap.add_argument('--out-root', type=Path, required=True)
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--frames', type=int, default=4)
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()
    out = a.out_root / 'per_uuid'
    out.mkdir(parents=True, exist_ok=True)
    clips = collect_clips(a.input)
    captioner = None
    for i, clip in enumerate(clips, 1):
        uuid = str(torch.load(clip / 'meta.pt', map_location='cpu', weights_only=True).get('uuid', clip.parent.name))
        path = out / f'{uuid}.txt'
        if path.exists() and not a.overwrite:
            continue
        captioner = captioner or QwenCaptioner(a.model)
        caption = captioner(sample_frames(clip / 'front.mp4', a.frames))
        path.write_text(caption)
        print(f'[{i}/{len(clips)}] {uuid}: {caption}', flush=True)


if __name__ == '__main__':
    main()
