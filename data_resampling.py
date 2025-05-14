import argparse
import json
import os

from tqdm import tqdm
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--resolution", required=True,
                    help="Resampling resolution")
parser.add_argument("--dirname",
                    help="Resampling resolution")

args = parser.parse_args()
resol = int(args.resolution)
output_suffix = f'_{resol}'

if args.dirname:
    base_dir = os.path.join('data/nerf_synthetic', args.dirname)
    splits = ['train', 'val', 'test']
else:
    base_dir = 'data/nerf_synthetic/lego'

for split in splits:
    input_json = os.path.join(base_dir, f'transforms_{split}.json')
    output_json = os.path.join(base_dir,
                               f'transforms_{split}_{resol}.json')

    with open(input_json, 'r') as f:
        data = json.load(f)

    for frame in data['frames']:
        old_path = frame['file_path']
        if old_path.startswith(split + '/'):
            frame['file_path'] = old_path.replace(split + '/',
                                                  f'{split}_{resol}/')
        elif old_path.startswith('./' + split + '/'):
            frame['file_path'] = old_path.replace(
                './' + split + '/',
                f'./{split}_{resol}/'
            )

    with open(output_json, 'w') as f:
        json.dump(data, f, indent=2)

    print(f'✔ {output_json} saved.')

    input_dir = os.path.join(base_dir, split)
    output_dir = os.path.join(base_dir, split + output_suffix)
    os.makedirs(output_dir, exist_ok=True)

    for fname in tqdm(os.listdir(input_dir), desc=f'Processing {split}'):
        if fname.endswith('.png'):
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname)

            with Image.open(in_path) as img:
                img_resized = img.resize((resol, resol), Image.LANCZOS)
                img_resized.save(out_path)
