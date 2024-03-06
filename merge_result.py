# MERGE RESULT DIRECTORIES INTO ONE
# e.g: results from "output_unbalanced_0", "output_unbalanced_20000", ""output_unbalanced_40000", ...
# will be merged into "output_unbalanced"


import argparse
from pathlib import Path
import shutil


def merge(dirs, merge_dir_name):
    if not Path(merge_dir_name).exists():
        Path(merge_dir_name).mkdir(parents=True)
    for d in dirs:
        for f in Path(d).iterdir():
            dst_path = f'{merge_dir_name}/{f.name}'
            assert not Path(dst_path).exists(), print(dst_path)
            shutil.copy(str(f), dst_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='instructBLIP_mc')
    parser.add_argument('--result_prefix', type=str, default='output_mc_blip2_t5_instruct_flant5xxl_unbalanced')
    args = parser.parse_args()

    result_dir = []
    for d in Path(f'results/result_{args.model}/').iterdir():
        if d.name == args.result_prefix:
            continue
        if d.is_dir() and d.name.startswith(args.result_prefix):
            result_dir.append(f'results/result_{args.model}/{d.name}')

    print(f'Found {len(result_dir)} directories to merge:')
    print(result_dir)

    merge(result_dir, f'results/result_{args.model}/{args.result_prefix}')
