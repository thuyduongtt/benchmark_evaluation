import csv
from pathlib import Path
import ast
import json


def export_caption(list_of_result_dir, output_dir, output_file_name):
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True)

    image_to_caption = {}
    for folder in list_of_result_dir:
        for csvfile in Path(folder).iterdir():
            if csvfile.name.startswith('.'):
                continue

            csv_file = f'{csvfile.parent}/{csvfile.name}'
            f = open(csv_file)

            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                image_to_caption[row['id'][:-4]] = ast.literal_eval(row['prediction'])[0]

    with open(f'{output_dir}/{output_file_name}', 'w') as f:
        json.dump(image_to_caption, f)


if __name__ == '__main__':
    root_dir = 'results/result_caption_coco_opt6.7b'
    ds_name = 'balanced_10'
    export_caption([f'{root_dir}/output_{ds_name}', f'{root_dir}/output_{ds_name}_test'], root_dir,
                   f'captions_{ds_name}.json')
