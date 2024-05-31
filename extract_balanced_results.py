# EXTRACT RESULT FOR BALANCED VERSION FROM UNBALANCED

import argparse
from pathlib import Path
import ijson
from datetime import datetime
import csv

UNBALANCED_DIR = ''
UNBALANCED_DATA = None
CSV_FIELDS = []


def load_unbalanced_data():
    global UNBALANCED_DATA
    UNBALANCED_DATA = {}

    global CSV_FIELDS

    for csvfile in Path(UNBALANCED_DIR).iterdir():
        csv_file = f'{csvfile.parent}/{csvfile.name}'
        f = open(csv_file, encoding="utf8")

        reader = csv.DictReader(f)

        CSV_FIELDS = reader.fieldnames

        for row in reader:
            values = []
            for k in CSV_FIELDS:
                values.append(row[k])

            if row['id'] not in UNBALANCED_DATA:
                UNBALANCED_DATA[row['id']] = []
            UNBALANCED_DATA[row['id']].append({
                'question': row['question'],
                # 'prediction': row['prediction'],
                'row_data': values
            })


def convert(image_id, question):
    if UNBALANCED_DATA is None:
        load_unbalanced_data()

    if image_id not in UNBALANCED_DATA:
        print('Missing image id:', image_id)
        return None

    for q in UNBALANCED_DATA[image_id]:
        if q['question'] == question:
            # return q['prediction']
            return q['row_data']

    return None


def run_pipeline_by_question(task, path_to_dataset, output_dir_name, limit=0, start_at=0, split='train'):
    def init_csv_file():
        if not Path(output_dir_name).exists():
            Path(output_dir_name).mkdir(parents=True)

        timestamp = datetime.now().isoformat()
        unique_name = timestamp.replace(':', '_')
        n = 1
        while Path(f'{output_dir_name}/result_{unique_name}').exists():
            n += 1
            unique_name = f'{timestamp}_{n}'

        csvfile_path = f'{output_dir_name}/result_{unique_name}'
        csvfile = open(csvfile_path + '.csv', 'w', encoding='utf-8', newline='')
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(CSV_FIELDS)

        return csvfile, csvwriter

    csv_file, csv_writer = init_csv_file()

    json_data = stream_data(f'{path_to_dataset}/{split}.json', limit=limit, start_at=start_at)

    i = 0
    n_error = 0
    for d in json_data:
        i += 1

        # if i == 1 or i % 100 == 0:
        #     print(f"[{i}]: {d['image_id']}")

        # split into smaller CSV file every 1000 records
        if i % 1000 == 0:
            csv_file.close()
            csv_file, csv_writer = init_csv_file()

        row_data = task(d['image_id'], d['question'])

        if row_data is None:
            n_error += 1
            continue

        csv_writer.writerow(row_data)

        # answers = d['answers']
        # csv_writer.writerow([d['image_id'], local_img_path, d['question'], answers,
        #                      prediction, d['n_hop'], d['has_scene_graph'], split])

    csv_file.close()
    print('Num of error:', n_error)


'''
n_questions: int
exported_time: datetime
questions: array
    image_id
    image_name
    image_dir
    dataset_name
    question_id
    question
    answers
    answers_scores
    choices
    choice_scores
    property_id
    property_label
    n_hop
    has_scene_graph
'''


def stream_data(path_to_json_file, limit=0, start_at=0):
    i = 0
    with open(path_to_json_file) as f:
        datareader = ijson.items(f, 'questions.item')
        for record in datareader:
            i += 1
            if i < start_at + 1:
                continue
            if 0 < limit < i - start_at:
                return

            yield record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mPLUGOwl2')
    parser.add_argument('--unbalanced_dir', type=str, default='output_unbalanced_score')
    parser.add_argument('--balanced_dir', type=str, default='output_balanced_10_score')
    parser.add_argument('--path_to_balanced_ds', type=str, default='D:/_Code/Git/tdthesis/ds/subset/balanced_10')
    args = parser.parse_args()

    global UNBALANCED_DIR
    UNBALANCED_DIR = f'results/result_{args.model}/{args.unbalanced_dir}'

    run_pipeline_by_question(convert, args.path_to_balanced_ds, f'results/result_{args.model}/{args.balanced_dir}')


if __name__ == '__main__':
    main()
