import csv
import ast
from pathlib import Path
from analysis_result import extract_answer, exact_match_score, substring_score, ANSWER_COL_INDEX, PREDICTION_COL_INDEX
import os


# count the number of rows in result file
def count_rows(path_to_result_dir):
    all_files = []
    for file in Path(path_to_result_dir).iterdir():
        if file.suffix != '.csv':
            continue
        all_files.append(file.name)

    all_files.sort()  # sort by name in alphabet

    for file in all_files:
        with open(f"{path_to_result_dir}/{file}") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            n_row = sum(1 for _ in csv_reader)
            print(file, '|', n_row)


# there is a bug in calculating substring_score
def find_wrong_substring_score(path_to_score_dir):
    limit = 10
    count = 0
    n_files = 0
    n_rows = 0

    for file in Path(path_to_score_dir).iterdir():
        if file.suffix != '.csv':
            continue
        n_files += 1
        with open(file) as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                n_rows += 1
                answer_str = row['answer'].lower()
                answer = ast.literal_eval(answer_str)
                if len(answer) == 0:
                    continue

                prediction_str = row['prediction'].lower()
                if prediction_str.startswith('['):
                    prediction_str = ast.literal_eval(prediction_str)[0]
                prediction = extract_answer(prediction_str)

                exact_match = ast.literal_eval(row['exact_match'])
                substring = ast.literal_eval(row['substring'])
                if exact_match > substring:
                    if count < limit:
                        print('Answer:', answer, 'Prediction:', prediction, 'exact_match:', row['exact_match'],
                              'substring:', row['substring'])
                        print(exact_match_score(prediction, answer), substring_score(prediction, answer))
                    count += 1

    print('Total checked:', n_files, 'files,', n_rows, 'rows', 'error:', count)


# in case we forget to clear the result dir before running
# there are duplicated results
def find_duplicate_results(path_to_result_dir, auto_remove=False):
    all_files = []
    id_flag = {}
    to_remove_files = []
    i = 0
    for file in Path(path_to_result_dir).iterdir():
        if file.suffix != '.csv':
            continue
        all_files.append(file.name)

    all_files.sort()  # sort by name in alphabet

    for file in all_files:
        i += 1
        with open(f"{path_to_result_dir}/{file}") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                img_id = row['id']
                if img_id not in id_flag:
                    id_flag[img_id] = {
                        'index': i,
                        'file': file
                    }
                    break

                elif id_flag[img_id]['index'] < i - 1:
                    # same image_id but not in 2 continuous files ==> duplicated file
                    print(id_flag[img_id]['file'], '|', file, '|',
                          check_identity(f"{path_to_result_dir}/{id_flag[img_id]['file']}",
                                         f"{path_to_result_dir}/{file}"))
                    to_remove_files.append(file)
                    break

    print('Found:', len(to_remove_files))
    print('\n'.join(to_remove_files))

    # start removing files
    if auto_remove:
        for file in to_remove_files:
            # remove the corresponding score
            score_file = f'{path_to_result_dir}_score/{file}'
            if Path(score_file).exists():
                os.remove(score_file)
            os.remove(f'{path_to_result_dir}/{file}')


def check_identity(csv_file_1, csv_file_2):
    img_ids_1 = []
    img_ids_2 = []
    with open(csv_file_1) as f1:
        csv_reader_1 = csv.DictReader(f1)
        for row in csv_reader_1:
            img_ids_1.append(row['id'] + '__' + row['question'])
    with open(csv_file_2) as f2:
        csv_reader_2 = csv.DictReader(f2)
        for row in csv_reader_2:
            img_ids_2.append(row['id'] + '__' + row['question'])
    l1 = len(img_ids_1)
    l2 = len(img_ids_2)
    exclusive = set(img_ids_1) ^ set(img_ids_2)
    l = len(exclusive)
    return l1, l2, l


def fix_substring_score(path_to_score_dir, limit=0):
    fixed_dir = path_to_score_dir + '_fixed'

    if not Path(fixed_dir).exists():
        Path(fixed_dir).mkdir(parents=True)

    count = 0
    for file in Path(path_to_score_dir).iterdir():
        if file.suffix != '.csv':
            continue

        if 0 < limit <= count:
            break

        count += 1

        with open(file) as csv_file:
            csv_reader = csv.reader(csv_file)
            fixed_file = open(f'{fixed_dir}/{file.name}', 'w', encoding='utf-8')
            csv_writer = csv.writer(fixed_file)

            header_row = next(csv_reader)
            csv_writer.writerow(header_row)

            for row in csv_reader:
                answer_str = row[ANSWER_COL_INDEX].lower()  # 3: answer, 4: prediction
                answer = ast.literal_eval(answer_str)
                if len(answer) == 0:
                    continue

                prediction_str = row[PREDICTION_COL_INDEX].lower()  # 3: answer, 4: prediction
                if prediction_str.startswith('['):
                    prediction_str = ast.literal_eval(prediction_str)[0]
                prediction = extract_answer(prediction_str)

                row[-3] = exact_match_score(prediction, answer)
                row[-2] = substring_score(prediction, answer)
                csv_writer.writerow(row)

            fixed_file.close()


if __name__ == '__main__':
    # find_wrong_substring_score('result_blip2/output_balanced_10_score')
    # find_duplicate_results('result_lavis/output_unbalanced', auto_remove=False)
    count_rows('result_lavis/output_balanced_10')
    # fix_substring_score('result_kosmos/output_balanced_10_score')
