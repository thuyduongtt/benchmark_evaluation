import argparse
import ast
import csv
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util
import jsonlines
import ijson
import spacy
from SUBSTRING_EXCEPTIONS import SUBSTRING_EXCEPTIONS

ANSWER_COL_INDEX = 3
PREDICTION_COL_INDEX = 4

METRICS = ['exact_match', 'substring', 'similarity']

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# https://github.com/UKPLab/sentence-transformers
similarity_model = None

en = spacy.load("en_core_web_sm")
stopwords = en.Defaults.stop_words


class Score:
    exact_match: int = 0
    substring: int = 0
    similarity: float = 0.0

    def __getitem__(self, item):
        if item == 'exact_match':
            return self.exact_match
        if item == 'substring':
            return self.substring
        if item == 'similarity':
            return self.similarity

    def __setitem__(self, key, value):
        if key == 'exact_match':
            self.exact_match = value
        if key == 'substring':
            self.substring = value
        if key == 'similarity':
            self.similarity = value

    def __str__(self):
        return f'exact_match: {self.exact_match}; substring: {self.substring}; similarity: {self.similarity:.4f};'

    def to_list(self):
        return [self[k] for k in METRICS]


# gt is the ground truth list of answers
def exact_match_score(pred, gt):
    return 1 if pred in gt else 0


def substring_score(pred, gt):
    if exact_match_score(pred, gt) == 1:
        return 1
    for s in gt:
        # only check substring directly for prediction with more than 1 words
        pred_words = pred.split()
        if len(pred_words) > 1 and pred in s:
            if check_substring_exception(pred, s):
                return 1

        gt_words = s.split()
        gt_words = [w for w in gt_words if w not in stopwords]  # remove stop words

        if pred in gt_words:
            return 1

        for w in gt_words:
            if w in pred:
                return 1
    return 0


def check_substring_exception(s1, s2):
    return f'{s1}___{s2}' not in SUBSTRING_EXCEPTIONS and f'{s2}___{s1}' not in SUBSTRING_EXCEPTIONS


# https://huggingface.co/tasks/sentence-similarity
def similarity_score(pred, gt):
    global similarity_model
    if similarity_model is None:
        similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    max_score = 0
    pred_emb = similarity_model.encode(pred, convert_to_tensor=True, device=device)
    for s in gt:
        emb = similarity_model.encode(s, convert_to_tensor=True, device=device)
        current_score = util.pytorch_cos_sim(pred_emb, emb).item()
        if current_score > max_score:
            max_score = current_score
    return max_score


# https://visualqa.org/evaluation.html
def vqa_acc(pred, gt):
    return 0.0


# depending on the model, answer might be given within a complete sentence. e.g.: [answer] The length is 300 meters
# we need to extract "The length is 300 meters" only
def extract_answer(answer_text):
    if answer_text.lower().startswith('answer:'):
        return answer_text[7:].strip()
    return answer_text.strip()


def get_ratio(a, b):
    if b == 0:
        return 0
    return a / b


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


def compute_score(list_of_result_dir, output_dir, limit=0):
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True)

    count = 0
    for folder in list_of_result_dir:
        if 0 < limit <= count:
            break
        for csvfile in Path(folder).iterdir():
            if 0 < limit <= count:
                break
            csv_file = f'{csvfile.parent}/{csvfile.name}'
            f = open(csv_file)

            count += 1
            if count % 100 == 0:
                print(count)

            csv_reader = csv.reader(f)

            score_file = open(f'{output_dir}/{csvfile.name}', 'w')
            csv_writer = csv.writer(score_file)

            row = next(csv_reader)
            csv_writer.writerow([*row, *METRICS])

            for row in csv_reader:
                # there's a bug that the answer set is empty, ignore them
                answer_str = row[ANSWER_COL_INDEX].lower()  # 3: answer, 4: prediction
                answer = ast.literal_eval(answer_str)
                if len(answer) == 0:
                    continue

                prediction_str = row[PREDICTION_COL_INDEX].lower()  # 3: answer, 4: prediction
                if prediction_str.startswith('['):
                    prediction_str = ast.literal_eval(prediction_str)[0]
                prediction = extract_answer(prediction_str)

                # compute all scores
                current_score = Score()
                if 'exact_match' in METRICS:
                    current_score.exact_match = exact_match_score(prediction, answer)
                if 'substring' in METRICS:
                    current_score.substring = substring_score(prediction, answer)
                if 'similarity' in METRICS:
                    # current_score.similarity = similarity_score(prediction, answer)
                    current_score.similarity = 0

                csv_writer.writerow([*row, *current_score.to_list()])

            score_file.close()
            f.close()


def compute_score_multichoice(path_to_dataset, question_jsonl, answer_jsonl):
    gt_answers = {}

    with jsonlines.open(question_jsonl) as reader:
        for obj in reader:
            gt_answers[obj['question_id']] = obj['answers']

    question_data = {}
    for split in ['train', 'test']:
        json_data = stream_data(f'{path_to_dataset}/{split}.json')
        for d in json_data:
            question_data[d['question_id']] = {
                **d,
                'shuffled_answer': gt_answers[d['question_id']]
                # answers in the form of (A, B, C, ...) after shuffling choices
            }

    total = 0
    correct = 0

    total_by_hop = {}
    score_by_hop = {}

    total_by_scene_graph = {
        'with': 0,
        'without': 0
    }
    score_by_scene_graph = {
        'with': 0,
        'without': 0
    }

    total_by_ds = {
        'VG': 0,
        'GLDv2': 0
    }
    score_by_ds = {
        'VG': 0,
        'GLDv2': 0
    }

    with jsonlines.open(answer_jsonl) as reader:
        for obj in reader:
            qid = obj['question_id']
            ans = obj['text']

            row = question_data[qid]
            is_correct = ans in row['shuffled_answer']

            total += 1

            n_hop = row['n_hop']
            if n_hop not in total_by_hop:
                total_by_hop[n_hop] = 0
                score_by_hop[n_hop] = 0
            total_by_hop[n_hop] += 1

            if row['has_scene_graph']:
                total_by_scene_graph['with'] += 1
            else:
                total_by_scene_graph['without'] += 1

            ds_name = 'VG' if row['image_id'].startswith('VG_') else 'GLDv2'
            total_by_ds[ds_name] += 1

            if is_correct:
                correct += 1
                score_by_hop[n_hop] += 1
                score_by_scene_graph['with' if row['has_scene_graph'] else 'without'] += 1
                score_by_ds[ds_name] += 1

    print('Total:', total, 'Acc:', f'{get_ratio(correct, total):.4f}')
    for h in score_by_hop.keys():
        print(f'{h}-hop:', f'{get_ratio(score_by_hop[h], total_by_hop[n_hop]):.4f}')
    print('W/ Scene graph:', f"{get_ratio(score_by_scene_graph['with'], total_by_scene_graph['with']):.4f}")
    print('W/O Scene graph:',
          f"{get_ratio(score_by_scene_graph['without'], total_by_scene_graph['without']):.4f}")
    for ds in total_by_ds.keys():
        print(ds, f"{get_ratio(score_by_ds[ds], total_by_ds[ds]):.4f}")


def anaylysis_score(result_dir, limit=0):
    total = 0
    score = Score()

    total_by_hop = {}
    score_by_hop = {}

    total_by_scene_graph = {
        'with': 0,
        'without': 0
    }
    score_by_scene_graph = {
        'with': Score(),
        'without': Score()
    }

    total_by_ds = {
        'VG': 0,
        'GLDv2': 0
    }
    score_by_ds = {
        'VG': Score(),
        'GLDv2': Score()
    }

    count = 0
    for csvfile in Path(result_dir).iterdir():
        if 0 < limit <= count:
            break
        csv_file = f'{csvfile.parent}/{csvfile.name}'
        f = open(csv_file)

        count += 1

        reader = csv.DictReader(f)
        for row in reader:
            # there's a bug that the answer set is empty, ignore them
            answer_str = row['answer'].lower()
            answer = ast.literal_eval(answer_str)
            if len(answer) == 0:
                continue

            total += 1

            n_hop = row['n_hop']
            if n_hop not in total_by_hop:
                total_by_hop[n_hop] = 0
                score_by_hop[n_hop] = Score()
            total_by_hop[n_hop] += 1

            if row['has_scene_graph']:
                total_by_scene_graph['with'] += 1
            else:
                total_by_scene_graph['without'] += 1

            ds_name = 'VG' if row['id'].startswith('VG_') else 'GLDv2'
            total_by_ds[ds_name] += 1

            for s in METRICS:
                val = ast.literal_eval(row[s])
                score[s] += val
                score_by_hop[n_hop][s] += val
                score_by_scene_graph['with' if row['has_scene_graph'] else 'without'][s] += val
                score_by_ds[ds_name][s] += val

        f.close()

    print('Total:', total, '| Score:', score)
    for s in METRICS:
        print('=====', s)
        print('Acc:', f'{get_ratio(score[s], total):.4f}')
        for h in score_by_hop.keys():
            print(f'{h}-hop:', f'{get_ratio(score_by_hop[h][s], total_by_hop[n_hop]):.4f}')
        print('W/ Scene graph:', f"{get_ratio(score_by_scene_graph['with'][s], total_by_scene_graph['with']):.4f}")
        print('W/O Scene graph:',
              f"{get_ratio(score_by_scene_graph['without'][s], total_by_scene_graph['without']):.4f}")
        for ds in total_by_ds.keys():
            print(ds, f"{get_ratio(score_by_ds[ds][s], total_by_ds[ds]):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()

    compute_score([f'result_{args.model}/output_{args.ds}', f'result_{args.model}/output_{args.ds}_test'], f'result_{args.model}/output_{args.ds}_score')
    # anaylysis_score(f'result_{args.model}/output_{args.ds}_score')

    # compute_score_multichoice(f'/Volumes/DATA/_Code/Git/tdthesis/ds/output/small_set/export_{args.ds}',
    #                           f'result_llava/{args.ds}.jsonl', f'result_llava/answers/merge_{args.ds}.jsonl')
