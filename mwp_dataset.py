import os
import json
import torch
import pandas as pd


def get_input_text(problem, option_inds):
    # table
    table = problem['table']
    # table_for_pd = problem['table_for_pd']
    # pd_table = pd.DataFrame.from_dict(table_for_pd)

    # question
    question = problem['question']
    unit = problem['unit']
    choices = problem['choices']
    solution = problem['solution']

    if unit: # if unit is not empty
        question = question + f" (Unit: {unit})"

    if choices:
        for i, c in enumerate(choices):
            question += f" ({option_inds[i]}) {c}"

    return table, question.strip(), solution


def _load_dataset(data_root, split, option_inds):
    """
    Load the dataset.
    """
    # load the data entries/annotations
    problems = json.load(open(os.path.join(data_root, f'problems_{split}.json')))
    pids = list(problems.keys())
    pids = pids[:2]
    print("number of problems for %s:" % (split), len(problems))

    entries = []
    for pid in pids:
        print("pid:", pid)
        prob = {}
        prob['pid'] = pid
        # prob['problem'] = problems[pid]
        # print("problem:", prob['problem'])
        prob['table'], prob['question'], prob['answer']= get_input_text(problems[pid], option_inds)
        # print(prob['table'])
        # print(prob['input_text'])
        # prob['answer'] = problems[pid]['answer']
        # print("answer:", prob['answer'])
        entries.append(prob)

    return entries
    

## Create PyTorch dataset
class MWPDataset(torch.utils.data.Dataset):

    def __init__(self, data_root, data_split, tokenizer, args):
        self.data_root = data_root
        self.data_split = data_split
        self.tokenizer = tokenizer
        self.option_inds = args.option_inds
        self.max_length = args.max_length

        # load the data entries/annotations
        self.entries = _load_dataset(data_root, data_split, self.option_inds)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        # get one entry of data
        entry = self.entries[idx] # annotation
        
        pid = entry['pid']
        table = entry['table']
        input_text = entry['input_text']
        answer = entry['answer']

        # print("# input_string:", text)
        # print("# output_string:", answer)

        batch = {"tables": table,
                 "input_strings": input_text,
                 "output_strings": answer}

        return pid, batch
    
    def __setGrade__(self, idx, grade):
        # get one entry of data
        entry = self.entries[idx] # annotation
        entry['grade'] = grade

    def __getitemInTxt__(self, idx):
        # get one entry of data
        entry = self.entries[idx]
        return entry['input_text'], entry['answer']
