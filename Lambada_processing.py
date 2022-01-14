# Script to process Lambada dataset
#
# Date: sometime 2019
# Author: Daria D.
#



import os

from parameters import datapaths
from utils import get_vocabulary, write_vocabulary


def read_lambada_file(input_file):
    text_list = []
    with open(input_file) as reader:
        # print(story_input_file)
        story_lines = reader.readlines()
        for text in story_lines:
            text = text.strip()
            if len(text) > 0:
                text_list.append(text)
    return text_list


def read_train_questions_examples( data_dir):
    vocabulary = set()

    folders = os.listdir(data_dir)
    for folder in folders:
        # print(folder)
        files = os.listdir(os.path.join(data_dir, folder))
        for file in files:
            story_input_file = os.path.join(data_dir, folder, file)

            text_list = read_lambada_file(story_input_file)
            _, local_vocab = get_vocabulary(text_list)
            vocabulary.update(local_vocab)

    return vocabulary


def get_vocabulary_ner(data_dir, subset):
    input_file = os.path.join(data_dir, "lambada_{}_plain_text.txt".format(subset))
    text_list = read_lambada_file(input_file)
    _, vocabulary = get_vocabulary(text_list)

    return vocabulary



task_name = "lambada"
datapath = datapaths[task_name]

train_vocabulary = read_train_questions_examples(os.path.join(datapath, "train-novels"))
dev_vocabulary   = get_vocabulary_ner(datapath, "development")
test_vocabulary  = get_vocabulary_ner(datapath, "test")


vocabulary = train_vocabulary
vocabulary.update(dev_vocabulary)
vocabulary.update(test_vocabulary)

print("Lambada VOCAB size:", len(vocabulary))
write_vocabulary(vocabulary, task_name)
