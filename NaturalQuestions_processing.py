# Script to process Natural Questions dataset
#
# Date: sometime 2019
# Author: Daria D.
#


import json
import os
import gzip

from spacy.lang.en import English

from parameters import datapaths
from utils import write_vocabulary, get_vocabulary, write_questions

nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)


def process_file(filename):
    question_list, passage_list, answer_list, instance_list = [], [], [], []
    question_text_list = []
    vocabulary_set, html_token_set = set(), set()
    with gzip.open(filename) as f:
        for line in f:
            entry = json.loads(line)
            #print(entry)
            passage_tokens = entry["document_tokens"]
            #print(passage_tokens)
            passage_text_tokens = []
            for passage_token in passage_tokens:
                token = passage_token["token"]
                if passage_token["html_token"]:
                    html_token_set.add(token)
                else:
                    passage_text_tokens.append(token)

            question_tokens = entry["question_tokens"]
            question = entry["question_text"]
            question_text_list.append(question)

            question_list.append(len(question_tokens))
            passage_list.append(len(passage_text_tokens))
            instance_list.append(entry["document_title"])

            answer = ""
            for annotation in entry["annotations"]:
                if annotation["yes_no_answer"] != "NONE":
                    answer_list.append(1)
                    answer = annotation["yes_no_answer"]
                else:
                    answer_annotation = annotation["short_answers"]
                    if len(answer_annotation) == 0:
                        answer_annotation = annotation["long_answer"]
                        if answer_annotation["candidate_index"] == -1:
                            continue
                    else:
                        answer_annotation = answer_annotation[0]
                    answer_start = answer_annotation["start_token"]
                    answer_end = answer_annotation["end_token"]
                    answer_list.append(answer_end - answer_start + 1)
#                    print(passage_text_tokens[answer_start:answer_end])


            _, vocabulary = get_vocabulary([" ".join(original_tokens) for original_tokens in [passage_text_tokens, question_tokens]])
            vocabulary_set.update(vocabulary)

    return question_list, passage_list, answer_list, instance_list, vocabulary_set, html_token_set, question_text_list


def output(question_list, passage_list, answer_list, instance_list, vocabulary, html_token_set):
    # print("Questions", question_list, len(question_list))
    # print("Passages", passage_list, len(passage_list))
    # print("Answers", answer_list, len(answer_list))
    print("Vocab", len(vocabulary))
    print("HTML", len(html_token_set))

    passage_avg = sum(passage_list) / len(passage_list)
    avg_q = sum(question_list) / len(question_list)
    avg_a = sum(answer_list) / len(answer_list)

    # & # instances	& # passages &	# A/Q & AVG Q len	& AVG P len	 & AVG A len & Vcabulary Size
    print("&".join([str(x) for x in ["NaturalQuestions", len(instance_list), len(passage_list), "-", avg_q, passage_avg, avg_a, len(vocabulary)]]))

    # print(vocabulary)
    # print(html_token_set)
    print("----------------------------------------------")


def get_all_examples(data_dir):
    question_list, passage_list, answer_list, instance_list = [], [], [], []
    all_questions = []
    vocabulary = set()
    html_token_set = set()

    for subset in ["dev", "train"]:
        folder = os.path.join(data_dir,  subset)
        files = os.listdir(folder)
        for file in files:
            print(file)
            q_list, p_list, a_list, i_list, vocab, html_set, question_text_list = process_file(os.path.join(folder, file))
            question_list += q_list
            passage_list += p_list
            answer_list += a_list
            instance_list += i_list
            vocabulary.update(vocab)
            html_token_set.update(html_set)
            all_questions += question_text_list

            output(question_list, passage_list, answer_list, set(instance_list), vocabulary, html_token_set)

    return vocabulary, all_questions

task_name = "naturalquestions"
data_dir = datapaths[task_name]
vocabulary, all_questions = get_all_examples(data_dir)
print("NaturalQuestions VOCAB size:", len(vocabulary))
write_vocabulary(vocabulary, task_name)
write_questions(all_questions, task_name)
