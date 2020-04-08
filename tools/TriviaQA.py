import utils.utils
import utils.dataset_utils
import os
from tqdm import tqdm
import random
import nltk
import argparse

import json
import os
import gzip

from spacy.lang.en import English

nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)


def output(question_list, passage_list, answer_list, instance_list, vocabulary):
    # print("Questions", question_list, len(question_list))
    # print("Passages", passage_list, len(passage_list))
    # print("Answers", answer_list, len(answer_list))
    print("Vocab", len(vocabulary))

    passage_avg = sum(passage_list) / len(passage_list)
    avg_q = sum(question_list) / len(question_list)
    avg_a = sum(answer_list) / len(answer_list)

    # & # instances	& # passages &	# A/Q & AVG Q len	& AVG P len	 & AVG A len & Vcabulary Size
    print("&".join([str(x) for x in ["TriviaQA", len(instance_list), len(passage_list), "-", avg_q, passage_avg, avg_a, len(vocabulary)]]))

    #print(vocabulary)
    print("----------------------------------------------")


def process_all_data(data_dir, web_dir, wiki_dir):
    question_list, passage_list, answer_list, instance_list = [], [], [], []
    vocabulary = set()

    # read dev file

    files = os.listdir(data_dir)
    for file in files:
        if "verified" in file:
            continue
        print(file)
        q_list, p_list, a_list, i_list, vocab = process_file(data_dir + file, web_dir, wiki_dir)
        question_list += q_list
        passage_list += p_list
        answer_list += a_list
        instance_list += i_list
        vocabulary.update(vocab)

        output(question_list, passage_list, answer_list, set(instance_list), vocabulary)
        #break


    print("writing vocabulary...")
    vocabulary_file_name = "vocabulary/TriviaQA.txt"
    vocabulary_file = open(vocabulary_file_name, "w")
    for v in sorted(vocabulary):
        string_out = v + "\n"
        vocabulary_file.write(string_out)
    vocabulary_file.close()



def get_file_contents(filename, encoding='utf-8'):
    with open(filename, encoding=encoding) as f:
        content = f.read()
    return content


def read_json(filename, encoding='utf-8'):
    contents = get_file_contents(filename, encoding=encoding)
    return json.loads(contents)


def get_text(qad, domain, web_dir, wikipedia_dir):
    local_file = os.path.join(web_dir, qad['Filename']) if domain == 'SearchResults' else os.path.join(
        wikipedia_dir, qad['Filename'])
    return get_file_contents(local_file, encoding='utf-8')



def read_clean_part(datum):
    for key in ['EntityPages', 'SearchResults']:
        new_page_list = []
        for page in datum.get(key, []):
            if page['DocPartOfVerifiedEval']:
                new_page_list.append(page)
        datum[key] = new_page_list
    assert len(datum['EntityPages']) + len(datum['SearchResults']) > 0
    return datum


def read_triviaqa_data(qajson):
    data = read_json(qajson)
    # read only documents and questions that are a part of clean data set
    if data['VerifiedEval']:
        clean_data = []
        for datum in data['Data']:
            if datum['QuestionPartOfVerifiedEval']:
                if data['Domain'] == 'Web':
                    datum = read_clean_part(datum)
                clean_data.append(datum)
        data['Data'] = clean_data
    return data


def add_triple_data(datum, page, domain, keys_list):
    qad = {'Source': domain}
    #for key in ['QuestionId', 'Question', 'Answer']:
    for key in keys_list:
        qad[key] = datum[key]
    for key in page:
        qad[key] = page[key]
    return qad


def get_qad_triples(data, keys_list):
    qad_triples = []
    for datum in data['Data']:
        for key in ['EntityPages', 'SearchResults']:
            for page in datum.get(key, []):
                qad = add_triple_data(datum, page, key, keys_list)
                qad_triples.append(qad)
    return qad_triples


def process_file(qa_json_file, web_dir, wikipedia_dir):
    question_list, passage_list, answer_list, instance_list = [], [], [], []
    vocabulary = set()

    qa_json = read_triviaqa_data(qa_json_file)
    keys_list = ['QuestionId', 'Question', 'Answer']
    if "without-answer" in qa_json_file:
        keys_list = ['QuestionId', 'Question']

    qad_triples = get_qad_triples(qa_json, keys_list)

    for qad in tqdm(qad_triples):
        qid = qad['QuestionId']

        passage = get_text(qad, qad['Source'], web_dir, wikipedia_dir)
        passage_tokens = tokenizer(passage)
        question = qad['Question']
        question_tokens = tokenizer(question)

        answer_tokens_list = []
        if "without-answer" not in qa_json_file:
            answer_aliases = qad['Answer']['NormalizedAliases']
            for answer in answer_aliases:
                answer_tokens = tokenizer(answer)
                vocabulary.update({x.lemma_.lower() for x in answer_tokens})
                answer_tokens_list.append(len(answer_tokens))
            answer_list += answer_tokens_list

        vocabulary.update({x.lemma_.lower() for x in passage_tokens})
        vocabulary.update({x.lemma_.lower() for x in question_tokens})

        question_list.append(len(question_tokens))
        passage_list.append(len(passage_tokens))

        #break

    return question_list, passage_list, answer_list, instance_list, vocabulary


if __name__ == '__main__':
    web_dir = "/home/pinecone/Data/TriviaQA/evidence/web/"
    wikipedia_dir = "/home/pinecone/Data/TriviaQA/evidence/wikipedia/"
    question_dir = "/home/pinecone/Data/TriviaQA/qa/"
    process_all_data(question_dir, web_dir, wikipedia_dir)
