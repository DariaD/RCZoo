import json
import os

from spacy.lang.en import English

nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)


def process_file(filename):
    vocabulary, html_token_set = set(), set()
    with open(filename, "r", encoding="utf-8") as reader:
        question_list, passage_list, answer_list, instance_list = [], [], [], []
        for i, data in enumerate(reader):
            entry = json.loads(data)
            passage_tokens = entry["string_sequence"]
            question_tokens = entry["question_string_sequence"]

            question_list.append(len(question_tokens))
            passage_list.append(len(passage_tokens))
            instance_list.append(entry["key"])

            answer_tokens = entry["answer_string_sequence"]
            answer_list.append(len(answer_tokens))

            text_to_tokenize = " ".join(passage_tokens + question_tokens + answer_tokens)
            tokens = tokenizer(text_to_tokenize)
            vocabulary.update({x.lemma_.lower() for x in tokens})
            # break

    return question_list, passage_list, answer_list, instance_list, vocabulary, html_token_set


def output(question_list, passage_list, answer_list, instance_list, vocabulary):
    # print("Questions", question_list, len(question_list))
    # print("Passages", passage_list, len(passage_list))
    # print("Answers", answer_list, len(answer_list))
    # print("Vocab", len(vocabulary))

    passage_avg = sum(passage_list) / len(passage_list)
    avg_q = sum(question_list) / len(question_list)
    avg_a = sum(answer_list) / len(answer_list)

    # & # instances	& # passages &	# A/Q & AVG Q len	& AVG P len	 & AVG A len & Vcabulary Size
    print("&".join([str(x) for x in ["WikiReading", len(instance_list), len(passage_list), "-", avg_q, passage_avg, avg_a, len(vocabulary)]]))

    #    print("----------------------------------------------")


def process_examples(data_dir):
    question_list, passage_list, answer_list, instance_list = [], [], [], []
    vocabulary = set()
    html_token_set = set()

    # read dev file
    files = os.listdir(data_dir)
    i = 0
    for file in files:
        if "tar" in file:
            continue
        if not "test-00000" in file:
            continue
        i+=1
        print(i, file)
        q_list, p_list, a_list, i_list, vocab, html_set = process_file(data_dir + file)
        question_list += q_list
        passage_list += p_list
        answer_list += a_list
        instance_list += i_list
        vocabulary.update(vocab)
        html_token_set.update(html_set)

        output(question_list, passage_list, answer_list, set(instance_list), vocabulary)
        break

    print("Files complite")

    print("writing vocabulary...")
    vocabulary_file_name = "vocabulary/WikiReading.txt"
    vocabulary_file = open(vocabulary_file_name, "w")
    for v in sorted(vocabulary):
        string_out = v + "\n"
        vocabulary_file.write(string_out)
    vocabulary_file.close()

data_dir = "/home/pinecone/Data/WikiReading/"
process_examples(data_dir)