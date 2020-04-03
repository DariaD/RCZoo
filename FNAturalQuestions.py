import json
import os
import gzip

from spacy.lang.en import English

nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)


def process_file(filename):
    question_list, passage_list, answer_list, instance_list = [], [], [], []
    vocabulary, html_token_set = set(), set()
    with gzip.open(filename) as f:
        for line in f:
            entry = json.loads(line)
            passage_tokens = entry["document_tokens"]
            passage_text_tokens = []
            for passage_token in passage_tokens:
                token = passage_token["token"]
                if passage_token["html_token"]:
                    html_token_set.add(passage_token["token"])
                else:
                    passage_text_tokens.append(token)

            question_tokens = entry["question_tokens"]

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

            text_to_tokenize = " ".join(passage_text_tokens + question_tokens + [answer])
            tokens = tokenizer(text_to_tokenize)
            vocabulary.update({x.lemma_.lower() for x in tokens})
            #break

    return question_list, passage_list, answer_list, instance_list, vocabulary, html_token_set


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
    vocabulary = set()
    html_token_set = set()

    # read dev file
    #input_file = "{}v1.0-simplified%2Fnq-dev-all.jsonl.gz".format(data_dir)
    folder  = data_dir + "dev/"
    files = os.listdir(folder)
    for file in files:
        print(file)
        q_list, p_list, a_list, i_list, vocab, html_set = process_file(folder + file)
        question_list += q_list
        passage_list += p_list
        answer_list += a_list
        instance_list += i_list
        vocabulary.update(vocab)
        html_token_set.update(html_set)

        output(question_list, passage_list, answer_list, set(instance_list), vocabulary, html_token_set)

    print("DEV complite")

    folder  = data_dir + "train/"
    files = os.listdir(folder)
    for file in files:
        print(file)
        q_list, p_list, a_list, i_list, vocab, html_set = process_file(folder + file)
        question_list += q_list
        passage_list += p_list
        answer_list += a_list
        instance_list += i_list
        vocabulary.update(vocab)
        html_token_set.update(html_set)

        output(question_list, passage_list, answer_list, set(instance_list), vocabulary, html_token_set)

    print("writing vocabulary...")
    vocabulary_file_name = "vocabulary/NaturalQuestions.txt"
    vocabulary_file = open(vocabulary_file_name, "w")
    for v in sorted(vocabulary):
        string_out = v + "\n"
        vocabulary_file.write(string_out)
    vocabulary_file.close()

# data_dir = "/home/pinecone/Data/NaturalQuestions/"
data_dir = "/home/pinecone/Data/NaturalQuestions/v1.0/v1.0/"
get_all_examples(data_dir)

