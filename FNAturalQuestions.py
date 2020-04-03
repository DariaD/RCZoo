import json
import os
import csv
import gzip
import jsonlines

from spacy.lang.en import English

nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)


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
        with gzip.open(folder+file) as f:
            for line in f:
                entry = json.loads(line)
                passage_tokens = entry["document_tokens"]
                passage_text_tokens = []
                for passage_token in passage_tokens:
                    # print(passage_token)
                    # print(type(passage_token))
                    token = passage_token["token"]
                    if passage_token["html_token"]:
                        html_token_set.add(passage_token["token"])
                    else:
                        passage_text_tokens.append(token)

                question_tokens = entry["question_tokens"]

                question_list.append(len(question_tokens))
                passage_list.append(len(passage_text_tokens))



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

                #text_to_tokenize = " ".join([x["token"] for x in passage_tokens] + question_tokens + [answer])
                text_to_tokenize = " ".join(passage_text_tokens + question_tokens + [answer])
                tokens = tokenizer(text_to_tokenize)
                vocabulary.update({x.lemma_.lower() for x in tokens})
                break


        print(question_list, len(question_list))
        print(passage_list, len(passage_list))
        print(answer_list, len(answer_list))
        print(len(vocabulary))

        passage_avg = sum(passage_list) / len(passage_list)
        avg_q = sum(question_list) / len(question_list)
        avg_a = sum(answer_list) / len(answer_list)

        # & # instances	& # passages &	# A/Q & AVG Q len	& AVG P len	 & AVG A len & Vcabulary Size
        print("&".join([str(x) for x in ["", "-", len(passage_list), "-", avg_q, passage_avg, avg_a,
                         len(vocabulary)]]))

        #print(vocabulary)
        print(html_token_set)

    print("DEV complite")

    # train reading
    folder  = data_dir + "train/"
    files = os.listdir(folder)
    for file in files:
    #input_file = "{}v1.0-simplified%2Fsimplified-nq-train.jsonl.gz".format(data_dir)
        with gzip.open(folder+file) as f:
            for line in f:
                entry = json.loads(line)
                print(type(entry))
                print(entry.keys())
                question = entry["question_text"]
                q_tokens = tokenizer(question)
                vocabulary.update({x.lemma_.lower() for x in q_tokens})
                question_list.append(len(q_tokens))

                passage = entry["document_text"]
                passage_tokens = passage.split()
                passage_text_tokens = []
                for token in passage_tokens:
                    if token not in html_token_set:
                        passage_text_tokens.append(token)
                p_tokens = tokenizer(" ".join(passage_text_tokens))
                vocabulary.update({x.lemma_.lower() for x in p_tokens})
                passage_list.append(len(passage_text_tokens))


                annotation = entry["annotations"][0]
                if annotation["yes_no_answer"] != "NONE":
                    answer_list.append(1)

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
                break
            break


    print(question_list, len(question_list))
    print(passage_list, len(passage_list))
    print(answer_list, len(answer_list))
    print("VOCAB", len(vocabulary))

    passage_avg = sum(passage_list) / len(passage_list)
    avg_q = sum(question_list) / len(question_list)
    avg_a = sum(answer_list) / len(answer_list)

    # & # instances	& # passages &	# A/Q & AVG Q len	& AVG P len	 & AVG A len & Vcabulary Size
    print("&".join([str(x) for x in ["", "-", len(passage_list), "-", avg_q, passage_avg, avg_a,
                     len(vocabulary)]]))

    print("writing vocabulary...")
    vocabulary_file_name = "vocabulary/NaturalQuestions.txt"
    vocabulary_file = open(vocabulary_file_name, "w")
    for v in sorted(vocabulary):
        string_out = v + "\n"
        vocabulary_file.write(string_out)
    vocabulary_file.close()

#data_dir = "/home/pinecone/Data/NaturalQuestions/"
data_dir = "/home/pinecone/Data/NaturalQuestions/v1.0/v1.0/"
get_all_examples(data_dir)

