import json
import os
import csv
import gzip
import jsonlines
from xml.dom import minidom

import pandas as pd
from tqdm import tqdm

# from WikiQA import WikiReaderIterable
from utils import DataProcessor, RCExample, write_vocabulary, get_all_subfiles




class IIRCProcessor(DataProcessor):


    def get_all_examples(self, data_dir, debug_flag):
        train_data = self._read_questions_examples(data_dir, "train", debug_flag)
        dev_data   = self._read_questions_examples(data_dir, "dev", debug_flag)
        #dev_data   = self._read_questions_examples(data_dir, "test",  debug_flag)

        examples = train_data["examples"] + dev_data["examples"]

        question_list = train_data["questions"] + dev_data["questions"]
        answer_list   = train_data["answers"]   + dev_data["answers"]
        passage_list  =  train_data["passages"]   + dev_data["passages"]

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": [],
                "answers": answer_list}


    def _read_questions_examples(self, data_dir, sub_set, debug_flag):
        """Creates examples for the training and dev sets."""
        examples = []
        question_list, answer_list, passage_list, instance_list = [], [], [], []
        input_file =  os.path.join(data_dir, "{}.json".format(sub_set))

        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)


        for entry in input_data:
            local_answer_list = []
            paragraph_list = [entry["text"]]
            for q_entry in entry["questions"]:
                for context in q_entry["context"]:
                    if context['passage'] != "main":
                        paragraph_list.append(context["text"])
                if q_entry["answer"]["type"] == "span":
                    local_answer_list += [ x["text"] for x in q_entry["answer"]["answer_spans"]]
                elif q_entry["answer"]["type"] == "none":
                    pass
                else:
                    #print(q_entry["answer"])
                    local_answer_list.append(q_entry["answer"]["answer_value"])
            local_question_list = [x["question"] for x in entry["questions"]]

            passage_list+=paragraph_list
            question_list+=local_question_list
            answer_list+=list(set(question_list))

            example = RCExample(
                guid="",
                passage=paragraph_list,
                question=local_question_list,
                answer=local_answer_list,
                instance=None
            )
            examples.append(example)
            if debug_flag:
                print(example)
                break

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}




class SubjQAProcessor(DataProcessor):


    def get_all_examples(self, data_dir, debug_flag):
        train_data = self._read_questions_examples(data_dir, "train", debug_flag)
        dev_data   = self._read_questions_examples(data_dir, "dev", debug_flag)
        test_data  = self._read_questions_examples(data_dir, "test", debug_flag)

        examples = train_data["examples"] + dev_data["examples"] + test_data["examples"]

        question_list = train_data["questions"] + dev_data["questions"] + test_data["questions"]
        answer_list   = train_data["answers"]   + dev_data["answers"]   + test_data["answers"]
        passage_list  = train_data["passages"]  + dev_data["passages"]  + test_data["passages"]
        instance_list = set(train_data["instances"] + dev_data["instances"] + test_data["instances"])

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}


    def _read_questions_examples(self, data_dir, subset, debug_flag):
        """Creates examples for the training and dev sets."""
        examples = []
        question_list, passage_list, answer_list, instance_list = [], [], [], []

        for domain in ["books", "electronics", "grocery", "movies", "restaurants", "tripadvisor"]:

            filename = os.path.join(data_dir, domain, "splits", "{}.csv".format(subset))
            data = pd.read_csv(filename)

            for index, entry in data.iterrows():
                passage = entry["review"]
                question = entry["question"]

                instance_list.append(entry["item_id"])
                passage_list.append(passage)
                question_list.append(question)
                answer = entry["human_ans_spans"]
                answer_list.append(answer)

                example = RCExample(
                    guid="",
                    passage=[passage],
                    question=[question],
                    answer=[answer],
                    instance=entry["item_id"]
                )
                examples.append(example)
                if debug_flag:
                    print(example)
                    break

            print(domain, len(examples))
        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}


class QASCProcessor(DataProcessor):


    def get_all_examples(self, data_dir, debug_flag):
        train_data = self._read_questions_examples(data_dir, "train", debug_flag)
        dev_data   = self._read_questions_examples(data_dir,  "dev", debug_flag)
        test_data  = self._read_questions_examples(data_dir,  "test", debug_flag)

        examples      = train_data["examples"]      + dev_data["examples"]  + test_data["examples"]
        question_list = train_data["questions"]     + dev_data["questions"] + test_data["questions"]
        answer_list   = train_data["answers"]       + dev_data["answers"]   + test_data["answers"]
        instance_list = set(train_data["instances"] + dev_data["instances"] + test_data["instances"])

        corpus_file = os.path.join(data_dir, "QASC_Corpus.txt")

        passage_list = []
        with open(corpus_file, "r") as reader:
            if debug_flag:
                passage_list.append(reader.readline())
            else:
                passage_list = reader.readlines()

        print(len(passage_list))

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}


    def _read_questions_examples(self, data_dir, sub_set, debug_flag):
        """Creates examples for the training and dev sets."""
        examples = []
        question_list, passage_list, answer_list, instance_list = [], [], [], []
        input_file = os.path.join(data_dir, "{}.jsonl".format(sub_set))
        with jsonlines.open(input_file, "r") as reader:
            for entry in reader.iter():
                question = entry["question"]["stem"]
                question_list.append(question)
                answers = [x["text"] for x in entry["question"]["choices"]]
                answer_list.append(answers)
                example = RCExample(
                    guid="",
                    passage="",
                    question=[question],
                    answer=answers,
                    instance=None
                )
                examples.append(example)
                if debug_flag:
                    print(example)
                    break

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}


class BiPaRProcessor(DataProcessor):


    def get_all_examples(self, data_dir, debug_flag):
        train_data = self._read_questions_examples(data_dir, "train", debug_flag)
        dev_data   = self._read_questions_examples(data_dir, "valid", debug_flag)
        test_data   = self._read_questions_examples(data_dir, "test",  debug_flag)

        examples = train_data["examples"] + dev_data["examples"] + test_data["examples"]

        question_list = train_data["questions"] + dev_data["questions"]+ test_data["questions"]
        answer_list   = train_data["answers"]   + dev_data["answers"]+ test_data["answers"]
        passage_list  =  train_data["passages"]   + dev_data["passages"]+ test_data["passages"]

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": [],
                "answers": answer_list}


    def _read_questions_examples(self, data_dir, sub_set, debug_flag):
        """Creates examples for the training and dev sets."""
        examples = []
        question_list, answer_list, passage_list, instance_list = [], [], [], []
        input_file =  os.path.join(data_dir, "Monolingual_EN_{}.json".format(sub_set))

        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]

        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                passage = paragraph["context"]
                passage_list.append(passage)

                local_question_list = []
                local_answer_list = []
                answers = []
                for qas in paragraph["qas"]:
                    question = qas["question"]
                    question_list.append(question)
                    local_question_list.append(question)
                    answers = [x["text"] for x in qas["answers"]]
                    answer_list+=list(set(answers))
                    local_answer_list+=list(set(answers))

                example = RCExample(
                    guid="",
                    passage=[passage],
                    question=local_question_list,
                    answer=local_answer_list,
                    instance=None
                )
                examples.append(example)
                if debug_flag:
                    print(example)
                    break
            if debug_flag:
                break


        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}


class ARCProcessor(DataProcessor):


    def get_all_examples(self, data_dir, debug_flag):
        train_data = self._read_questions_examples(data_dir, "Challenge-Train", debug_flag)
        dev_data   = self._read_questions_examples(data_dir,  "Challenge-Dev", debug_flag)
        test_data  = self._read_questions_examples(data_dir,  "Challenge-Test", debug_flag)
        train_data_easy = self._read_questions_examples(data_dir, "Easy-Train", debug_flag)
        dev_data_easy   = self._read_questions_examples(data_dir, "Easy-Dev", debug_flag)
        test_data_easy  = self._read_questions_examples(data_dir, "Easy-Test", debug_flag)

        examples = train_data["examples"] + dev_data["examples"] + test_data["examples"] + train_data_easy["examples"] + dev_data_easy["examples"] + test_data_easy["examples"]

        question_list = train_data["questions"] + dev_data["questions"] + test_data["questions"] + train_data_easy["questions"] + dev_data_easy["questions"] + test_data_easy["questions"]
        answer_list   = train_data["answers"]   + dev_data["answers"]   + test_data["answers"]   + train_data_easy["answers"]   + dev_data_easy["answers"]   + test_data_easy["answers"]
        instance_list = set(train_data["instances"] + dev_data["instances"] + test_data["instances"] + train_data_easy["instances"] + dev_data_easy["instances"] + test_data_easy["instances"])

        return {"examples": examples,
                "questions": question_list,
                "passages": [],
                "instances": instance_list,
                "answers": answer_list}


    def _read_questions_examples(self, data_dir, sub_set, debug_flag):
        """Creates examples for the training and dev sets."""
        examples = []
        question_list, passage_list, answer_list, instance_list = [], [], [], []
        input_file = "{}/ARC-{}.jsonl".format(data_dir, sub_set)
        with jsonlines.open(input_file, "r") as reader:
            for entry in reader.iter():
                question = entry["question"]["stem"]
                question_list.append(question)
                answers = [x["text"] for x in entry["question"]["choices"]]
                answer_list.append(answers)
                example = RCExample(
                    guid="",
                    passage="",
                    question=[question],
                    answer=answers,
                    instance=None
                )
                examples.append(example)
                if debug_flag:
                    print(example)
                    break

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}


class OpenBookQAProcessor(DataProcessor):


    def get_all_examples(self, data_dir, debug_flag):
        train_data = self._read_questions_examples(data_dir, "train", debug_flag)
        dev_data   = self._read_questions_examples(data_dir,  "dev", debug_flag)
        test_data  = self._read_questions_examples(data_dir,  "test", debug_flag)

        examples        = train_data["examples"]   + dev_data["examples"] + test_data["examples"]
        question_list   = train_data["questions"] + dev_data["questions"] + test_data["questions"]
        passage_list    = train_data["passages"] + dev_data["passages"] + test_data["passages"]
        answer_list     = train_data["answers"]   + dev_data["answers"]   + test_data["answers"]
        instance_list   = set(train_data["instances"] + dev_data["instances"] + test_data["instances"])

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}


    def _read_questions_examples(self, data_dir, sub_set, debug_flag):
        """Creates examples for the training and dev sets."""
        examples = []
        question_list, passage_list, answer_list, instance_list = [], [], [], []
        input_file = "{}/{}_complete.jsonl".format(data_dir, sub_set)
        with jsonlines.open(input_file, "r") as reader:
            for entry in reader.iter():
                question = entry["question"]["stem"]
                question_list.append(question)
                answers = [x["text"] for x in entry["question"]["choices"]]
                answer_list.append(answers)
                passage =  entry["fact1"]
                example = RCExample(
                    guid="",
                    passage=[passage],
                    question=[question],
                    answer=answers,
                    instance=None
                )
                examples.append(example)
                if debug_flag:
                    print(example)
                    break

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}






class QuAILProcessor(DataProcessor):


    def get_all_examples(self, data_dir, debug_flag):
        train_data = self._read_questions_examples(data_dir, "train_questions", debug_flag)
        dev_data = self._read_questions_examples(data_dir, "dev_questions", debug_flag)

        examples = train_data["examples"] + dev_data["examples"]

        question_list = train_data["questions"] + dev_data["questions"]
        answer_list = train_data["answers"] + dev_data["answers"]
        passage_list = train_data["passages"] + dev_data["passages"]
        instance_list = set(train_data["instances"] + dev_data["instances"])

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}


    def _read_questions_examples(self, data_dir, sub_set, debug_flag):
        """Creates examples for the training and dev sets."""
        examples = []
        question_list, answer_list, passage_list, instance_list = [], [], [], []
        input_file = os.path.join(data_dir, "{}.json".format(sub_set))

        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)

        data = input_data["data"]
        for entry_id in data:
            entry = data[entry_id]
            #print(entry)
            passage = entry["context"]
            if len(passage) == 0:
                continue
            passage_list.append(passage)

            local_questions = []
            local_answers = []
            for qas in list(entry["questions"].values()):
                #print(qas)
                question = qas["question"]
                question_list.append(question)
                local_questions.append(question)
                answers = list(qas["answers"].values())
                #print(answers)
                local_answers+=answers
                answer_list+=answers

            example = RCExample(
                guid=entry_id,
                passage=[passage],
                question=local_questions,
                answer=local_answers,
                instance=None
            )
            examples.append(example)
            if debug_flag:
                print(example)
                break

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}



class TweetQAProcessor(DataProcessor):


    def get_all_examples(self, data_dir, debug_flag):
        train_data = self._read_questions_examples(data_dir, "train", debug_flag)
        dev_data =   self._read_questions_examples(data_dir, "dev",   debug_flag)
        test_data =  self._read_questions_examples(data_dir, "test",  debug_flag)

        examples = train_data["examples"] + dev_data["examples"] + test_data["examples"]

        question_list = train_data["questions"] + dev_data["questions"] + test_data["questions"]
        answer_list   = train_data["answers"]   + dev_data["answers"]   + test_data["answers"]
        passage_list  = train_data["passages"]  + dev_data["passages"]  + test_data["passages"]
        instance_list = set(train_data["instances"] + dev_data["instances"] + test_data["instances"])

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": [],
                "answers": answer_list}


    def _read_questions_examples(self, data_dir, sub_set, debug_flag):
        """Creates examples for the training and dev sets."""
        examples = []
        question_list, answer_list, passage_list, instance_list = [], [], [], []
        input_file = os.path.join(data_dir, "{}.json".format(sub_set))

        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)

        for entry in input_data:
            passage = entry["Tweet"]
            passage_list.append(passage)

            question = entry["Question"]
            question_list.append(question)
            answers = []
            if sub_set is not "test":
                answers = entry["Answer"]
                answer_list+=answers

            example = RCExample(
                guid=entry["qid"],
                passage=[passage],
                question=[question],
                answer=answers,
                instance=None
            )
            examples.append(example)
            if debug_flag:
                print(example)
                break

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}



class ReCoRDProcessor(DataProcessor):


    def get_all_examples(self, data_dir, debug_flag):
        train_data = self._read_questions_examples(data_dir, "train", debug_flag)
        dev_data = self._read_questions_examples(data_dir, "dev", debug_flag)

        examples = train_data["examples"] + dev_data["examples"]

        question_list = train_data["questions"] + dev_data["questions"]
        answer_list = train_data["answers"] + dev_data["answers"]
        passage_list = train_data["passages"] + dev_data["passages"]
        instance_list = set(train_data["instances"] + dev_data["instances"])

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}


    def _read_questions_examples(self, data_dir, sub_set, debug_flag):
        """Creates examples for the training and dev sets."""
        examples = []
        question_list, answer_list, passage_list, instance_list = [], [], [], []
        input_file = os.path.join(data_dir, "{}.json".format(sub_set))

        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)

        for entry in input_data["data"]:
            instance_list.append(entry["id"])
            passage = entry["passage"]["text"].strip()
            if len(passage) == 0:
                continue
            passage_list.append(passage)

            local_questions = []
            local_answers = []
            for qas in entry["qas"]:
                question = qas["query"]
                question_list.append(question)
                local_questions.append(question)
                for answer_entry in qas["answers"]:
                    answer = answer_entry["text"]
                    if answer not in local_answers:
                        local_answers.append(answer)
                        answer_list.append(answer)

            example = RCExample(
                guid="",
                passage=[passage],
                question=local_questions,
                answer=local_answers,
                instance=None
            )
            examples.append(example)
            if debug_flag:
                print(example)
                break

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}



class SciQProcessor(DataProcessor):


    def get_all_examples(self, data_dir, debug_flag):
        train_data = self._read_questions_examples(data_dir, "train", debug_flag)
        dev_data = self._read_questions_examples(data_dir, "valid", debug_flag)
        test_data = self._read_questions_examples(data_dir, "test", debug_flag)

        examples = train_data["examples"] + dev_data["examples"] + test_data["examples"]

        question_list = train_data["questions"] + dev_data["questions"] + test_data["questions"]
        answer_list = train_data["answers"] + dev_data["answers"] + test_data["answers"]
        passage_list = train_data["passages"] + dev_data["passages"] + test_data["passages"]
        instance_list = set(train_data["instances"] + dev_data["instances"] + test_data["instances"])

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": [],
                "answers": answer_list}


    def _read_questions_examples(self, data_dir, sub_set, debug_flag):
        """Creates examples for the training and dev sets."""
        examples = []
        question_list, answer_list, passage_list, instance_list = [], [], [], []
        input_file = os.path.join(data_dir, "{}.json".format(sub_set))

        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)

        for entry in input_data:
            passage = entry["support"].strip()
            if len(passage) == 0:
                continue
            passage_list.append(passage)

            question = entry["question"]
            question_list.append(question)
            answers = []
            answers.append(entry["distractor1"])
            answers.append(entry["distractor2"])
            answers.append(entry["distractor3"])
            answers.append(entry["correct_answer"])

            answer_list+=answers

            example = RCExample(
                guid="",
                passage=[passage],
                question=[question],
                answer=answers,
                instance=None
            )
            examples.append(example)
            if debug_flag:
                print(example)
                break

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}



class TriviaQAProcessor(DataProcessor):

    def get_all_examples(self, data_dir, debug_flag):
        examples = []
        question_list, passage_list, answer_list, instance_list = [], [], [], []

        web_dir = os.path.join(data_dir, "evidence", "web")
        wikipedia_dir = os.path.join(data_dir, "evidence", "wikipedia")
        question_dir = os.path.join(data_dir, "qa/")

        files = os.listdir(question_dir)
        for file in files:
            if "verified" in file:
                continue
            print(file)
            examples, q_list, p_list, a_list, i_list = self._process_file(question_dir + file, web_dir, wikipedia_dir, debug_flag)
            question_list += q_list
            passage_list += p_list
            answer_list += a_list
            instance_list += i_list
            examples += examples

            if debug_flag:
                break

        # print("Passage", passage_list)
        # print("Question", question_list)
        # print("Answer", answer_list)



        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}

    def _get_file_contents(self, filename, encoding='utf-8'):
        with open(filename, encoding=encoding) as f:
            content = f.read()
        return content

    def _read_json(self, filename, encoding='utf-8'):
        contents = self._get_file_contents(filename, encoding=encoding)
        return json.loads(contents)

    def _get_text(self, qad, domain, web_dir, wikipedia_dir):
        local_file = os.path.join(web_dir, qad['Filename']) if domain == 'SearchResults' else os.path.join(
            wikipedia_dir, qad['Filename'])
        return self._get_file_contents(local_file, encoding='utf-8')

    def _read_clean_part(self, datum):
        for key in ['EntityPages', 'SearchResults']:
            new_page_list = []
            for page in datum.get(key, []):
                if page['DocPartOfVerifiedEval']:
                    new_page_list.append(page)
            datum[key] = new_page_list
        assert len(datum['EntityPages']) + len(datum['SearchResults']) > 0
        return datum

    def _read_triviaqa_data(self, qajson):
        data = self._read_json(qajson)
        # read only documents and questions that are a part of clean data set
        if data['VerifiedEval']:
            clean_data = []
            for datum in data['Data']:
                if datum['QuestionPartOfVerifiedEval']:
                    if data['Domain'] == 'Web':
                        datum = self._read_clean_part(datum)
                    clean_data.append(datum)
            data['Data'] = clean_data
        return data

    def _add_triple_data(self, datum, page, domain, keys_list):
        qad = {'Source': domain}
        # for key in ['QuestionId', 'Question', 'Answer']:
        for key in keys_list:
            qad[key] = datum[key]
        for key in page:
            qad[key] = page[key]
        return qad

    def _get_qad_triples(self, data, keys_list):
        qad_triples = []
        for datum in data['Data']:
            for key in ['EntityPages', 'SearchResults']:
                for page in datum.get(key, []):
                    qad = self._add_triple_data(datum, page, key, keys_list)
                    qad_triples.append(qad)
        return qad_triples


    def _process_file(self, qa_json_file, web_dir, wikipedia_dir, debug_flag):
        question_list, passage_list, answer_list, instance_list = [], [], [], []
        examples = []

        qa_json = self._read_triviaqa_data(qa_json_file)
        keys_list = ['QuestionId', 'Question', 'Answer']
        if "without-answer" in qa_json_file:
            keys_list = ['QuestionId', 'Question']

        qad_triples = self._get_qad_triples(qa_json, keys_list)

        for qad in tqdm(qad_triples):
            qid = qad['QuestionId']

            passage = self._get_text(qad, qad['Source'], web_dir, wikipedia_dir)
            question = qad['Question']

            answer_aliases = []
            if "without-answer" not in qa_json_file:
                answer_aliases = qad['Answer']['NormalizedAliases']
                for answer in answer_aliases:
                    answer_list.append(answer)

            question_list.append(question)
            passage_list.append(passage)
            example = RCExample(
                guid=qid,
                passage=[passage],
                question=[question],
                answer=answer_aliases,
                instance=None
            )
            examples.append(example)
            if debug_flag:
                print(example)
                break



        return examples, question_list, passage_list, answer_list, instance_list


class RACEProcessor(DataProcessor):

    def get_all_examples(self, data_dir, debug_flag):
        examples = []
        question_list, passage_list, answer_list, instance_list = [], [], [], []

        all_files_list = get_all_subfiles(data_dir, [], ".txt")
        # print(all_files_list)

        for file in all_files_list:
            with open(file, "r") as reader:
                input_data = json.load(reader)

                passage = input_data["article"].replace("\n", " ")
                questions = input_data["questions"]
                answer_options = input_data["options"]

                passage_list.append(passage)
                question_list += questions

                local_answer_list = []
                for options in answer_options:
                    answer_list+=options
                    local_answer_list+=options

                examples.append(RCExample(
                    guid="",
                    passage=[passage],
                    question=questions,
                    answer=local_answer_list,
                    instance=None
                ))
            if debug_flag:
                break

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}






class CBTestProcessor(DataProcessor):


    def get_all_examples(self, data_dir, debug_flag):
        examples = []
        question_list, passage_list, answer_list, instance_list = [], [], [], []

        files = os.listdir(data_dir)
        for file in files:
            if ".txt" not in file:
                continue

            story_input_file = os.path.join(data_dir, file)
            with open(story_input_file) as reader:
                all_lines = reader.readlines()
                paragraph_lines = []
                for line in all_lines:
                    line = line.strip()
                    if len(line) == 0:
                        continue

                    parts = line.split()
                    number = int(parts[0])
                    text = " ".join(parts[1:])

                    if "XXXXX" in text:
                        q_parts = line.split("\t")
                        question = " ".join(q_parts[0].split()[1:])
                        question_list.append(question)
                        answer = [z.strip() for z in q_parts[-1].split("|")]
                        answer_list+=answer

                        passage = " ".join(paragraph_lines)
                        passage_list.append(passage)
                        paragraph_lines = []

                        example = RCExample(
                            guid="",
                            passage=[passage],
                            question=[question],
                            answer=answer,
                            instance=None
                        )
                        examples.append(example)
                        if debug_flag:
                            print(example)
                            break

                    else:
                        paragraph_lines.append(text)
            if debug_flag:
                break

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": [],
                "answers": answer_list}




class ReClorProcessor(DataProcessor):


    def get_all_examples(self, data_dir, debug_flag):
        train_data = self._read_questions_examples(data_dir, "train", debug_flag)
        dev_data   = self._read_questions_examples(data_dir,  "val", debug_flag)
        test_data  = self._read_questions_examples(data_dir,  "test", debug_flag)

        examples = train_data["examples"] + dev_data["examples"] + test_data["examples"]

        question_list = train_data["questions"] + dev_data["questions"] + test_data["questions"]
        answer_list   = train_data["answers"]   + dev_data["answers"]   + test_data["answers"]
        passage_list  = train_data["passages"]  + dev_data["passages"]  + test_data["passages"]
        instance_list = set(train_data["instances"] + dev_data["instances"] + test_data["instances"])

        # print(question_list)
        # print(passage_list)
        # print(answer_list)
        # print(instance_list)


        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}


    def _read_questions_examples(self, data_dir, sub_set, debug_flag):
        """Creates examples for the training and dev sets."""
        examples = []
        question_list, passage_list, answer_list, instance_list = [], [], [], []
        input_file = os.path.join(data_dir, "{}.json".format(sub_set))
        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)

        for entry in input_data:
            # print(entry)
            passage = entry["context"]
            passage_list.append(passage)
            question = entry["question"]
            question_list.append(question)
            answer_list += entry["answers"]
            example = RCExample(
                guid="",
                passage=[passage],
                question=[question],
                answer=entry["answers"],
                instance=None
            )
            examples.append(example)
            if debug_flag:
                print(example)
                break

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}


class DREAMProcessor(DataProcessor):


    def get_all_examples(self, data_dir, debug_flag):
        train_data = self._read_questions_examples(data_dir, "train", debug_flag)
        dev_data   = self._read_questions_examples(data_dir,  "dev", debug_flag)
        test_data  = self._read_questions_examples(data_dir,  "test", debug_flag)

        examples = train_data["examples"] + dev_data["examples"] + test_data["examples"]

        question_list = train_data["questions"] + dev_data["questions"] + test_data["questions"]
        answer_list   = train_data["answers"]   + dev_data["answers"]   + test_data["answers"]
        passage_list  = train_data["passages"]  + dev_data["passages"]  + test_data["passages"]
        instance_list = set(train_data["instances"] + dev_data["instances"] + test_data["instances"])

        # print(question_list)
        # print(passage_list)
        # print(answer_list)
        # print(instance_list)


        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}


    def _read_questions_examples(self, data_dir, sub_set, debug_flag):
        """Creates examples for the training and dev sets."""
        examples = []
        question_list, passage_list, answer_list, instance_list = [], [], [], []
        input_file = os.path.join(data_dir, "{}.json".format(sub_set))
        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)

        for entry in input_data:
            #print(entry)
            passage = " ".join(entry[0])
            passage_list.append(passage)
            #print(entry[1])
            local_question_list = []
            local_answer_list = []
            for qa in entry[1]:
                question = qa["question"]
                question_list.append(question)
                local_question_list.append(question)
                answer_list += qa["choice"]
                local_answer_list += qa["choice"]

            example = RCExample(
                guid="",
                passage=[passage],
                question=local_question_list,
                answer=local_answer_list,
                instance=None
            )
            examples.append(example)
            if debug_flag:
                print(example)
                break

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}


class MultiRCProcessor(DataProcessor):


    def get_all_examples(self, data_dir, debug_flag):
        train_data = self._read_questions_examples(data_dir, "train", debug_flag)
        dev_data   = self._read_questions_examples(data_dir,  "val", debug_flag)
        test_data  = self._read_questions_examples(data_dir,  "test", debug_flag)

        examples = train_data["examples"] + dev_data["examples"] + test_data["examples"]

        question_list = train_data["questions"] + dev_data["questions"] + test_data["questions"]
        answer_list   = train_data["answers"]   + dev_data["answers"]   + test_data["answers"]
        passage_list  = train_data["passages"]  + dev_data["passages"]  + test_data["passages"]
        instance_list = set(train_data["instances"] + dev_data["instances"] + test_data["instances"])

        # print(question_list)
        # print(passage_list)
        # print(answer_list)
        # print(instance_list)


        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}


    def _read_questions_examples(self, data_dir, sub_set, debug_flag):
        """Creates examples for the training and dev sets."""
        examples = []
        question_list, passage_list, answer_list, instance_list = [], [], [], []
        input_file = "{}/{}.jsonl".format(data_dir, sub_set)
        with jsonlines.open(input_file, "r") as reader:
            question_count = 0
            for entry in reader.iter():
                paragraph = entry["passage"]
                passage = paragraph["text"]
                passage_list.append(passage)

                local_question_list = []
                local_answer_list = []
                for qas in paragraph["questions"]:
                    question = qas["question"]
                    question_list.append(question)
                    local_question_list.append(question)
                    question_count += 1

                    answers = [x["text"] for x in qas["answers"]]
                    answer_list += answers
                    local_answer_list+=answers

                example = RCExample(
                    guid="",
                    passage=[passage],
                    question=local_question_list,
                    answer=local_answer_list,
                    instance=None
                )
                examples.append(example)
                if debug_flag:
                    print(example)
                    break

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}


class QuACProcessor(DataProcessor):


    def get_all_examples(self, data_dir, debug_flag):
        train_data = self._read_questions_examples(data_dir, "train", debug_flag)
        dev_data   = self._read_questions_examples(data_dir, "val", debug_flag)

        examples = train_data["examples"] + dev_data["examples"]

        question_list = train_data["questions"] + dev_data["questions"]
        answer_list   = train_data["answers"]   + dev_data["answers"]
        passage_list  =  train_data["passages"]   + dev_data["passages"]

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": [],
                "answers": answer_list}


    def _read_questions_examples(self, data_dir, sub_set, debug_flag):
        """Creates examples for the training and dev sets."""
        examples = []
        question_list, answer_list, passage_list, instance_list = [], [], [], []
        input_file =  os.path.join(data_dir, "{}_v0.2.json".format(sub_set))

        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]

        for entry in input_data:
            text_list = [entry["title"], entry["background"], entry["section_title"]]
            for paragraph in entry["paragraphs"]:
                passage = " ".join([paragraph["context"]] + text_list)
                passage_list.append(passage)

                local_question_list = []
                answers = []
                for qas in paragraph["qas"]:
                    question = qas["question"]
                    question_list.append(question)
                    local_question_list.append(question)
                    answers = [x["text"] for x in (qas["answers"]+[qas["orig_answer"]])]
                    answer_list+=answers

                example = RCExample(
                    guid="",
                    passage=[passage],
                    question=local_question_list,
                    answer=answers,
                    instance=None
                )
                examples.append(example)
                if debug_flag:
                    print(example)
                    break
            if debug_flag:
                break


        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}


class CoQAProcessor(DataProcessor):


    def get_all_examples(self, data_dir, debug_flag):
        train_data = self._read_questions_examples(data_dir, "train", debug_flag)
        dev_data   = self._read_questions_examples(data_dir, "dev", debug_flag)

        examples = train_data["examples"] + dev_data["examples"]

        question_list = train_data["questions"] + dev_data["questions"]
        answer_list   = train_data["answers"]   + dev_data["answers"]
        passage_list  =  train_data["passages"]   + dev_data["passages"]

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": [],
                "answers": answer_list}


    def _read_questions_examples(self, data_dir, sub_set, debug_flag):
        """Creates examples for the training and dev sets."""
        examples = []
        question_list, answer_list, passage_list, instance_list = [], [], [], []

        input_file = "{}coqa-{}-v1.0.json".format(data_dir, sub_set)

        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]

        for entry in input_data:
            passage_list.append(entry["story"])

            local_question_list = []
            local_answer_list = []
            for qestion in entry["questions"]:
                question_list.append(qestion["input_text"])
                local_question_list.append(qestion["input_text"])
            for answer in entry["answers"]:
                answer_list.append(answer["input_text"])
                local_answer_list.append(answer["input_text"])

            example = RCExample(
                guid="",
                passage=[entry["story"]],
                question=local_question_list,
                answer=local_answer_list,
                instance=None
            )
            examples.append(example)
            if debug_flag:
                print(example)
                break

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}



class EmrQAProcessor(DataProcessor):


    def get_all_examples(self, data_dir, debug_flag):
        examples = []
        question_list, passage_list, answer_list, instance_list = [], [], [], []

        file_name = os.path.join(data_dir, "data.json")
        with open(file_name, "r") as reader:
            input_data = json.load(reader)["data"]

        for challenge in input_data:
            for entry in challenge["paragraphs"]:
                passage = entry["context"]
                if type(passage) is not str:
                    passage = " ".join(passage)
                passage = passage.replace("\n", " ")
                instance_list.append(entry["note_id"])
                passage_list.append(passage)

                # print(passage)

                local_answer_list = []
                local_question_list = []
                for qa in entry["qas"]:
                    for answer_structure in qa["answers"]:
                        answer = answer_structure["text"]
                        if len(answer) > 0:
                            if type(answer) is not str:
                                answer = " ".join(answer)
                            answer_list.append(answer)
                            local_answer_list.append(answer)
                    for question in qa["question"]:
                        question_list.append(question)
                        if question not in local_answer_list:
                            local_question_list.append(question)

                example = RCExample(
                    guid="",
                    passage=[passage],
                    question=local_question_list,
                    answer=local_answer_list,
                    instance=entry["note_id"]
                )
                examples.append(example)
                if debug_flag:
                    print(example)
                    break

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}



class CosmosQAProcessor(DataProcessor):


    def get_all_examples(self, data_dir, debug_flag):
        train_data = self._read_questions_examples(data_dir, "train", debug_flag)
        dev_data   = self._read_questions_examples(data_dir,  "valid", debug_flag)
        test_data  = self._read_test_questions_examples(data_dir,  "test", debug_flag)

        examples = train_data["examples"] + dev_data["examples"] + test_data["examples"]

        question_list = train_data["questions"] + dev_data["questions"] + test_data["questions"]
        answer_list   = train_data["answers"]   + dev_data["answers"]   + test_data["answers"]
        passage_list  = train_data["passages"]  + dev_data["passages"]  + test_data["passages"]
        instance_list = set(train_data["instances"] + dev_data["instances"] + test_data["instances"])

        # print(question_list)
        # print(passage_list)
        # print(answer_list)
        # print(instance_list)


        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}

    def _read_questions_examples(self, data_dir, subset, debug_flag):
        """Creates examples for the training and dev sets."""
        examples = []
        question_list, passage_list, answer_list, instance_list = [], [], [], []

        input_file = "{}{}.csv".format(data_dir, subset)
        with open(input_file, newline='') as csvfile:
            data = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(data)
            for entry in data:
                instance_list.append(entry[0])
                passage_list.append(entry[1])
                question_list.append(entry[2])
                answer_list.append(entry[3])
                answer_list.append(entry[4])
                answer_list.append(entry[5])
                answer_list.append(entry[6])

                example = RCExample(
                    guid="",
                    passage=[entry[1]],
                    question=[entry[2]],
                    answer=[entry[3], entry[4], entry[5], entry[6]],
                    instance=entry[0]
                )
                examples.append(example)
                if debug_flag:
                    print(example)
                    break


        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}

    def _read_test_questions_examples(self, data_dir, subset, debug_flag):
        """Creates examples for the training and dev sets."""
        examples = []
        question_list, passage_list, answer_list, instance_list = [], [], [], []

        input_file = "{}/{}.jsonl".format(data_dir, subset)
        with jsonlines.open(input_file, "r") as reader:
            for entry in reader.iter():
                instance_list.append(entry["id"])
                passage_list.append(entry["context"])
                question_list.append(entry["question"])
                answer_list.append(entry["answer0"])
                answer_list.append(entry["answer1"])
                answer_list.append(entry["answer2"])
                answer_list.append(entry["answer3"])

                example = RCExample(
                    guid="",
                    passage=[entry["context"]],
                    question=[entry["question"]],
                    answer=[entry["answer0"], entry["answer1"], entry["answer2"], entry["answer3"]],
                    instance=entry["id"]
                )
                examples.append(example)
                if debug_flag:
                    print(example)
                    break

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}


class MovieQAProcessor(DataProcessor):


    def get_all_examples(self, data_dir, debug_flag):
        passage_list, passages_dict = self._get_plot_vocabulary(data_dir)
        train_data = self._read_questions_examples(passages_dict, data_dir, "train", debug_flag)
        #dev_data   = self._read_questions_examples(passages_dict, data_dir,  "dev", debug_flag)


        examples = train_data["examples"]# + dev_data["examples"]

        question_list = train_data["questions"] #+ dev_data["questions"]
        answer_list   = train_data["answers"]   #+ dev_data["answers"]


        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": [],
                "answers": answer_list}

    def _read_questions_examples(self, passages_dict, data_dir, sub_set, debug_flag):
        """Creates examples for the training and dev sets."""
        examples = []
        question_list, answer_list = [], []

        input_file = "{}data/qa.json".format(data_dir)

        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)

        movie_questions = {}
        movie_answers = {}
        for entry in input_data:
            moview_key = entry["imdb_key"]
            if moview_key not in list(movie_questions.keys()):
                movie_questions[moview_key] = []
                movie_answers[moview_key] = []
            question_list.append(entry["question"])
            movie_questions[moview_key].append(entry["question"])
            movie_answers[moview_key]+=entry["answers"]

            local_answers = []
            for answer in entry["answers"]:
                answer_list.append(answer)
                local_answers.append(answer)

        for movie_key in list(movie_questions.keys()):
            if movie_key not in list(passages_dict.keys()):
                #print(movie_key)
                continue
            example = RCExample(
                guid="",
                passage=[passages_dict[movie_key]],
                question=movie_questions[movie_key],
                answer=movie_answers[movie_key],
                instance=None
            )
            examples.append(example)
            if debug_flag:
                print(example)
                break


        return {"examples": examples,
                "questions": question_list,
                "answers": answer_list}


    def _get_plot_vocabulary(self, folder):
        passages = []
        passages_dict = {}

        plot_directory = os.path.join(folder, "story", "plot")
        plot_files = os.listdir(plot_directory)
        for file in plot_files:
            with open(os.path.join(plot_directory, file), "r") as reader:
                file_lines = reader.readlines()
                plot_key = file[:-5] # remove wiki
                current_passage_list = []
                for text in file_lines:
                    text = text.strip()
                    if len(text) > 0:
                        current_passage_list.append(text)
                current_passage = " ".join(current_passage_list)
                passages.append(current_passage)
                passages_dict[plot_key] = current_passage

        return passages, passages_dict


class MCScriptProcessor(DataProcessor):


    def get_all_examples(self, data_dir, debug_flag):
        train_data = self._read_questions_examples(data_dir, "train", debug_flag)
        dev_data   = self._read_questions_examples(data_dir,  "dev", debug_flag)
        test_data  = self._read_questions_examples(data_dir,  "test", debug_flag)

        examples = train_data["examples"] + dev_data["examples"] + test_data["examples"]

        question_list = train_data["questions"] + dev_data["questions"] + test_data["questions"]
        answer_list   = train_data["answers"]   + dev_data["answers"]   + test_data["answers"]
        passage_list  = train_data["passages"]  + dev_data["passages"]  + test_data["passages"]
        instance_list = set(train_data["instances"] + dev_data["instances"] + test_data["instances"])

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}

    def _read_questions_examples(self, data_dir, sub_set, debug_flag):
        """Creates examples for the training and dev sets."""
        examples = []
        question_list, passage_list, answer_list, instance_list = [], [], [], []

        input_file = "{}{}-data.xml".format(data_dir, sub_set)
        xmldoc = minidom.parse(input_file)

        instancelist = xmldoc.getElementsByTagName('instance')
        for instance in instancelist:
            passage = instance.getElementsByTagName('text')[0].firstChild.nodeValue
            passage_list.append(passage)
            local_q_list = []
            local_a_list = []
            question_instance_list = instance.getElementsByTagName('question')
            for question in question_instance_list:
                question_str = question.attributes['text'].value
                question_list.append(question_str)
                local_q_list.append(question_str)
                for answer in question.getElementsByTagName('answer'):
                    answer_list.append(answer.attributes['text'].value)
                    local_a_list.append(answer.attributes['text'].value)

            example = RCExample(
                guid="",
                passage=[passage],
                question=local_q_list,
                answer=local_a_list,
                instance=None
            )
            examples.append(example)
            if debug_flag:
                print(example)
                break

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}


class AmazonYesNoProcessor(DataProcessor):


    def get_all_examples(self, data_dir, debug_flag):
        examples = []
        question_list, passage_list, answer_list, instance_list = [], [], [], []

        domain_folders = os.listdir(data_dir)
        for domain in domain_folders:
            folder = os.path.join(data_dir, domain, "yes_no_balanced")
            files = os.listdir(folder)
            for file in files:
                file_name = os.path.join(folder, file)
                with open(file_name, "r") as reader:
                    input_data = json.load(reader)

#                print(file_name)
                for entry_element in input_data:
                    for product_id in entry_element.keys():
                        instance_list.append(product_id)
                        entry = entry_element.get(product_id)
                        passage = entry["review"]
                        passage_list.append(passage)

                        local_q_list = []
                        local_a_list = []
                        for question in entry["qa"]:
                            question_list.append(question["q"])
                            local_q_list.append(question["q"])
                            answer_list.append(question["a"])
                            local_a_list.append(question["a"])

                        example = RCExample(
                            guid=product_id,
                            passage=[passage],
                            question=local_q_list,
                            answer=local_a_list,
                            instance=None
                        )
                        examples.append(example)
                        if debug_flag:
                            print(example)
                            break
                    if debug_flag:
                        break
                if debug_flag:
                    break
            if debug_flag:
                break


        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}


class MCTestProcessor(DataProcessor):


    def get_all_examples(self, data_dir, debug_flag):
        train_data = self._read_questions_examples(data_dir, "train", debug_flag)
        dev_data   = self._read_questions_examples(data_dir,  "dev", debug_flag)
        test_data  = self._read_questions_examples(data_dir,  "test", debug_flag)

        examples = train_data["examples"] + dev_data["examples"] + test_data["examples"]

        question_list = train_data["questions"] + dev_data["questions"] + test_data["questions"]
        answer_list   = train_data["answers"]   + dev_data["answers"]   + test_data["answers"]
        passage_list  = train_data["passages"]  + dev_data["passages"]  + test_data["passages"]
        instance_list = set(train_data["instances"] + dev_data["instances"] + test_data["instances"])

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}

    def _read_questions_examples(self, data_dir, subset, debug_flag):
        """Creates examples for the training and dev sets."""
        examples = []
        question_list, passage_list, answer_list, instance_list = [], [], [], []

        files = os.listdir(data_dir)
        for file in files:
            if "{}.tsv".format(subset) not in file:
                continue
            with open(os.path.join(data_dir, file)) as fd:
               # print(data_dir + file)
                rd = csv.reader(fd, delimiter="\t", quotechar='"')
                for row in rd:
                    actual_data = row[2:]
                    passage = actual_data[0].replace("\\newline", " ")
                    passage_list.append(passage)
                    local_q_list = []
                    local_a_list = []
                    for text in actual_data[1:]:
                        if "one:" in text:
                            question_list.append(text[4:].strip())
                            local_q_list.append(text[4:].strip())
                        elif "multiple:" in text:
                            question_list.append(text[9:].strip())
                            local_q_list.append(text[9:].strip())
                        else:
                            answer_list.append(text)
                            local_a_list.append(text)

                    #break

                # print(question_list)
                # print(passage_list)
                # print(answer_list)

                    example = RCExample(
                        guid="",
                        passage=[passage],
                        question=local_q_list,
                        answer=local_a_list,
                        instance=None
                    )
                    examples.append(example)
                    if debug_flag:
                        print(example)
                        break

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}




class TyDiProcessor(DataProcessor):


    def get_all_examples(self, data_dir, debug_flag):
        train_data = self._read_questions_examples(data_dir, "train", debug_flag)
        dev_data   = self._read_questions_examples(data_dir,  "dev", debug_flag)
#        print("Number of Questions", len(train_data["questions"] + dev_data["questions"]))
        train_data_gp = self._read_gold_passage(data_dir, "train", debug_flag)
        dev_data_pg   = self._read_gold_passage(data_dir,  "dev", debug_flag)

        examples = train_data["examples"] + dev_data["examples"] + train_data_gp["examples"]

        question_list = train_data["questions"] + dev_data["questions"] + train_data_gp["questions"] + dev_data_pg["questions"]
        answer_list   = train_data["answers"]   + dev_data["answers"]   + train_data_gp["answers"] + dev_data_pg["answers"]
        passage_list  = train_data["passages"]  + dev_data["passages"]  + train_data_gp["passages"] + dev_data_pg["passages"]
        instance_list = set(train_data["instances"] + dev_data["instances"] + train_data_gp["instances"] + dev_data_pg["instances"])

        # print(len(passage_list), passage_list)
        #print(len(passage_list))
        #print(len(question_list), question_list)
        #print("Number of Questions", len(question_list))
        #print(len(answer_list), answer_list)
        #print(len(instance_list), instance_list)


        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}

    def _read_questions_examples(self, folder, sub_set, debug_flag):
        """Creates examples for the training and dev sets."""
        examples = []
        question_list, passage_list, answer_list, instance_list = [], [], [], []
        input_file = "{}tydiqa-goldp-v1.1-{}.json".format(folder, sub_set)

        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]

        for entry in input_data:
            isEnglish = False
            for passage_entity in entry["paragraphs"]:
                local_questions = []
                local_answers = []
                for qas in passage_entity["qas"]:
                    if "english" in qas["id"]:
                        isEnglish = True
                        question_list.append(qas["question"])
                        local_questions.append(qas["question"])
                        for ans in qas["answers"]:
                            answer_list.append(ans["text"])
                            local_answers.append(ans["text"])
                if isEnglish:
                    passage = passage_entity["context"]
                    passage_list.append(passage_entity["context"])


                    example = RCExample(
                        guid="",
                        passage=[passage],
                        question=local_questions,
                        answer=local_answers,
                    )
                    examples.append(example)
                    if debug_flag:
                        print(example)
                        break

            if isEnglish:
                instance_list.append(entry["title"])




        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "candidates": [],
                "answers": answer_list}



    def _read_gold_passage(self, folder, sub_set, debug_flag):
        """Creates examples for the training and dev sets."""
        examples = []
        question_list, passage_list, answer_list, instance_list = [], [], [], []

        input_file = "{}tydiqa-v1.0-{}.jsonl.gz".format(folder, sub_set)
        with gzip.open(input_file) as f:
            for entry in f:
                entry = eval(entry)
                if entry["language"] == "english":
                    passage = entry["document_plaintext"]
                    passage_list.append(passage)
                    question = entry["question_text"]
                    question_list.append(question)
                    local_answers = []
                    for annotation in entry["annotations"]:
                        if annotation["yes_no_answer"] != "NONE":
                            answer_list.append(annotation["yes_no_answer"])
                            local_answers.append(annotation["yes_no_answer"])
                        else:
                            start = annotation["minimal_answer"]['plaintext_start_byte']
                            end   = annotation["minimal_answer"]['plaintext_end_byte']
                            if start != -1:
                                answer = passage.encode()[start:end].decode()
                                answer_list.append(answer)
                                local_answers.append(answer)

                    example = RCExample(
                        guid="",
                        passage=[passage],
                        question=[question],
                        answer=local_answers,
                    )
                    examples.append(example)
                    if debug_flag:
                        print(example)
                        break
                    #break


        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "candidates": [],
                "answers": answer_list}


class SearchQAProcessor(DataProcessor):


    def get_all_examples(self, data_dir, debug_flag):
        train_data = self._read_questions_examples(data_dir, "train", debug_flag)
        dev_data   = self._read_questions_examples(data_dir,  "val", debug_flag)
        test_data  = self._read_questions_examples(data_dir,  "test", debug_flag)

        examples = train_data["examples"] + dev_data["examples"] + test_data["examples"]

        question_list = train_data["questions"] + dev_data["questions"] + test_data["questions"]
        answer_list   = train_data["answers"]   + dev_data["answers"]   + test_data["answers"]
        passage_list  = train_data["passages"]  + dev_data["passages"]  + test_data["passages"]
        instance_list = set(train_data["instances"] + dev_data["instances"] + test_data["instances"])

#        print(len(passage_list), passage_list)
#         print(len(passage_list))
#         print(len(question_list), question_list)
#         print(len(answer_list), answer_list)
#         print(len(instance_list), instance_list)


        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}

    def _read_questions_examples(self, folder, sub_set, debug_flag):
        """Creates examples for the training and dev sets."""
        examples = []
        question_list, passage_list, answer_list, instance_list = [], [], [], []

        files = os.listdir(os.path.join(folder, sub_set))
        for file in files:
            input_file = os.path.join(folder, sub_set, file)

            with open(input_file, "r", encoding='utf-8') as reader:
                entry = json.load(reader)

                instance = entry["category"]
                instance_list.append(instance)
                search_results = entry["search_results"]
                local_passage_list = []
                for s_result in search_results:
                    if s_result["snippet"]:
                        passage = s_result["snippet"]
                        passage = passage.replace("\n", "")
                        #passage.replace("...", "").replace("....", "").replace(".....", "").replace("......", "")

                        passage_list.append(passage)
                        local_passage_list.append(passage)


                question = entry["question"]
                question_list.append(question)
                answer = entry["answer"]
                answer_list.append(answer)

                example = RCExample(
                    guid="",
                    passage=local_passage_list,
                    question=[question],
                    answer=[answer],
                    instance=instance
                )
                examples.append(example)

                if debug_flag:
                    print(example)
                    break

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "candidates": [],
                "answers": answer_list}


class NewsQAProcessor(DataProcessor):
    def get_all_examples(self, data_dir, debug_flag):
        """Creates examples for the training and dev sets."""
        examples = []
        question_list, passage_list, answer_list, instance_list, ansercandidate_list = [], [], [], [], []

        input_file = "{}combined-newsqa-data-v1.json".format(data_dir)

        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]

        for entry in input_data:
            instance_list.append(entry["storyId"])
            passage = entry["text"]
            passage_list.append(entry["text"])

            local_question_list = []
            local_answer_list = []
            for qas in entry["questions"]:
                question = qas["q"]
                question_list.append(question)
                local_question_list.append(question)
                answer_set = set()
                for answer_entity in qas["answers"]:
                    for potential_answer in answer_entity["sourcerAnswers"]:
                        if "noAnswer" in potential_answer.keys():
                            continue
                        start =  potential_answer["s"]
                        end =  potential_answer["e"]

                        answer = passage[start:end]
                        answer_set.update([answer.strip()])
                answer_list+=(list(answer_set))
                local_answer_list+=(list(answer_set))

            example = RCExample(
                guid="",
                passage=[passage],
                question=local_question_list,
                answer=local_answer_list,
                instance=entry["storyId"]
            )
            examples.append(example)
            if debug_flag:
                print(example)
                break

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": set(instance_list),
                "candidates": ansercandidate_list,
                "answers": answer_list}



class MsMarcoProcessor(DataProcessor):


    def get_all_examples(self, data_dir, debug_flag):

        dev_data   = self._read_questions_examples(data_dir,  "dev", debug_flag)
        test_data  = self._read_questions_examples(data_dir,  "eval", debug_flag)
        train_data = self._read_questions_examples(data_dir, "train", debug_flag)

        examples = train_data["examples"] + dev_data["examples"] + test_data["examples"]

        question_list = train_data["questions"] + dev_data["questions"] + test_data["questions"]
        answer_list   = train_data["answers"]   + dev_data["answers"]   + test_data["answers"]
        passage_list  = train_data["passages"]  + dev_data["passages"]  + test_data["passages"]
        instance_list = set(train_data["instances"] + dev_data["instances"] + test_data["instances"])

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}

    def _read_questions_examples(self, folder, set, debug_flag):
        """Creates examples for the training and dev sets."""
        examples = []
        question_list, passage_list, answer_list, instance_list = [], [], [], []

        context_input_file = os.path.join(folder, "{}_v2.1.json.gz".format(set))
        if set == "eval":
            context_input_file = os.path.join(folder, "{}_v2.1_public.json.gz".format(set))
        with gzip.open(context_input_file) as f:
            for entry in f:
                entry = eval(entry)
                query_id = entry["query_id"]

                all_passages = entry["passages"]
                queries = entry["query"]
                for keys in query_id.keys():
                    q_id = query_id[keys]
                    #print(q_id)


                    passages = all_passages[keys]
                    local_passage_list = []
                    for passage in passages:
                        passage_list.append(passage["passage_text"])
                        local_passage_list.append(passage["passage_text"])
                    #print("Passages", passages)

                    question = queries[keys]
                    #print("Question", question)
                    question_list.append(question)

                    answer = ""
                    if set != "eval":
                        answers = entry["answers"]
                        answer = answers[keys]
                        answer_list += answer
                        #print("Answer:", answer)


                    example = RCExample(
                        guid=q_id,
                        passage=local_passage_list,
                        question=[question],
                        answer=[answer],
                        instance=None
                    )
                    examples.append(example)
                    if debug_flag:
                        print(example)
                        break

                if debug_flag:
                    break


        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}



class CNNDailyMailProcessor(DataProcessor):

    def get_all_examples(self, data_dir, debug_flag):

        #questions and answers
        train_data = self._read_examples(data_dir, "training", debug_flag)
        dev_data   = self._read_examples(data_dir,  "validation", debug_flag)
        test_data  = self._read_examples(data_dir,  "test", debug_flag)

        examples = train_data["examples"] + dev_data["examples"] + test_data["examples"]

        question_list = train_data["questions"] + dev_data["questions"] + test_data["questions"]
        answer_list   = train_data["answers"]   + dev_data["answers"]   + test_data["answers"]
        passage_list  = train_data["passages"]  + dev_data["passages"]  + test_data["passages"]
        instance_list = set(train_data["instances"] + dev_data["instances"] + test_data["instances"])
        candidate_list = train_data["candidates"] + dev_data["candidates"] + test_data["candidates"]
        named_entities = set()
        named_entities.update(train_data["named_entities"])
        named_entities.update(dev_data["named_entities"])
        named_entities.update(test_data["named_entities"])
        #write_vocabulary(named_entities, "cnn_named_entities")

        #print("Named entities size is:", len(named_entities))

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "candidates": candidate_list,
                "answers": answer_list}

    def _return_named_entity(self, text, entity_dict):
        tokens = text.split()
        for i in range(len(tokens)):
            if tokens[i] in entity_dict:
                tokens[i] = entity_dict[tokens[i]]
        return " ".join(tokens)

    def _read_examples(self, folder, sub_set, debug_flag):
        """Creates examples for the training and dev sets."""
        question_data_dir = "{}/questions/{}".format(folder, sub_set)
        #  story_data_dir = "{}stories/".format(folder)

        examples = []
        question_list, passage_list, answer_list, instance_list, ansercandidate_list = [], [], [], [], []
        named_entities = set()

        files = os.listdir(question_data_dir)
        for file in files:
            story_input_file = os.path.join(question_data_dir, file)
            with open(story_input_file) as reader:
                all_lines = reader.readlines()
                new_lines = []
                for line in all_lines:
                    line = line.strip()
                    if len(line) > 0:
                        new_lines.append(line)
                passage = new_lines[1]
                question = new_lines[2]
                answer = new_lines[3]
                entities = new_lines[4:]
                entity_dict = {}
                for entity_line in entities:
                    entity_parts = entity_line.strip().split(":")
                    entity_dict[entity_parts[0]] = entity_parts[1]
                    named_entities.update([entity_parts[1]])

                passage = self._return_named_entity(passage, entity_dict)
                question = self._return_named_entity(question, entity_dict)
                answer = self._return_named_entity(answer, entity_dict)

                # print("Passage:", passage)
                # print("Question:", question)
                # print("Answer:", answer)
                # print("Named Entities:", named_entities)


                passage_list.append(passage)
                question_list.append(question)
                answer_list.append(answer)
                #break

                example = RCExample(
                    guid="",
                    passage=[passage],
                    question=[question],
                    answer=[answer],
                    instance=None
                )
                examples.append(example)
            if debug_flag:
                print(example)
                break


        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "candidates": ansercandidate_list,
                "answers": answer_list,
                "named_entities": named_entities}



class WhoDidWhatProcessor(DataProcessor):

    def get_all_examples(self, data_dir, debug_flag):

        #questions and answers
        train_data = self._read_examples(data_dir, "train", debug_flag)
        dev_data   = self._read_examples(data_dir,  "val", debug_flag)
        test_data  = self._read_examples(data_dir,  "test", debug_flag)

        examples = train_data["examples"] + dev_data["examples"] + test_data["examples"]

        question_list = train_data["questions"] + dev_data["questions"] + test_data["questions"]
        answer_list   = train_data["answers"]   + dev_data["answers"]   + test_data["answers"]
        passage_list  = train_data["passages"]  + dev_data["passages"]  + test_data["passages"]
        instance_list = set(train_data["instances"] + dev_data["instances"] + test_data["instances"])
        candidate_list = train_data["candidates"] + dev_data["candidates"] + test_data["candidates"]

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "candidates": candidate_list,
                "answers": answer_list}



    def _read_examples(self, folder, set, debug_flag):
        """Creates examples for the training and dev sets."""
        if set == "train":
            input_file = os.path.join(folder, "Relaxed", "{}_key.xml".format(set))
        else:
            input_file = os.path.join(folder, "Strict", "{}_key.xml".format(set))

        examples = []
        question_list, passage_list, answer_list, instance_list, ansercandidate_list = [], [], [], [], []

        with open(input_file) as reader:
            all_lines = reader.readlines()
            left_part = ""
            question, answer = "", ""
            i = 0
            local_answer_list = []
            for line in all_lines:
                if "leftContext" in line:
                    left_part = line.split(">")[1].split("<")[0]
                if "rightContext" in line:
                    right_part = line.split(">")[1].split("<")[0]
                    question = " @placeholder ".join([left_part, right_part])
                    question_list.append(question)
                if "choice" in line:
                    answer = line.split(">")[1].split("<")[0]
                    answer_list.append(answer)
                    local_answer_list.append(answer)

                if "</mc>" in line:
                    # NO LICENCE TO OBTAIN GIGIAWORD DATA = > no passage
                    example = RCExample(
                        guid="",
                        passage=[],
                        question=[question],
                        answer=local_answer_list,
                        instance=None
                    )
                    examples.append(example)
                    if debug_flag:
                        print(len(examples))
                        print(example)
                        break
                    local_answer_list = []

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "candidates": ansercandidate_list,
                "answers": answer_list}




# class WikiQAProcessor(DataProcessor):
#
#     def get_all_examples(self, data_dir, debug_flag):
#         examples = []
#         """Read a WikiQA json file into a list of SquadExample."""
#         question_list, passage_list, answer_list, instance_list = [], [], [], []
#
#         filename = data_dir + "WikiQA.tsv"
#         documents = WikiReaderIterable("doc", filename)
#         passage_list+=[" ".join(text) for text in list(iter(documents))]
#
#         questions = WikiReaderIterable("query", filename)
#         question_list += list(iter(questions))
#
#         return {"examples": examples,
#                 "questions": question_list,
#                 "passages": passage_list,
#                 "instances": [],
#                 "answers": answer_list}



class bAbIProcessor(DataProcessor):

    def get_all_examples(self, data_dir, debug_flag):
        examples = []
        """Read a SQuAD json file into a list of SquadExample."""
        question_list, passage_list, answer_list, instance_list = [], [], [], []

        files = os.listdir(data_dir)
        for file in files:
            story_input_file = os.path.join(data_dir, file)
            with open(story_input_file) as reader:
                all_lines = reader.readlines()
                paragraph_lines = []
                prev_number = 0
                for line in all_lines:
                    parts = line.split()
                    number = int(parts[0])
                    text = " ".join(parts[1:])
                    if number < prev_number:
                        passage = " ".join(paragraph_lines)
                        passage_list.append(passage)
                        paragraph_lines = []

                    if "?" in text:
                        q_parts = line.split("\t")
                        question = " ".join(q_parts[0].split()[1:])
                        question_list.append(question)
                        answer = q_parts[1]
                        answer_list.append(answer)
                        example = RCExample(
                                            guid="",
                                            passage=[" ".join(paragraph_lines)],
                                            question=[question],
                                            answer=[answer],
                                            instance=None
                                        )
                        examples.append(example)
                        if debug_flag:
                            print(example)
                    else:
                        paragraph_lines.append(text)
                    prev_number = number
                    #print(paragraph_lines)
                # print("Passage", passage_list)
                # print("Question", question_list)
                # print("Answer", answer_list)



                # last story
                passage = " ".join(paragraph_lines)
                passage_list.append(passage)


                # Return one example from every file
                if debug_flag:
                    break


        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": [],
                "answers": answer_list}




class CliCRProcessor(DataProcessor):
    def get_all_examples(self, data_dir, debug_flag):

        #questions and answers
        train_data = self._read_examples(data_dir, "train", debug_flag)
        dev_data   = self._read_examples(data_dir,  "dev", debug_flag)
        test_data  = self._read_examples(data_dir,  "test", debug_flag)

        examples = train_data["examples"] + dev_data["examples"] + test_data["examples"]

        question_list = train_data["questions"] + dev_data["questions"] + test_data["questions"]
        answer_list   = train_data["answers"]   + dev_data["answers"]   + test_data["answers"]
        passage_list  = train_data["passages"]  + dev_data["passages"]  + test_data["passages"]
        instance_list = set(train_data["instances"] + dev_data["instances"] + test_data["instances"])
        candidate_list = train_data["candidates"] + dev_data["candidates"] + test_data["candidates"]

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "candidates": candidate_list,
                "answers": answer_list}



    def _read_examples(self, folder, set, debug_flag):


        def clean_text(text):
            text = text.replace("\n", " ").replace("BEG__", "").replace("__END", "")
            return text


        """Creates examples for the training and dev sets."""
        input_file = "{}{}1.0.json".format(folder, set)

        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]

        examples = []
        question_list, passage_list, answer_list, instance_list, ansercandidate_list = [], [], [], [], []
        local_question_list, local_answer_list, = [], []
        for entry in input_data:
            entry = entry["document"]

            passage = clean_text(entry["context"])
            passage_list.append(passage)

            for qas in entry["qas"]:
                question = clean_text(qas["query"])
                local_question_list.append(question)
                question_list.append(question)
                for answer_entity in qas["answers"]:
                    answer = answer_entity["text"]
                    answer_list.append(answer)
                    local_answer_list.append(answer)

            example = RCExample(
                guid="",
                passage=[passage],
                question=local_question_list,
                answer=local_answer_list,
                instance=None
            )
            examples.append(example)
            if debug_flag:
                print(example)
                break

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "candidates": ansercandidate_list,
                "answers": answer_list}



class DuoRCProcessor(DataProcessor):
    """Processor for the DuoRC data set (GLUE version)."""
    def get_all_examples(self, data_dir, debug_flag):

        #questions and answers
        train_data = self._read_questions_examples(data_dir, "train", debug_flag)
        dev_data   = self._read_questions_examples(data_dir,  "dev", debug_flag)
        test_data   = self._read_questions_examples(data_dir,  "test", debug_flag)

        examples = train_data["examples"] + dev_data["examples"] + test_data["examples"]

        question_list = train_data["questions"] + dev_data["questions"] + test_data["questions"]
        answer_list   = train_data["answers"]   + dev_data["answers"]   + test_data["answers"]
        passage_list  = train_data["passages"]  + dev_data["passages"]  + test_data["passages"]
        instance_list = set(train_data["instances"] + dev_data["instances"] + test_data["instances"])

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}


    def _read_questions_examples(self, folder, set, debug_flag):
        """Creates examples for the training and dev sets."""
        input_file = "{}/ParaphraseRC_{}.json".format(folder, set)

        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)

        examples = []
        question_list, passage_list, answer_list, instance_list = [], [], [], []
        for entry in input_data:

            passage = entry["plot"]
            instance = entry["id"]
            passage_list.append(passage)
            instance_list.append(instance)
            local_question_list = []
            local_answer_list = []
            for question_entity in entry["qa"]:
                question = question_entity["question"]
                answer = question_entity["answers"]

                local_question_list.append(question)
                local_answer_list+=answer

                question_list.append(question)
                answer_list+=answer

            example = RCExample(
                guid=entry["id"],
                passage=[passage],
                question=local_question_list,
                answer=local_answer_list,
                instance=instance
            )
            examples.append(example)

            if debug_flag:
                print(example)
                break

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}

class DROPProcessor(DataProcessor):
        """Processor for the DROP data set (GLUE version)."""

        def get_all_examples(self, data_dir, debug_flag):

            # questions and answers
            train_data = self._read_questions_examples(data_dir, "train", debug_flag)
            dev_data = self._read_questions_examples(data_dir, "dev", debug_flag)

            examples = train_data["examples"] + dev_data["examples"]
            question_list = train_data["questions"] + dev_data["questions"]
            answer_list = train_data["answers"] + dev_data["answers"]
            passage_list = train_data["passages"] + dev_data["passages"]
            instance_list = set(train_data["instances"] + dev_data["instances"])

            return {"examples": examples,
                    "questions": question_list,
                    "passages": passage_list,
                    "instances": instance_list,
                    "answers": answer_list}

        def _read_questions_examples(self, folder, set, debug_flag):
            """Creates examples for the training and dev sets."""
            input_file = "{}/drop_dataset_{}.json".format(folder, set)

            with open(input_file, "r", encoding='utf-8') as reader:
                input_data = json.load(reader)

            examples = []
            question_list, passage_list, answer_list, instance_list = [], [], [], []
            for instance in input_data:
                entry = input_data[instance]
                #print(entry)

                passage = entry["passage"]
                passage_list.append(passage)
                instance_list.append(instance)
                local_q_list = []
                local_a_list = []

                for question_entity in entry["qa_pairs"]:
                    question = question_entity["question"]

                    answer_number = question_entity["answer"]["number"]
                    answer_date = question_entity["answer"]["date"]
                    answer_list = question_entity["answer"]["spans"]

                    if len(answer_date) > 0:
                        date = " ".join([answer_date["day"], answer_date["month"], answer_date["year"]])

                        if len(date) > 3:
                            #print(date)
                            answer_list.append(date)

                    if len(answer_number) > 0:
                        answer_list.append(answer_number)


                    answer =  " ".join(answer_list)

                    local_q_list.append(question)
                    local_a_list.append(answer)

                    question_list.append(question)
                    answer_list.append(answer)

                example = RCExample(
                    guid=instance,
                    passage=[passage],
                    question=local_q_list,
                    answer=local_a_list,
                    instance=instance
                )
                examples.append(example)

                if debug_flag:
                    print(example)
                    break

            return {"examples": examples,
                    "questions": question_list,
                    "passages": passage_list,
                    "instances": instance_list,
                    "answers": answer_list}




class AmazonQAProcessor(DataProcessor):
    """Processor for the AmazonQA data set (GLUE version)."""
    def get_all_examples(self, data_dir, debug_flag):

        #questions and answers
        train_data = self._read_questions_examples(data_dir, "train", debug_flag)
        dev_data   = self._read_questions_examples(data_dir,  "val", debug_flag)

        examples = train_data["examples"] + dev_data["examples"]

        question_list = train_data["questions"] + dev_data["questions"]
        answer_list   = train_data["answers"]   + dev_data["answers"]
        passage_list  = train_data["passages"]  + dev_data["passages"]
        instance_list = set(train_data["instances"] + dev_data["instances"])

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list,}


    def _read_questions_examples(self, folder, set, debug_flag):
        """Creates examples for the training and dev sets."""

        examples = []
        question_list, passage_list, answer_list, instance_list, candidates_list = [], [], [], [], []

        input_file = "{}{}-qar.jsonl".format(folder, set)
        with jsonlines.open(input_file, "r") as reader:
            #input_data = json.load(reader)

            for entry in reader.iter():
            #for entry in input_data:
                asin = entry["asin"]
                qas_id = entry["qid"]
                question = entry["questionText"]
                passage = " ".join(entry["review_snippets"])
                answer = []
                for answer_entity in entry["answers"]:
                    answer.append(answer_entity["answerText"])


                answer_list+=(answer)
                question_list.append(question)
                passage_list.append(passage)
                instance_list.append(asin)

                example = RCExample(
                    guid="_".join([set, str(qas_id)]),
                    passage=[passage],
                    question=[question],
                    answer=answer,
                    instance=asin,
                )
                examples.append(example)
                if debug_flag:
                    print(example)
                    break


        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list,
                "candidates": candidates_list}

class QuasarProcessor(DataProcessor):


    def get_all_examples(self, data_dir, debug_flag):
        train_data = self._read_questions_examples(data_dir, "train", debug_flag)
        dev_data   = self._read_questions_examples(data_dir,  "dev", debug_flag)
        test_data  = self._read_questions_examples(data_dir,  "test", debug_flag)

        examples = train_data["examples"] + dev_data["examples"] + test_data["examples"]

        question_list = train_data["questions"] + dev_data["questions"] + test_data["questions"]
        answer_list   = train_data["answers"]   + dev_data["answers"]   + test_data["answers"]
        passage_list  = train_data["passages"]  + dev_data["passages"]  + test_data["passages"]
        instance_list = set(train_data["instances"] + dev_data["instances"] + test_data["instances"])

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}


    def _read_questions_examples(self, folder, set, debug_flag):

        """Creates examples for the training and dev sets."""
        examples = []
        question_list, passage_list, answer_list, instance_list = [], [], [], []


        context_input_file = "{}contexts/short/{}_contexts.json.gz".format(folder, set)
        #context_input_file = "{}contexts/long/{}_contexts.json.gz".format(folder, set)
        passage_dict = {}
        with gzip.open(context_input_file) as f:
            for entry in f:
                entry = eval(entry)
                context_list = []
                for sentence in entry["contexts"]:
                    context_list.append(sentence[1])
                passage = " ".join(context_list)
                passage_dict[entry["uid"]] = passage
                passage_list.append(passage)


        questions_input_file = "{}questions/{}_questions.json.gz".format(folder, set)
        with gzip.open(questions_input_file) as f:
            for line in f:
                entry = eval(line)
                q_id = entry["uid"]
                question = entry["question"]
                answer = entry["answer"]

                answer_list.append(answer)
                question_list.append(question)

                example = RCExample(
                    guid=q_id,
                    passage=[passage_dict[q_id]],
                    question=[question],
                    answer=[answer],
                    instance=None
                )
                examples.append(example)
                if debug_flag:
                    #print(example)
                    break

                #print(example)

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}


class RecipeQAProcessor(DataProcessor):
    def get_all_examples(self, data_dir, debug_flag):

        #questions and answers
        train_data = self._read_questions_examples(data_dir, "train", debug_flag)
        dev_data   = self._read_questions_examples(data_dir,  "val", debug_flag)
        test_data  = self._read_questions_examples(data_dir,  "test", debug_flag)

        examples = train_data["examples"] + dev_data["examples"] + test_data["examples"]

        question_list = train_data["questions"]     + dev_data["questions"]  + test_data["questions"]
        answer_list   = train_data["answers"]       + dev_data["answers"]    + test_data["answers"]
        passage_list  = train_data["passages"]      + dev_data["passages"]   + test_data["passages"]
        instance_list = set(train_data["instances"] + dev_data["instances"]  + test_data["instances"])
        candidate_list = train_data["candidates"]   + dev_data["candidates"] + test_data["candidates"]

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "candidates": candidate_list,
                "answers": answer_list}


    def _read_questions_examples(self, folder, set, debug_flag):
        """Creates examples for the training and dev sets."""
        input_file = "{}{}.json".format(folder, set)

        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]

        examples = []
        question_list, passage_list, answer_list, instance_list, ansercandidate_list = [], [], [], [], []
        for entry in input_data:
            if entry["task"] != "textual_cloze":
                continue
            qas_id = entry["recipe_id"]
            question = " ".join(entry["question"])

            context_list = []
            for context in entry["context"]:

                context_list.append(context["body"])
            passage = " ".join(context_list)

            ansercandidate = entry["choice_list"]
            answer = ansercandidate[entry["answer"]]

            answer_list.append(answer)
            question_list.append(question)
            passage_list.append(passage)
            ansercandidate_list+=ansercandidate

            example = RCExample(
                guid=qas_id,
                passage=[passage],
                question=[question],
                answer=[answer],
                ansercandidate=ansercandidate,
                instance=None
            )
            examples.append(example)
            if debug_flag:
                print(example)
                break

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "candidates": ansercandidate_list,
                "answers": answer_list}


class QAngarooProcessor(DataProcessor):
    """Processor for the QAngaroo data set (GLUE version)."""
    def get_all_examples(self, data_dir, debug_flag):

        #questions and answers
        train_data = self._read_questions_examples(data_dir, "train", debug_flag)
        dev_data   = self._read_questions_examples(data_dir,  "dev", debug_flag)

        examples = train_data["examples"] + dev_data["examples"]

        question_list = train_data["questions"] + dev_data["questions"]
        answer_list   = train_data["answers"]   + dev_data["answers"]
        passage_list  = train_data["passages"]  + dev_data["passages"]
        instance_list = set(train_data["instances"] + dev_data["instances"])
        candidate_list = train_data["candidates"]  + dev_data["candidates"]

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list,
                "candidates": candidate_list}


    def _read_questions_examples(self, folder, set, debug_flag):
        """Creates examples for the training and dev sets."""

        input_file = "{}/{}.json".format(folder, set)
        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)

        examples = []
        question_list, passage_list, answer_list, instance_list, candidates_list = [], [], [], [], []
        for entry in input_data:
            qas_id = entry["id"]
            question = entry["query"]
            candidates = entry["candidates"]
            candidates_list += candidates
            context = entry["supports"]

            passage = " ".join(context)
            answer = entry["answer"]


            answer_list.append(answer)
            question_list.append(question)
            passage_list.append(passage)

            example = RCExample(
                guid=qas_id,
                passage=[passage],
                question=[question],
                answer=[answer],
                instance=None,
            )
            examples.append(example)
            if debug_flag:
                print(example)
                break


        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list,
                "candidates": candidates_list}


class HotpotQAProcessor(DataProcessor):
    """Processor for the HotpotQA data set (GLUE version)."""
    def get_all_examples(self, data_dir, debug_flag):

        #questions and answers
        train_data = self._read_questions_examples(data_dir, "train", debug_flag)
        dev_data   = self._read_questions_examples(data_dir,  "dev", debug_flag)
        test_data   = self._read_questions_examples(data_dir,  "test", debug_flag)

        examples = train_data["examples"] + dev_data["examples"] + test_data["examples"]

        question_list = train_data["questions"] + dev_data["questions"] + test_data["questions"]
        answer_list   = train_data["answers"]   + dev_data["answers"]   + test_data["answers"]
        passage_list  = train_data["passages"]  + dev_data["passages"]  + test_data["passages"]
        instance_list = set(train_data["instances"] + dev_data["instances"] + test_data["instances"])

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list,
                "candidates": []}


    def _read_questions_examples(self, folder, set, debug_flag):
        """Creates examples for the training and dev sets."""
        input_file = "{}/hotpot_{}_fullwiki_v1.json".format(folder, set)
        if set == "train":
            input_file = "{}/hotpot_{}_v1.1.json".format(folder, set)

        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)

        examples = []
        question_list, passage_list, answer_list, instance_list, ansercandidate_list = [], [], [], [], []
        for entry in input_data:
            qas_id = entry["_id"]
            question = entry["question"]

            facts_list = []
            for context in entry["context"]:
                instance = context[0]
                facts_list+=context[1]
                instance_list.append(instance)

            passage = " ".join(facts_list)
            #ansercandidate = entry["candidates"]
            answer = None
            if set is not "test":
                answer = entry["answer"]
                answer_list.append(answer)
            question_list.append(question)
            passage_list.append(passage)
            #ansercandidate_list.append(ansercandidate)

            example = RCExample(
                guid=qas_id,
                passage=[passage],
                question=[question],
                answer=[answer],
                #ansercandidate=ansercandidate,
                instance=None
            )
            examples.append(example)

            if debug_flag:
                print(example)
                break

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                #"ansercandidate": ansercandidate_list,
                "answers": answer_list}


class TurkQAProcessor(DataProcessor):
    """Processor for the TurkQA data set (GLUE version)."""

    def get_all_examples(self, data_dir, debug_flag):
        examples = []
        question_list, passage_list, answer_list, instance_list = [], [], [], []

        for i in range(0, 30000):
            story_input_file = os.path.join(data_dir, "problems", str(i) + ".txt")
            if not os.path.exists(story_input_file):
                continue
            with open(story_input_file) as reader:
                paragrath_lines = reader.readlines()
                passage = " ".join(paragrath_lines)
                passage_list.append(passage)

            qa_input_file = os.path.join(data_dir, "results", str(i) + ".txt")
            if not os.path.exists(qa_input_file):
                continue
            with open(qa_input_file) as reader:
                lines = reader.readlines()
                j = 0
                while j < len(lines):
                    question = lines[j]
                    j += 1
                    answer = lines[j]
                    j += 1
                    indexes = answer.split()
                    if len(indexes) == 2:
                        answer = passage[int(indexes[0]):int(indexes[1])]

                    examples.append(RCExample(
                        guid="",
                        passage=[passage],
                        question=[question],
                        answer=[answer],
                        instance=None
                    ))

                    answer_list.append(answer)
                    question_list.append(question)

                    if debug_flag:
                        print(examples)
                        break
            if debug_flag:
                break

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": [],
                "answers": answer_list}




class ShARCProcessor(DataProcessor):
    """Processor for the ShaRC data set (GLUE version)."""
    def get_all_examples(self, data_dir, debug_flag):

        #questions and answers
        train_data = self._read_questions_examples(data_dir, "train", debug_flag)
        dev_data   = self._read_questions_examples(data_dir,  "dev", debug_flag)

        examples = train_data["examples"] + dev_data["examples"]

        question_list = train_data["questions"] + dev_data["questions"]
        answer_list = train_data["answers"] + dev_data["answers"]
        instance_list = set(train_data["instances"] + dev_data["instances"])
        passage_list = train_data["passages"] + dev_data["passages"]

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}


    def _read_questions_examples(self, folder, set, debug_flag):
        """Creates examples for the training and dev sets."""
        input_file = folder + "sharc_" + set + ".json"
        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)

        examples = []
        question_list, passage_list, answer_list, instance_list = [], [], [], []
        for entry in input_data:
            instance = entry["tree_id"]
            instance_list.append(instance)
            question = entry["question"]
            answer = entry["answer"]
            qas_id = entry["utterance_id"]
            follow_up_questions = []
            for history in entry["history"]:
                follow_up_questions.append(history["follow_up_question"])
                follow_up_questions.append(history["follow_up_answer"])


            passage = " ".join([entry["snippet"], entry["scenario"]] + follow_up_questions)

            question_list.append(question)
            answer_list.append(answer)
            passage_list.append(passage)

            example = RCExample(
                guid=qas_id,
                passage=[passage],
                question=[question],
                answer=[answer],
                instance=instance
            )
            examples.append(example)

            if debug_flag:
                print(examples)
                break

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}


class NarrativeQAProcessor(DataProcessor):
    """Processor for the NarrativeQA data set (GLUE version)."""

    def get_all_examples(self, data_dir, debug_flag):
        examples = []
        question_list, answer_list, passage_list = [], [], []

        pre_example_passage_dict = {}
        pre_example_question_dict = {}
        pre_example_answer_dict = {}
        with open(data_dir + 'third_party/wikipedia/summaries.csv', newline='') as csvfile:
            narrative_data = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(narrative_data)
            for row in narrative_data:
                id = row[0]
                passage = row[2]
                pre_example_passage_dict[id] = passage
                passage_list.append(passage)
                if debug_flag:
                    break

        with open(data_dir + 'qaps.csv', newline='') as csvfile:
            narrative_data = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(narrative_data)
            i = 0
            for row in narrative_data:
                doc_id = row[0]
                if doc_id not in pre_example_question_dict:
                    pre_example_question_dict[doc_id] = []
                    pre_example_answer_dict[doc_id] = []



                question = row[2]
                pre_example_question_dict[doc_id].append(question)
                answer_1 = row[3]
                answer_2 = row[4]
                pre_example_answer_dict[doc_id].append(answer_1)
                pre_example_answer_dict[doc_id].append(answer_2)

                question_list.append(question)

                answer_list.append(answer_1)
                answer_list.append(answer_2)


            i += 1
        for doc_id in pre_example_passage_dict:
            example = RCExample(
                guid=doc_id,
                passage=[pre_example_passage_dict[doc_id]],
                question=pre_example_question_dict[doc_id],
                answer=pre_example_answer_dict[doc_id],
                instance=None
            )

            examples.append(example)
            if debug_flag:
                print(example)
                break


        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": [],
                "answers": answer_list}


class WikiMovieProcessor(DataProcessor):
    """Processor for the WikiMovie data set (GLUE version)."""

    def get_all_examples(self, data_dir, debug_flag):

        # Data (passages)
        knowledge_data = self._read_passages(data_dir)

        #questions and answers
        train_data = self._read_questions_examples(data_dir, "train", debug_flag)
        dev_data   = self._read_questions_examples(data_dir,  "dev", debug_flag)
        test_data  = self._read_questions_examples(data_dir, "test", debug_flag)

        examples = train_data["examples"] + dev_data["examples"] + test_data["examples"]

        question_list = train_data["questions"] + dev_data["questions"] + test_data["questions"]
        answer_list = train_data["answers"] + dev_data["answers"] + test_data["answers"]

        passage_list = knowledge_data["passages"]

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": [],
                "answers": answer_list}


    def _read_passages(self, folder):

        """Read a SQuAD json file into a list of SquadExample."""
        input_file = folder + "knowledge_source/full/full_kb.txt"
        passage_list = []
        with open(input_file) as reader:
            data = reader.readlines()
            for i in range(len(data)):
                line = data[i]
                passage_lines = []
                while len(line.strip()) > 0:
                    parts = line.strip().split()
                    passage_lines.append(" ".join(parts[1:]))
                    i+=1
                    line = data[i]
                passage_list.append(". ".join(passage_lines))
                i+=1

        return {"passages": passage_list}


    def _read_questions_examples(self, folder, set, debug_flag):
        """Creates examples for the training and dev sets."""
        examples = []
        """Read a SQuAD json file into a list of SquadExample."""
        input_file = folder + "questions/full/full_qa_" + set + ".txt"
        question_list, passage_list, answer_list, instance_list = [], [], [], []
        with open(input_file) as reader:
            data = reader.readlines()
            for line in data:
                chech_the_first = line[0]
                if int(chech_the_first) != 1:
                    print("Irregular line:", line)
                parts = line[1:].strip().split("\t")

                question = parts[0]
                passages = ""
                answer = parts[1]

                question_list.append(question)
                passage_list += passages

                answer_list.append(answer)

                example = RCExample(
                    guid= "",
                    passage=" ".join(passages),
                    question=question,
                    answer=answer,
                    instance=None
                )
                examples.append(example)
                if debug_flag:
                    print(example)
                    break


        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}


class SQuAD2Processor(DataProcessor):
    """Processor for the SQuAD2 data set (GLUE version)."""

    def get_examples(self, data_dir, subset, debug_flag):
        """See base class."""
        filename = os.path.join(data_dir, subset + "-v2.0.json")
        return self._read_examples(filename, subset, debug_flag)

    def get_all_examples(self, data_dir, debug_flag):
        train_data = self.get_examples(data_dir, "train", debug_flag)
        dev_data = self.get_examples(data_dir, "dev", debug_flag)

        examples = train_data["examples"] + dev_data["examples"]

        question_list = train_data["questions"] + dev_data["questions"]
        passage_list = train_data["passages"] + dev_data["passages"]
        instance_list = set(train_data["instances"] + dev_data["instances"])
        answer_list = train_data["answers"] + dev_data["answers"]

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}



    def _read_examples(self, input_file, str_id, debug_flag):

        """Read a SQuAD json file into a list of SquadExample."""
        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]

        examples = []
        question_list, passage_list, answer_list, instance_list = [], [], [], []
        for entry in input_data:
            instance = entry["title"]
            instance_list.append(instance)
            passage_count = 0
            for paragraph in entry["paragraphs"]:
                passage = paragraph["context"]
                passage_list.append(passage)

                local_question_list = []
                local_answer_list = []
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question = qa["question"]
                    question = question.replace("??", "?")
                    local_question_list.append(question)

                    is_impossible = qa["is_impossible"]
                    if not is_impossible:
                        answer = qa["answers"][0]["text"]
                        answer_list.append(answer)
                        local_answer_list.append(answer)
                    else:
                        answer = None
                        local_answer_list.append("")


                    question_list.append(question)

                passage_count += 1
                example = RCExample(
                    guid="{}_{}".format(instance, passage_count),
                    passage=[passage],
                    question=local_question_list,
                    answer=local_answer_list,
                    instance=instance
                )
                examples.append(example)
                if debug_flag:
                    print(example)
                    break

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}


class SQuADProcessor(DataProcessor):
    """Processor for the SQuAD data set (GLUE version)."""

    def get_examples(self, data_dir, subset, debug_flag):
        """See base class."""
        filename = os.path.join(data_dir, subset + "-v1.1.json")
        return self._read_examples(filename, subset, debug_flag)

    def get_all_examples(self, data_dir, debug_flag):
        train_data = self.get_examples(data_dir, "train", debug_flag)
        dev_data = self.get_examples(data_dir, "dev", debug_flag)

        examples = train_data["examples"] + dev_data["examples"]

        question_list = train_data["questions"] + dev_data["questions"]
        passage_list = train_data["passages"] + dev_data["passages"]
        instance_list = set(train_data["instances"] + dev_data["instances"])
        answer_list = train_data["answers"] + dev_data["answers"]

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}



    def _read_examples(self, input_file, str_id, debug_flag):

        """Read a SQuAD json file into a list of SquadExample."""
        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]

        examples = []
        question_list, passage_list, answer_list, instance_list = [], [], [], []
        for entry in input_data:
            instance = entry["title"]
            instance_list.append(instance)
            for paragraph in entry["paragraphs"]:
                passage = paragraph["context"]
                passage_list.append(passage)

                local_question_list = []
                local_answer_list = []
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question = qa["question"]
                    question = question.replace("??", "?")
                    local_question_list.append(question)

                    answer = qa["answers"][0]["text"]
                    local_answer_list.append(answer)

                    question_list.append(question)
                    answer_list.append(answer)

                example = RCExample(
                    guid=qas_id,
                    passage=[passage],
                    question=local_question_list,
                    answer=local_answer_list,
                    instance=instance
                )
                examples.append(example)

                if debug_flag:
                    print(example)
                    break

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}


class PubMedProcessor(DataProcessor):
    """Processor for the PubMed data set (GLUE version)."""

    def get_all_examples(self, data_dir, debug_flag):
        filename = os.path.join(data_dir, "ori_pqal.json")
        return self._read_examples(filename, "", debug_flag)


    def get_labels(self):
        """See base class."""
        return [False, True, "Maybee"]


    def _read_examples(self, input_file, str_id, debug_flag):
        """Creates examples for the training and dev sets."""
        examples = []
        """Read a SQuAD json file into a list of SquadExample."""
        with open(input_file) as json_file:
            data = json.load(json_file)
            question_list, passage_list, answer_list, instance_list = [], [], [], []
            for i, data_entry in enumerate(data):
                json_entry = data[data_entry]
                #print(data_entry)
                #print(json_entry)

                question = json_entry["QUESTION"]
                passages = json_entry["CONTEXTS"]
                answer = json_entry["final_decision"]

                question_list.append(question)
                passage_list += passages

                answer_list.append(json_entry["final_decision"])

                example = RCExample(
                    guid=data_entry,
                    passage=passages,
                    question=[question],
                    answer=[answer],
                    instance=None
                )
                examples.append(example)

                if debug_flag:
                    print(example)
                    break


        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}


class BoolQProcessor(DataProcessor):
    """Processor for the BoolQ data set (GLUE version)."""

    def get_examples(self, data_dir, subset, debug_flag):
        """See base class."""
        filename = os.path.join(data_dir, subset + ".jsonl")
        return self._read_examples(filename, subset, debug_flag)

    def get_all_examples(self, data_dir, debug_flag):
        train_data = self.get_examples(data_dir, "train", debug_flag)
        dev_data = self.get_examples(data_dir,  "dev", debug_flag)
        test_data = self.get_examples(data_dir, "test", debug_flag)

        examples = train_data["examples"] + dev_data["examples"] + test_data["examples"]

        question_list = train_data["questions"] + dev_data["questions"] + test_data["questions"]
        passage_list = train_data["passages"] + dev_data["passages"] + test_data["passages"]
        instance_list = set(train_data["instances"] + dev_data["instances"] + test_data["instances"])
        #answer_list = train_data["answers"] + dev_data["answers"] + test_data["answers"]
        answer_list = []

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}

    def get_labels(self):
        """See base class."""
        return [False, True]


    def _read_examples(self, input_file, str_id, debug_flag):
        """Creates examples for the training and dev sets."""
        examples = []
        """Read a SQuAD json file into a list of SquadExample."""
        with open(input_file, "r", encoding="utf-8") as reader:
            question_list, passage_list, answer_list, instance_list = [], [], [], []
            for i, data in enumerate(reader):
                json_entry = json.loads(data)
                question_list.append(json_entry["question"])
                passage_list.append(json_entry["passage"])
                instance_list.append(json_entry["title"])

                answer = ""
                if str_id != "test":
                    answer = json_entry["answer"]
                    answer_list.append(json_entry["answer"])

                example = RCExample(
                    guid="_".join([str_id, str(i)]),
                    passage=[json_entry["passage"]],
                    question=[json_entry["question"]],
                    answer=[answer],
                    instance=json_entry["title"]
                )
                examples.append(example)

                if debug_flag:
                    break


        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}




class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""



    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")

    def get_labels(self):
        """See base class."""
        # return ["contradiction", "entailment", "neutral"]
        return ["contradiction", "entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            if label == "neutral":
                label = "contradiction"
            examples.append(RCExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples






processors = {
    "boolq": BoolQProcessor,
    "mnli": MnliProcessor,
    "pubmedqa": PubMedProcessor,
    "squad": SQuADProcessor,
    "squad2.0": SQuAD2Processor,
    "wikimovies": WikiMovieProcessor,
    "narrativeqa": NarrativeQAProcessor,
    "sharc": ShARCProcessor,
    "turkqa": TurkQAProcessor,
    "hotpotqa": HotpotQAProcessor,
    "medhop": QAngarooProcessor,
    "wikihop": QAngarooProcessor,
    "recipeqa": RecipeQAProcessor,
    "quasars": QuasarProcessor,
    "quasart": QuasarProcessor,
    "amazonqa": AmazonQAProcessor,
   # "naturalquestions": NaturalQuestionsProcessor,
    "duorc": DuoRCProcessor,
    "drop": DROPProcessor,
    "clicr": CliCRProcessor,
    "babi": bAbIProcessor,
    #"wikiqa": WikiQAProcessor,
    "wdw": WhoDidWhatProcessor,
    "cnn": CNNDailyMailProcessor,
    "dailymail": CNNDailyMailProcessor,
    "msmarco": MsMarcoProcessor,
    "newsqa": NewsQAProcessor,
    "searchqa": SearchQAProcessor,
    #"wikireading": WikiReadingProcessor,
    "tydi": TyDiProcessor,
    "emrqa": EmrQAProcessor,

    #from vocab
    "quac": QuACProcessor,
    "multirc": MultiRCProcessor,
    "coqa": CoQAProcessor,
    "cosmosqa": CosmosQAProcessor,
    "movieqa": MovieQAProcessor,
    "mcscript": MCScriptProcessor,
    "mcscript2.0": MCScriptProcessor,
    "amazonyesno": AmazonYesNoProcessor,
    "mctest160": MCTestProcessor,
    "mctest500": MCTestProcessor,
    #"lambada": LambadaProcessor,

    "dream": DREAMProcessor,
    "reclor": ReClorProcessor,
    "cbt": CBTestProcessor,
    "race": RACEProcessor,
    "racec": RACEProcessor,
    "triviaqa": TriviaQAProcessor,
    "sciq": SciQProcessor,
    "record": ReCoRDProcessor,
    "tweetqa": TweetQAProcessor,
    "quail": QuAILProcessor,

    # NEWEST for CAMeRA READY
    "arc": ARCProcessor,
    "bipar": BiPaRProcessor,
    "openbookqa": OpenBookQAProcessor,
    "qasc": QASCProcessor,
    "subjqa": SubjQAProcessor,
    "iirc":IIRCProcessor,

}

