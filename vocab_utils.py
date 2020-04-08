import json
import os
import csv
import gzip
import jsonlines
from xml.dom import minidom

from utils import VocabularyProcessor
from spacy.lang.en import English

nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)

class LambadaProcessor(VocabularyProcessor):

    def get_vocabulary_size(self, data_dir):
        dev_vocabulary, dev_question_len_list, dev_passage_len_list = self._get_vocabulary(data_dir, "development")
       # print("Dev done")
        test_vocabulary, test_question_len_list, test_passage_len_list = self._get_vocabulary(data_dir, "test")
       # print("Test done")

        question_len_list = dev_question_len_list + test_question_len_list
        avg_q_len = sum(question_len_list) / len(question_len_list)
       # print("Average question length:", avg_q_len)

        passage_len_list = dev_passage_len_list + test_passage_len_list
        avg_q_len = sum(passage_len_list) / len(passage_len_list)
       # print("Average passage length:", avg_q_len)



        train_vocabulary = self._get_train_vocabulary(os.path.join(data_dir, "train-novels"))
       # print("Train done")

        vocabulary = train_vocabulary
        vocabulary.update(dev_vocabulary)
        vocabulary.update(test_vocabulary)
        return vocabulary

    def _get_train_vocabulary(self, data_dir):
        vocabulary = set()

        folders = os.listdir(data_dir)
        for folder in folders:
           # print(folder)
            files = os.listdir(os.path.join(data_dir, folder))
            for file in files:
                story_input_file = os.path.join(data_dir, folder, file)
                with open(story_input_file) as reader:
                    # print(story_input_file)
                    story_lines = reader.readlines()
                    for text in story_lines:
                        text = text.strip()
                        tokens = tokenizer(text)
                        vocabulary.update({x.lemma_.lower() for x in tokens})


        return vocabulary

    def _get_vocabulary(self, data_dir, subset):
        vocabulary = set()
        question_len_list = []
        passage_len_list = []

        input_file = os.path.join(data_dir, "lambada_{}_plain_text.txt".format(subset))
        with open(input_file) as reader:
          #  print(input_file)
            story_lines = reader.readlines()
            for text in story_lines:
                text = text.strip()
                #print(text)

                whole_passage = text.split()
                question = text.split(".")[-1]
                question_len = len(question.split())
                question_len_list.append(question_len)
                passage_len_list.append(len(whole_passage) - question_len)

                tokens = tokenizer(text)
                vocabulary.update({x.lemma_.lower() for x in tokens})



        return vocabulary, question_len_list, passage_len_list



class MCTestProcessor(VocabularyProcessor):

    def get_vocabulary_size(self, data_dir):
        train_vocabulary = self._get_vocabulary(data_dir, "train")
        dev_vocabulary = self._get_vocabulary(data_dir, "dev")
        test_vocabulary = self._get_vocabulary(data_dir, "test")

        vocabulary = train_vocabulary
        vocabulary.update(dev_vocabulary)
        vocabulary.update(test_vocabulary)
        vocab_size = len(vocabulary)
        return vocabulary

    def _get_vocabulary(self, data_dir, subset):
        vocabulary = set()

        files = os.listdir(data_dir)
        for file in files:
            if "{}.tsv".format(subset) not in file:
                continue
            with open(os.path.join(data_dir, file)) as fd:
               # print(data_dir + file)
                rd = csv.reader(fd, delimiter="\t", quotechar='"')
                for row in rd:
                    for text in row[2:]:
                        text = text.strip()
                        tokens = tokenizer(text)
                        vocabulary.update({x.lemma_.lower() for x in tokens})

            return vocabulary


class AmazonYesNoProcessor(VocabularyProcessor):

    def get_vocabulary_size(self, data_dir):
        vocabulary = set()

        domain_folders = os.listdir(data_dir)
        for folder in domain_folders:
            if "copy" in folder:
                continue
            files = os.listdir(data_dir + folder)
            for file in files:
                if "yes_no.balanced" not in file:
                    continue
                file_name = data_dir + folder + "/" + file
                with open(file_name, "r") as reader:
                    #print(file_name)
                    input_data = json.load(reader)

                for entry_element in input_data:
                    entry = input_data.get(entry_element)
                    text_list = []
                    text_list.append(entry["review"])

                    for qestion in entry["qa"]:
                        text_list.append(qestion["q"])
                        text_list.append(qestion["a"])

                    for text in text_list:
                        tokens = tokenizer(text)
                        vocabulary.update({x.lemma_.lower() for x in tokens})
                    #print(text_list)
                    #break
            #break

        vocab_size = len(vocabulary)
        return vocabulary


class MCScriptProcessor(VocabularyProcessor):

    def get_vocabulary_size(self, data_dir):
        train_vocabulary = self._get_vocabulary(data_dir, "train")
        dev_vocabulary = self._get_vocabulary(data_dir, "dev")
        test_vocabulary = self._get_vocabulary(data_dir, "test")

        vocabulary = train_vocabulary
        vocabulary.update(dev_vocabulary)
        vocabulary.update(test_vocabulary)
        vocab_size = len(vocabulary)
        return vocabulary


    def _get_vocabulary(self, folder, sub_set):
        input_file = "{}{}-data.xml".format(folder, sub_set)
        vocabulary = set()
        text_list = []
        xmldoc = minidom.parse(input_file)

        instancelist = xmldoc.getElementsByTagName('instance')
        for instance in instancelist:
            text_list.append(instance.getElementsByTagName('text')[0].firstChild.nodeValue)

            question_list = instance.getElementsByTagName('question')
            for question in question_list:
                text_list.append(question.attributes['text'].value)
                for answer in question.getElementsByTagName('answer'):
                    text_list.append(answer.attributes['text'].value)

            for text in text_list:
                tokens = tokenizer(text)
                vocabulary.update({x.lemma_.lower() for x in tokens})
            #break

        return vocabulary


class MovieQAProcessor(VocabularyProcessor):

    def get_vocabulary_size(self, data_dir):
        qa_vocabulary = self._get_qa_vocabulary(data_dir, "train")
        plot_vocabulary = self._get_plot_vocabulary(data_dir, "test")

        vocabulary = plot_vocabulary
        vocabulary.update(qa_vocabulary)
        vocab_size = len(vocabulary)
        return vocabulary


    def _get_qa_vocabulary(self, folder, sub_set):
        input_file = "{}data/qa.json".format(folder, sub_set)
        vocabulary = set()
        answer_len_list = []

        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)

        for entry in input_data:
            tokens = tokenizer(entry["question"])
            vocabulary.update({x.lemma_.lower() for x in tokens})

            for answer in entry["answers"]:
                tokens = tokenizer(answer)
                answer_len_list.append(len(tokens))
                vocabulary.update({x.lemma_.lower() for x in tokens})
           # break

        avg_answer_len = sum(answer_len_list) / len(answer_len_list)
        #print("AVG answer len:", avg_answer_len)
        return vocabulary


    def _get_plot_vocabulary(self, folder, sub_set):
        vocabulary = set()

        plot_files = os.listdir(folder + "story/plot/")
        for file in plot_files:
            with open(folder + "story/plot/" + file, "r") as reader:
                file = reader.readlines()
                for text in file:
                    text = text.strip()
                    tokens = tokenizer(text)
                    vocabulary.update({x.lemma_.lower() for x in tokens})
#            break

        return vocabulary


class CosmosQAProcessor(VocabularyProcessor):

    def get_vocabulary_size(self, data_dir):
        train_vocabulary = self._get_vocabulary(data_dir, "train")
        dev_vocabulary = self._get_vocabulary(data_dir, "valid")
        test_vocabulary = self._get_test_vocabulary(data_dir, "test")

        vocabulary = train_vocabulary
        vocabulary.update(dev_vocabulary)
        vocabulary.update(test_vocabulary)
        vocab_size = len(vocabulary)
        return vocabulary


    def _get_vocabulary(self, folder, sub_set):
        vocabulary = set()
        input_file = "{}{}.csv".format(folder, sub_set)
        with open(input_file, newline='') as csvfile:
            data = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(data)
            for entry in data:
                for text in entry[1:-1]:
                    tokens = tokenizer(text)
                    vocabulary.update({x.lemma_.lower() for x in tokens})

            return vocabulary


    def _get_test_vocabulary(self, folder, sub_set):
        vocabulary = set()

        input_file = "{}/{}.jsonl".format(folder, sub_set)
        with jsonlines.open(input_file, "r") as reader:
            for entry in reader.iter():
                text_list = []
                text_list.append(entry["context"])
                text_list.append(entry["question"])
                text_list.append(entry["answer0"])
                text_list.append(entry["answer1"])
                text_list.append(entry["answer2"])
                text_list.append(entry["answer3"])

                for text in text_list:
                    tokens = tokenizer(text)
                    vocabulary.update({x.lemma_.lower() for x in tokens})

        return vocabulary


class CoQAProcessor(VocabularyProcessor):

    def get_vocabulary_size(self, data_dir):
        train_vocabulary = self._get_vocabulary(data_dir, "train")
        dev_vocabulary = self._get_vocabulary(data_dir, "dev")

        vocabulary = train_vocabulary
        vocabulary.update(dev_vocabulary)
        vocab_size = len(vocabulary)
        return vocabulary

    def _get_vocabulary(self, folder, sub_set):
        input_file =  "{}coqa-{}-v1.0.json".format(folder, sub_set)
        vocabulary = set()

        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]

        for entry in input_data:
            text_list = []
            text_list.append(entry["story"])

            for qestion in entry["questions"]:
                text_list.append(qestion["input_text"])
            for answer in entry["answers"]:
                text_list.append(answer["input_text"])

            for text in text_list:
                tokens = tokenizer(text)
                vocabulary.update({x.lemma_.lower() for x in tokens})
            # print(text_list)
            # break

        return vocabulary


class QuACProcessor(VocabularyProcessor):

    def get_vocabulary_size(self, data_dir):
        train_vocabulary = self._get_vocabulary(data_dir, "train")
        dev_vocabulary = self._get_vocabulary(data_dir, "val")

        vocabulary = train_vocabulary
        vocabulary.update(dev_vocabulary)
        vocab_size = len(vocabulary)
        return vocabulary

    def _get_vocabulary(self, folder, sub_set):
        input_file =  "{}/{}_v0.2.json".format(folder, sub_set)
        vocabulary = set()

        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]

        for entry in input_data:
            text_list = [entry["title"], entry["background"], entry["section_title"]]
            for paragraph in entry["paragraphs"]:
                passage = paragraph["context"]
                text_list.append(passage)

                for qas in paragraph["qas"]:
                    question = qas["question"]
                    text_list.append(question)
                    answers = [x["text"] for x in (qas["answers"]+[qas["orig_answer"]])]
                    text_list+=answers
#                    text_list.append(qas["orig_answer"]["text"])

            for text in text_list:
                tokens = tokenizer(text)
                vocabulary.update({x.lemma_.lower() for x in tokens})
            #break

        return vocabulary


class MultiRCProcessor(VocabularyProcessor):

    def get_vocabulary_size(self, data_dir):
        train_vocabulary = self._get_vocabulary(data_dir, "train")
        dev_vocabulary = self._get_vocabulary(data_dir, "val")
        test_vocabulary = self._get_vocabulary(data_dir, "test")

        vocabulary = train_vocabulary
        vocabulary.update(dev_vocabulary)
        vocabulary.update(test_vocabulary)
        vocab_size = len(vocabulary)
        return vocabulary

    def _get_vocabulary(self, folder, sub_set):
        vocabulary = set()

        #input_file =  "{}/{}_fixedIds.json".format(folder, sub_set)
        # with open(input_file, "r", encoding='utf-8') as reader:
        #     input_data = json.load(reader)["data"]
        #
        # question_count = 0
        # for entry in input_data:
        #     text_list = []
        #     paragraph = entry["paragraph"]
        #     passage = paragraph["text"]
        #     text_list.append(passage)
        #
        #     for qas in paragraph["questions"]:
        #         question = qas["question"]
        #         text_list.append(question)
        #         question_count+=1
        #
        #         answers = [x["text"] for x in qas["answers"]]
        #         text_list+=answers
        #
        #     for text in text_list:
        #         tokens = tokenizer(text)
        #         vocabulary.update({x.lemma_.lower() for x in tokens})


        input_file = "{}/{}.jsonl".format(folder, sub_set)
        with jsonlines.open(input_file, "r") as reader:
            question_count = 0
            for entry in reader.iter():
                text_list = []
                paragraph = entry["passage"]
                passage = paragraph["text"]
                text_list.append(passage)

                for qas in paragraph["questions"]:
                    question = qas["question"]
                    text_list.append(question)
                    question_count += 1

                    answers = [x["text"] for x in qas["answers"]]
                    text_list += answers

                for text in text_list:
                    tokens = tokenizer(text)
                    vocabulary.update({x.lemma_.lower() for x in tokens})
                # print(text_list)
                # break

            #print(question_count)
        return vocabulary



processors = {
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
    "lambada": LambadaProcessor,

}