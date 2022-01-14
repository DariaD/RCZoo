import copy
import csv
import json
import logging
import os
import en_core_web_sm
from nltk import sent_tokenize

from parameters import dataset_names, save_data_path

logger = logging.getLogger(__name__)


# spicy tokeniser
from spacy.lang.en import English


#nlp_NER = en_core_web_sm.load()

nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)

# sentencizer = nlp.create_pipe("sentencizer")
# nlp.add_pipe(sentencizer)

def process_lingu_data(text_list):
    s, avg_tokens = 0, 0
    vocabulary = set()
    named_entities = {}
    #for text in text_list:
        # # print(text)
        # sentences = sent_tokenize(text)
        # for sentence in sentences:
        #     try:
        #         # VOCABULARY
        #         tokens = tokenizer(sentence)
        #         vocabulary.update({x.lemma_.lower() for x in tokens})
        #
        #         # NER
        #         # doc = nlp_NER(sentence)
        #         # if len(doc.ents) > 0:
        #         #     for X in doc.ents:
        #         #         if X.label_ not in named_entities:
        #         #             named_entities[X.label_] = set()
        #         #         named_entities[X.label_].update({X.text.lower()})
        #
        #     except UnicodeEncodeError:
        #         print("Unicode error happens")
        #         tokens = sentence.split()
        #     s = s + len(tokens)

    if len(text_list) > 0:
        avg_tokens = s/len(text_list)

    return "{:.1f}".format(avg_tokens), vocabulary, named_entities


def get_vocabulary(text_list):
    s, avg_tokens = 0, 0
    vocabulary = set()
    for text in text_list:
        try:
            tokens = tokenizer(text)
            vocabulary.update({x.lemma_.lower() for x in tokens})

        except UnicodeEncodeError:
            print("Unicode error happens")
            tokens = text.split()
        s = s + len(tokens)

    if len(text_list) > 0:
        avg_tokens = s / len(text_list)

    return "{:.1f}".format(avg_tokens), vocabulary


def update_ner(nemed_entities_dict, new_dict):
    for key in new_dict.keys():
        if key in nemed_entities_dict.keys():
            nemed_entities_dict[key].update(new_dict[key])
        else:
            nemed_entities_dict[key] = new_dict[key]
    return nemed_entities_dict



def write_vocabulary(vocabulary, taskname):
    # print("writing vocabulary...")
    dataset = dataset_names[taskname]
    vocabulary_path = os.path.join(save_data_path, "Vocabulary")
    if not os.path.exists(vocabulary_path):
        os.makedirs(vocabulary_path)
    vocabulary_file_name = os.path.join(vocabulary_path, "{}.txt".format(dataset))
    vocabulary_file = open(vocabulary_file_name, "w")
    for v in sorted(vocabulary):
        string_out = v + "\n"
        vocabulary_file.write(string_out)
    vocabulary_file.close()


def write_questions(question_list, taskname):
    # print("writing vocabulary...")
    dataset = dataset_names[taskname]
    path = os.path.join(save_data_path, "Questions")
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = os.path.join(path, "{}.txt".format(dataset))
    file = open(file_name, "w")
    for v in question_list:
        string_out = v + "\n"
        file.write(string_out)
    file.close()


def write_named_entities(named_entities, taskname):
    # print("writing vocabulary...")
    dataset = dataset_names[taskname]
    for label in named_entities.keys():
        ne_file_path = os.path.join(save_data_path, "NamedEntity", dataset)
        ne_file_name = os.path.join(ne_file_path, "{}_{}.txt".format(label, dataset))
        if not os.path.exists(ne_file_path):
            os.makedirs(ne_file_path)
        ne_file = open(ne_file_name, "w")
        ne_vocabulary = named_entities[label]
        for v in sorted(ne_vocabulary):
            string_out = v + "\n"
            ne_file.write(string_out)
        ne_file.close()


def write_dictionary(data, folder, taskname):
    dataset = dataset_names[taskname]
    for label in data.keys():
        ne_file_path = os.path.join(save_data_path, folder, dataset)
        ne_file_name = os.path.join(ne_file_path, "{}_{}.txt".format(label, dataset))
        if not os.path.exists(ne_file_path):
            os.makedirs(ne_file_path)
        ne_file = open(ne_file_name, "w")
        ne_vocabulary = data[label]
        for v in sorted(ne_vocabulary):
            string_out = v + "\n"
            ne_file.write(string_out)
        ne_file.close()


def write_list(list_to_write, folder, taskname):
    # print("writing vocabulary...")
    dataset = dataset_names[taskname]
    path = os.path.join(save_data_path, folder)
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = os.path.join(path, "{}.txt".format(dataset))
    file = open(file_name, "w")
    for v in sorted(list_to_write):
        string_out = v + "\n"
        file.write(string_out)
    file.close()


def get_all_subfiles(folder, current_list, file_type):
    inside_files = os.listdir(folder)

    for file in inside_files:
        file_path = os.path.join(folder,file)
        if file_type in file:
            current_list.append(file_path)
        else:
            get_all_subfiles(file_path, current_list, file_type)
    return current_list




class RCDocExample(object):
    """
    A single RC dataset example.

    Args:
        guid: Unique id for the example.
        question: question
        passage: passage or concatination of the passages
        answer: answer if available
        instance:

        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, questions, passages=None, answercandidate=None, answer=None, instance=None):
        self.guid = guid
        self.passage = passages
        self.question = questions
        self.answercandidate = answercandidate
        self.answer = answer
        self.instance = instance

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"




class RCExample(object):
    """
    A single RC dataset example.

    Args:
        guid: Unique id for the example.
        question: question
        passage: passage or concatination of the passages
        answer: answer if available
        instance:

        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, question, passage=None, ansercandidate=None, answer=None, instance=None):
        self.guid = guid
        self.question = question
        self.passage = passage
        self.ansercandidate = ansercandidate
        self.answer = answer
        self.instance = instance

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"



class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""


    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_examples(self, data_dir, subset, debug_flag):
        """Gets a collection of `InputExample`s for the set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def get_passages(self):
        """Gets the list of all passages (docs) for this data set."""
        raise NotImplementedError()

    def get_questions(self):
        """Gets the list of all questions for this data set."""
        raise NotImplementedError()

    def get_answers(self):
        """Gets the list of all ansers and anser candidates for this data set."""
        raise NotImplementedError()

    def tfds_map(self, example):
        """Some tensorflow_datasets datasets are not formatted the same way the GLUE datasets are.
        This method converts examples to the correct format."""
        if len(self.get_labels()) > 1:
            example.label = self.get_labels()[int(example.label)]
        return example

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))


class VocabularyProcessor(object):
    def get_vocabulary_size(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_text_list(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()


