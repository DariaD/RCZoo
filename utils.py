import copy
import csv
import json
import logging



logger = logging.getLogger(__name__)


def write_vocabulary(vocabulary, taskname):
    # print("writing vocabulary...")
    vocabulary_file_name = "/home/pinecone/Data/Vocabulary/{}.txt".format(taskname)
    vocabulary_file = open(vocabulary_file_name, "w")
    for v in sorted(vocabulary):
        string_out = v + "\n"
        vocabulary_file.write(string_out)
    vocabulary_file.close()


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

    def get_examples(self, data_dir, subset):
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

