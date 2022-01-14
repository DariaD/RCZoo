"""This file is based by WikiReader repository.

This file contains WikiReaderIterable and WikiReaderStatic for handling the WikiQA dataset

Use WikiReaderIterable when you want data in the format of query, docs, labels seperately
Example:
query_iterable = WikiReaderIterable('query', path_to_file)

Use WikiReaderStatic when you want a dump of the test data with the doc_ids and query_ids
It is useful for saving predictions in the TREC format

A datapoint in this dataset has a query, a document and thier relevance(0: irrelevant, 1: relevant)

Example data point:
QuestionID  Question    DocumentID  DocumentTitle   SentenceID  Sentence    Label
Q8  How are epithelial tissues joined together? D8  Tissue (biology)    D8-0    Cross section of sclerenchyma fibers in plant ground tissue 0

"""
import os

import numpy as np
import re
import csv
from spacy.lang.en import English

from parameters import dataset_names, datapaths
from utils import get_vocabulary, write_vocabulary, write_questions

nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)

class WikiReaderIterable:
    """Returns an iterable for the given `iter_type` after extracting from the WikiQA tsv

    Parameters
    ----------
    iter_type : {'query', 'doc', 'label'}
        The type of data point
    fpath : str
        Path to the .tsv file
    """

    def __init__(self, iter_type, fpath):
        self.type_translator = {'query': 0, 'doc': 1, 'label': 2}
        self.iter_type = iter_type
        with open(fpath, encoding='utf8') as tsv_file:
            tsv_reader = csv.reader(tsv_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_NONE)
            self.data_rows = []
            for row in tsv_reader:
                self.data_rows.append(row)

    def preprocess_sent(self, sent):
        """Utility function to lower, strip and tokenize each sentence
        Replace this function if you want to handle preprocessing differently

        Parameters
        ----------
        sent : str
        """
        #return re.sub("[^a-zA-Z0-9]", " ", sent.strip().lower()).split()
        return sent

    def __iter__(self):
        # Defining some consants for .tsv reading
        # These refer to the column indexes of certain data
        QUESTION_ID_INDEX = 0
        QUESTION_INDEX = 1
        ANSWER_INDEX = 5
        LABEL_INDEX = 6

        # We will be grouping all documents and labels which belong to one question into
        # one group. This helps in getting MAP scores.
        document_group = []
        label_group = []

        # We keep count of number of documents so we can remove those question-doc pairs
        # which do not have even one relevant document
        n_relevant_docs = 0
        n_filtered_docs = 0

        queries = []
        docs = []
        labels = []

        for i, line in enumerate(self.data_rows[1:], start=1):
            if i < len(self.data_rows) - 1:  # check if out of bounds might occur
                # If the question id index doesn't change
                if self.data_rows[i][QUESTION_ID_INDEX] == self.data_rows[i + 1][QUESTION_ID_INDEX]:
                    document_group.append(self.preprocess_sent(self.data_rows[i][ANSWER_INDEX]))
                    label_group.append(int(self.data_rows[i][LABEL_INDEX]))
                    n_relevant_docs += int(self.data_rows[i][LABEL_INDEX])
                else:
                    document_group.append(self.preprocess_sent(self.data_rows[i][ANSWER_INDEX]))
                    label_group.append(int(self.data_rows[i][LABEL_INDEX]))

                    n_relevant_docs += int(self.data_rows[i][LABEL_INDEX])

                    if n_relevant_docs > 0:
                        docs.append(document_group)
                        labels.append(label_group)
                        queries.append(self.preprocess_sent(self.data_rows[i][QUESTION_INDEX]))

                        yield [queries[-1], document_group, label_group][self.type_translator[self.iter_type]]
                    else:
                        # Filter out a question if it doesn't have a single relevant document
                        n_filtered_docs += 1

                    n_relevant_docs = 0
                    document_group = []
                    label_group = []

            else:
                # If we are on the last line
                document_group.append(self.preprocess_sent(self.data_rows[i][ANSWER_INDEX]))
                label_group.append(int(self.data_rows[i][LABEL_INDEX]))
                n_relevant_docs += int(self.data_rows[i][LABEL_INDEX])

                if n_relevant_docs > 0:
                    docs.append(document_group)
                    labels.append(label_group)
                    queries.append(self.preprocess_sent(self.data_rows[i][QUESTION_INDEX]))
                    # Return the index of the doc requested
                    yield [queries[-1], document_group, label_group][self.type_translator[self.iter_type]]
                else:
                    n_filtered_docs += 1
                    n_relevant_docs = 0



#filename = "/home/pinecone/Data/WikiQACorpus/WikiQA-dev.tsv"
#filename = "/home/pinecone/Data/WikiQACorpus/WikiQA.tsv"
task_name = "wikiqa"
filename = os.path.join(datapaths[task_name], "WikiQA.tsv")

documents = WikiReaderIterable("doc", filename)
passages_list = list(iter(documents))

questions = WikiReaderIterable("query", filename)
questions_list = list(iter(questions))
print("Number of questions", len(questions_list))

task_name = "wikiqa"
answers = WikiReaderIterable("label", filename)
answers_list = list(iter(answers))

vocabulary = set()

if len(questions_list) > 0:
    avg_q, vocab_q = get_vocabulary(questions_list)
    #print(questions_list)
    vocabulary.update(vocab_q)

if len(passages_list) > 0:
    #print(passages_list)
    new_passage_list = [" ".join(sentences) for sentences in passages_list]
    avg_p, vocab_p = get_vocabulary(new_passage_list)
    vocabulary.update(vocab_p)

# if len(answers_list) > 0:
#     #print(answers_list)
#     flatten_list = []
#     for answer_list_entity in answers_list:
#         flatten_list += answer_list_entity
#     avg_a, vocab_a = get_vocabulary(flatten_list)
#     vocabulary.update(vocab_a)


write_vocabulary(vocabulary, task_name)
write_questions(questions_list, task_name)

dataset = dataset_names[task_name]
# & # instances	& # passages &	# A/Q & AVG Q len	& AVG P len	 & AVG A len & Vcabulary Size
# print("&".join(
#     [str(x) for x in [dataset, len_instances, len(passages), avg_a_candidate, avg_q, avg_p, avg_a, len(vocabulary)]]))
#
# for text in questions_list:
#     tokens = tokenizer(text)
#     vocabulary.update({x.lemma_.lower() for x in tokens})
#
# new_passages_list = []
# for text in passages_list:
#     text = " ".join(text)
#     new_passages_list.append(text)
#     tokens = tokenizer(text)
#     vocabulary.update({x.lemma_.lower() for x in tokens})


# #print(passages_list)
# list_of_length = [len(x) for x in questions_list]
# avg_q = sum(list_of_length) / len(questions_list)
# avg_p = sum([len(x) for x in new_passages_list]) / len(new_passages_list)

# & # instances	& # passages &	# A/Q & AVG Q len	& AVG P len	 & AVG A len & Vcabulary Size
print("&".join([str(x) for x in ["", "-", len(passages_list), "-", avg_q, avg_p, "-",
                                 len(vocabulary)]]))