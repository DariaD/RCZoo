import argparse
import os
import jsonlines
import json

from utils import write_vocabulary, write_named_entities, get_vocabulary, update_ner, write_questions

from parameters import datapaths, dataset_names, data_folder, result_folder
from rc_zoo_utils import processors

import stanza

stanza.download('en')
nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse,ner')

common_NE_tags = ["PERSON", "ORG", "GPE", "LOC"]

# Alternative lemmas for super long text
from nltk.stem import PorterStemmer
porter = PorterStemmer()


# used for saving questions, vocabulary and
def save_strings_data(path, filename, data_to_save, sorted_flag=False):
    file = open(os.path.join(path, "{}.txt".format(filename)), "w")
    unicode_error_count = 0
    if sorted_flag:
        data_to_save = sorted(data_to_save)
    for v in data_to_save:
        try:
            file.write("{}\n".format(v))
        except UnicodeEncodeError:
            unicode_error_count+=1
            continue
    file.close()
    #print("File {} saved. Unicode error happened {} times.".format(filename, unicode_error_count))



def process_doc(input_doc):
    words, lemmas, named_entities = [], [], []
    # print(input_doc)
    for sentence in input_doc.sentences:
        for word in sentence.words:
            words.append(word.text)
            lemmas.append(word.lemma)
        for ent in sentence.ents:
            named_entities.append(ent)
    return words, lemmas, named_entities, len(input_doc.sentences)


def compare_ner(ner_q, ner_p):
    question_ner = set()
    personal_question_ner = set()
    passage_ner = set()
    personal_passage_ner = set()

    NE_overlap = False
    personal_NE_overlap = False

    for named_entity in ner_p:
        question_ner.add(named_entity.text.lower())
        if named_entity.type in common_NE_tags:
            personal_question_ner.add(named_entity.text.lower())

    for named_entity in ner_q:
        passage_ner.add(named_entity.text.lower())
        if named_entity.type in common_NE_tags:
            personal_passage_ner.add(named_entity.text.lower())

    for q_named_entity in question_ner:
        if q_named_entity in passage_ner:
            NE_overlap = True

    for q_named_entity in personal_question_ner:
        if q_named_entity in personal_passage_ner:
            personal_NE_overlap = True

    return NE_overlap, personal_NE_overlap




def get_stanza_doc(input_text):
    try:
        stanza_doc = nlp(input_text)
    except Exception as e:
        print(e)
        print("Input text:", input_text)
        print(len(input_text))
        return ""

    return stanza_doc



# The main method called from run_pipiline
def process_examples(dataset, examples, debug_flag, light_flag):


    # Paths: where to save:
    path = os.path.join(result_folder, dataset)
    if debug_flag:
        path = os.path.join(path, "debug")
    sample_path = os.path.join(path, "samples")
    named_entity_path = os.path.join(path, "named_entities")
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
    if not os.path.exists(named_entity_path):
        os.makedirs(named_entity_path)

    # Inicialisation

    # Number of samples (question) where the is no any NE in question or passage
    no_NER_count = 0
    # Number of samples (question) where the question and passage have a common entity
    overlap_NER_count = 0
    # Same as above but only: LOC, PERSON, ORG
    overlap_personal_NER_count = 0

    all_examples = []
    samples_count = 0

    question_token_size_list = []
    question_ner_size_list = []

    passage_token_size_list = []
    passage_sentence_size_list = []
    passage_ner_size_list = []

    answer_token_size_list = []
    answer_ner_size_list = []

    word_vocabulary = set()
    lemma_vocabulary = set()

    all_questions_list = []

    NER_dict = {}

    # Example processing
    invalid_count = 0
    currapt_count = 0

    #
    # # SPECIAL CASE FOR ARC AND QASC
    # if dataset in ["arc", "qasc"]:
    #     for passage in external_passages:
    #         is_passage_a_valid_string = isinstance(passage, str) and len(passage.strip()) > 0
    #         if not is_passage_a_valid_string:
    #             print("Passage is invalid: ", passage)
    #             invalid_count += 1
    #             continue
    #         passage_doc = get_stanza_doc(passage)
    #         if passage_doc == "":
    #             print("Currupted passage skipped")
    #             currapt_count += 1
    #             break
    #         p_words, p_lemmas, p_ner, sentence_size = process_doc(passage_doc)
    #         #p_ne_list_of_lists.append(p_ner)
    #         passage_ner_size_list.append(len(p_ner))
    #         passage_sentence_size_list.append(sentence_size)
    #
    #         passage_token_size_list.append(len(p_words))
    #
    #         word_vocabulary.update(p_words)
    #         lemma_vocabulary.update(p_lemmas)
    #


    for example in examples:

        # Passage
        passage_list = example.passage
        p_ne_list_of_lists = []
        passage_to_safe_list = []
        for passage in passage_list:
            is_passage_a_valid_string = isinstance(passage, str) and len(passage.strip()) > 0
            if not is_passage_a_valid_string:
                print("Passage is invalid: ", passage)
                invalid_count += 1
                continue
            #too long text
            #if len(passage) > 100000:
            #     print("The passage is too long:", len(passage))
            #     p_words = passage.split()
            #     p_lemmas = [porter.stem(x) for x in p_words]
            #     passage_to_safe = {"text": passage, "doc": {},
            #                        "named_entities": {}}
            #
            # else:
            passage_doc = get_stanza_doc(passage)
            if passage_doc  == "":
                print("Currupted passage skipped")
                currapt_count += 1
                break
#                continue
            p_words, p_lemmas, p_ner, sentence_size = process_doc(passage_doc)
            p_ne_list_of_lists.append(p_ner)
            passage_ner_size_list.append(len(p_ner))
            passage_sentence_size_list.append(sentence_size)
            passage_to_safe = {"text": passage, "doc": passage_doc.to_dict(),
                                   "named_entities": [x.to_dict() for x in p_ner]}

            passage_to_safe_list.append(passage_to_safe)

            passage_token_size_list.append(len(p_words))

            word_vocabulary.update(p_words)
            lemma_vocabulary.update(p_lemmas)
            # else:
            #     print("Wrong passage:", passage)
            #     print(example)


        # Question
        question_list = example.question
        q_ne_list_of_lists = []
        question_to_safe_list = []
        for question in question_list:
            all_questions_list.append(question)
            #print("In Question")
            question_doc = get_stanza_doc(question)
            if question_doc  == "":
                print("Currupted question skipped")

                continue
            q_words, q_lemmas, q_ner, sentence_size = process_doc(question_doc)
            q_ne_list_of_lists.append(q_ner)
            question_token_size_list.append(len(q_words))
            question_ner_size_list.append(len(q_ner))

            word_vocabulary.update(q_words)
            lemma_vocabulary.update(q_lemmas)

            question_to_safe = {"text": question, "doc": question_doc.to_dict(),
                                "named_entities": [x.to_dict() for x in q_ner]}
            question_to_safe_list.append(question_to_safe)



        # Answer
        answer_list = example.answer
        a_ne_list_of_lists = []
        answer_to_safe_list = []
        a_ner = []
        for answer in answer_list:
            is_answer_a_valid_string = isinstance(answer, str) and len(answer.strip()) > 0
            answer_doc = ""
            #if debug_flag:
            #    print(is_answer_a_valid_string)
            if is_answer_a_valid_string:
                #print("In Answer")
                answer_doc = get_stanza_doc(answer)
                if answer_doc == "":
                    print("Currapted answer skipped")
                    continue
                a_words, a_lemmas, a_ner, sentence_size = process_doc(answer_doc)
                a_ne_list_of_lists.append(a_ner)
                answer_token_size_list.append(len(a_words))
                word_vocabulary.update(a_words)
                lemma_vocabulary.update(a_lemmas)
                answer_ner_size_list.append(len(a_ner))


                word_vocabulary.update(a_words)
                lemma_vocabulary.update(a_lemmas)

            if isinstance(answer, bool) or isinstance(answer, int):
                answer_token_size_list.append(1)

            answer_to_safe = {"text": answer,
                              "doc": answer_doc.to_dict() if is_answer_a_valid_string else answer_doc,
                              "named_entities": [x.to_dict() for x in a_ner]}
            answer_to_safe_list.append(answer_to_safe)


        # Calculation overlap of NE in question and passage
        for p_ner in p_ne_list_of_lists:
            for q_ner in q_ne_list_of_lists:
                if len(q_ner) > 0 and len(p_ner) > 0:
                    NE_overlap_bool, personal_NE_overlap_bool = compare_ner(q_ner, p_ner)
                    if NE_overlap_bool:
                        overlap_NER_count += 1
                    if personal_NE_overlap_bool:
                        overlap_personal_NER_count += 1
                else:
                    no_NER_count += 1


        # Create a list of NE per type in Dictionary format
        all_NE = q_ne_list_of_lists + p_ne_list_of_lists + a_ne_list_of_lists

        for ner_list in all_NE:
            for named_entity in ner_list:
                if named_entity.type not in NER_dict:
                    NER_dict[named_entity.type] = set()
                NER_dict[named_entity.type].update([named_entity.text.lower()])


        # Create example to save
        if not light_flag or debug_flag:
            example_to_save = {}
            example_to_save["id"]       = example.guid
            example_to_save["passage"]  = passage_to_safe_list
            example_to_save["question"] = question_to_safe_list
            example_to_save["answer"]   = answer_to_safe_list

            all_examples.append(example_to_save)


            # save samples individually in txt
            file_name = os.path.join(sample_path, "{}.txt".format(example.guid))
            file = open(file_name, "w")
            file.write(str(example_to_save))
            file.close()

        samples_count += 1

        # output process
        if samples_count % 2000 == 0:
            print(samples_count, "samples processed..")

        if samples_count % 50000 == 0:
            save_strings_data(path, "questions_backup",  all_questions_list, False)
            save_strings_data(path, "vocabulary_backup", lemma_vocabulary,   True)
            print(samples_count, "saving backup")

    # end of example loop
    print("Number of examples:", len(examples))
    print("Len currupted:", currapt_count)
    print("Len invalid:", invalid_count)

    if debug_flag:
        print("example procced")

    ###############################
    ####### SAVE FILES ############
    ###############################

    # save questions and answers

    save_strings_data(path, "questions", all_questions_list, False)
    save_strings_data(path, "answers",  answer_to_safe_list, False)

    # save vocabulary
    save_strings_data(path, "vocabulary", lemma_vocabulary, True)

    # safe NER
    ner_label_size = {}
    for label in NER_dict.keys():
        ne_vocabulary = NER_dict[label]
        ner_label_size[label] = len(ne_vocabulary)
        save_strings_data(named_entity_path, label, ne_vocabulary, True)

    if debug_flag:
        print("NE saved")


    ###############################
    ####### SAVE STATISTICS #######
    ###############################


    number_of_questions = len(question_token_size_list)
    number_of_passages = len(passage_token_size_list)
    number_of_answers = len(answer_token_size_list)
    unique_ner_size = sum([len(NER_dict[x]) for x in NER_dict.keys()])

    avg_question_len = sum(question_token_size_list) / number_of_questions
    if number_of_passages > 0:
        avg_passage_len = sum(passage_token_size_list) / number_of_passages
        avg_passage_len_in_sentences = sum(passage_sentence_size_list) / number_of_passages
    else:
        avg_passage_len = "-"
        avg_passage_len_in_sentences = "-"
    avg_answer_len = sum(answer_token_size_list) / number_of_answers

    statistics = {}

    statistics["dataset"] = dataset
    statistics["general"] = {
        "question": number_of_questions,
        "passage": number_of_passages,
        # "answer"   : ,
    # AVG
        "AVG_question_len": avg_question_len,
        "AVG_passage_len": avg_passage_len,
        "AVG_passage_len_in sentences": avg_passage_len_in_sentences,
        "AVG_answer_len": avg_answer_len,
    }

    statistics["vocabulary"] = {
        "words": len(word_vocabulary),
        "lemmas": len(lemma_vocabulary),
        #"ratio": "{:.2f}".format(len(lemma_vocabulary) / len(word_vocabulary))
        "ratio": len(lemma_vocabulary) / len(word_vocabulary)
    }

    statistics["NE"] = {
        "total": sum(question_ner_size_list) + sum(passage_ner_size_list),
        "unique": unique_ner_size,
        "question_ratio": sum(question_ner_size_list) / number_of_questions,
        "passage_ratio": sum(passage_ner_size_list) / number_of_passages if number_of_passages > 0 else "-",
        "label_count": ner_label_size,

        "no_named_entities": no_NER_count / number_of_passages if number_of_passages > 0 else "-",
        "all_ovelap": overlap_NER_count / number_of_passages if number_of_passages > 0 else "-",
        "personal_overlap": overlap_personal_NER_count / number_of_passages if number_of_passages > 0 else "-",
    }

    # with jsonlines.open(os.path.join(path, "statistics.json"), 'w') as writer:
    #     writer.write_all(statistics)

    with open(os.path.join(path, "statistics.json"), 'w') as statisticsfile:
        json.dump(statistics, statisticsfile)

    if debug_flag:
        print("Statistics saved to ", os.path.join(path, "statistics.json"))

    str_to_write = """
    Dataset: {}
    Number of Examples: \t{}
    Number of Questions: \t{}
    Number of Passages: \t{}\n

    AVG Question length: \t{:.2f} 
    AVG Passage  length (tokens): \t{:.2f}
    AVG Passage  length (sentences): \t{:.2f}
    AVG Answer   length: \t{:.2f}\n

    Vocabulary:
    \t words:  \t{}
    \t lemmas: \t{}
    \t ration of lemmas to words: {:.2f}\n

    Named Entities:
    \t total:        \t\t{}
    \t unique:       \t\t{}
    \t per question: \t{:.2f}
    \t per passage:  \t{:.2f}

    \t no NE:        \t{:.2f}
    \t common NE:    \t{:.2f}
    \t all NE:       \t{:.2f}

    \n

    """.format(

        # General
        dataset, samples_count, number_of_questions, number_of_passages,
        # AVG
        avg_question_len,
        avg_passage_len if number_of_passages > 0 else 0,
        avg_passage_len_in_sentences if number_of_passages > 0 else 0,
        avg_answer_len,
        # Vocabulary
        len(word_vocabulary), len(lemma_vocabulary),
        len(lemma_vocabulary) / len(word_vocabulary),
        # NER
        sum(question_ner_size_list) + sum(passage_ner_size_list),
        unique_ner_size,
        sum(question_ner_size_list) / number_of_questions if number_of_passages > 0 else 0,
        sum(passage_ner_size_list) / number_of_passages if number_of_passages > 0 else 0,
        no_NER_count / number_of_passages if number_of_passages > 0 else 0,
        overlap_NER_count / number_of_passages if number_of_passages > 0 else 0,
        overlap_personal_NER_count / number_of_passages if number_of_passages > 0 else 0,
        )
    # """.format([str(x) for x in output_parameters])

    file = open(os.path.join(path, "statistics.txt"), "w")
    file.write(str_to_write)
    ner_stat_out = ""
    for label in NER_dict.keys():
        ne_vocabulary = NER_dict[label]
        ner_stat_out = "{}\t{}\t{}\n".format(ner_stat_out, label, str(len(ne_vocabulary)))
    file.write(ner_stat_out)
    file.close()
    if debug_flag:
        print("Statistics saved to ", os.path.join(path, "statistics.txt"))


    # save all data
    if not light_flag or debug_flag:
        data_save_path = os.path.join(path, "data.jsonl")
        with jsonlines.open(data_save_path, 'w') as writer:
            writer.write_all(all_examples)

        if debug_flag:
            print("Data saved to ", data_save_path)

    print("Complete successfully!")

