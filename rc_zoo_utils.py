import json
import os
import csv
import gzip
import jsonlines
from xml.dom import minidom

# from WikiQA import WikiReaderIterable
from utils import DataProcessor, RCExample, write_vocabulary


class MsMarcoProcessor(DataProcessor):


    def get_all_examples(self, data_dir):
        train_data = self._read_questions_examples(data_dir, "train")
        print("Finished train...")
        dev_data   = self._read_questions_examples(data_dir,  "dev")
        print("Finished dev...")
        test_data  = self._read_questions_examples(data_dir,  "eval")
        print("Finished eval...")

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

    def _read_questions_examples(self, folder, set):
        """Creates examples for the training and dev sets."""
        examples = []
        question_list, passage_list, answer_list, instance_list = [], [], [], []

        context_input_file = "{}{}_v2.1.json.gz".format(folder, set)
        if set == "eval":
            context_input_file = "{}{}_v2.1_public.json.gz".format(folder, set)
        with gzip.open(context_input_file) as f:
            for entry in f:
                entry = eval(entry)
                query_id = entry["query_id"]

                all_passages = entry["passages"]
                queries = entry["query"]
                for keys in query_id.keys():
                    q_id = query_id[keys]
                    print(q_id)


                    passages = all_passages[keys]
                    for passage in passages:
                        passage_list.append(passage["passage_text"])
                    print("Passages", passages)

                    question = queries[keys]
                    print("Question", question)
                    question_list.append(question)


                    if set != "eval":
                        answers = entry["answers"]
                        answer = answers[keys]
                        answer_list += answer
                        print("Answer:", answer)

                    break
                break

        #         example = RCExample(
        #             guid=q_id,
        #             passage=passage_dict[q_id],
        #             question=question,
        #             answer=answer,
        #             instance=None
        #         )
        #         examples.append(example)
        #         #print(example)
        #
        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}



class CNNDailyMailProcessor(DataProcessor):

    def get_all_examples(self, data_dir):

        #questions and answers
        train_data = self._read_examples(data_dir, "training")
        dev_data   = self._read_examples(data_dir,  "validation")
        test_data  = self._read_examples(data_dir,  "test")

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
        write_vocabulary(named_entities, "cnn_named_entities")

        print("Named entities size is:", len(named_entities))

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

    def _read_examples(self, folder, sub_set):
        """Creates examples for the training and dev sets."""
        # question_data_dir = "{}cnn/questions/{}".format(folder, sub_set)
        question_data_dir = "{}/questions/{}".format(folder, sub_set)
        story_data_dir = "{}stories/".format(folder)

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

        # files = os.listdir(story_data_dir)
        # for file in files:
        #     story_input_file = os.path.join(story_data_dir, file)
        #     with open(story_input_file) as reader:
        #         all_lines = reader.readlines()
        #         print(all_lines)
        #         passage = ""
        #         passage_list.append(passage)
        #         break


        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "candidates": ansercandidate_list,
                "answers": answer_list,
                "named_entities": named_entities}



class WhoDidWhatProcessor(DataProcessor):

    def get_all_examples(self, data_dir):

        #questions and answers
        train_data = self._read_examples(data_dir, "train")
        dev_data   = self._read_examples(data_dir,  "val")
        test_data  = self._read_examples(data_dir,  "test")

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



    def _read_examples(self, folder, set):
        """Creates examples for the training and dev sets."""
        if set == "train":
            input_file = "{}Relaxed/{}_key.xml".format(folder, set)
        else:
            input_file = "{}Strict/{}_key.xml".format(folder, set)

        examples = []
        question_list, passage_list, answer_list, instance_list, ansercandidate_list = [], [], [], [], []

        with open(input_file) as reader:
            all_lines = reader.readlines()
            left_part = ""
            i = 0
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


                # i+=1
                # if i > 18:
                #     example = RCExample(
                #         guid="",
                #         passage="",
                #         question=question_list,
                #         answer=answer_list,
                #         instance=None
                #     )
                #     print(example)
                #     # examples.append(example)
                #     break



        # xmldoc = minidom.parse(input_file)
        #
        # instancelist = xmldoc.getElementsByTagName('mc')
        # for instance in instancelist:
        #     question = " ".join(instance.getElementsByTagName('leftContext'), instance.getElementsByTagName('rightContext'))
        #     question_list.append(question)
        #
        #     for answer in instance.getElementsByTagName('choice'):
        #         answer_list.append(answer)
        #
        #

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "candidates": ansercandidate_list,
                "answers": answer_list}




# class WikiQAProcessor(DataProcessor):
#
#     def get_all_examples(self, data_dir):
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
    """Processor for the BoolQ data set (GLUE version)."""

    def get_all_examples(self, data_dir):
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


        #             examples.append(RCExample(
        #                 guid="",
        #                 passage=passage,
        #                 question=question,
        #                 answer=answer,
        #                 instance=None
        #             ))
        #
        #             answer_list.append(answer)
        #             question_list.append(question)
        #
        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": [],
                "answers": answer_list}




class CliCRProcessor(DataProcessor):
    def get_all_examples(self, data_dir):

        #questions and answers
        train_data = self._read_examples(data_dir, "train")
        dev_data   = self._read_examples(data_dir,  "dev")
        test_data  = self._read_examples(data_dir,  "test")

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



    def _read_examples(self, folder, set):


        def clean_text(text):
            text = text.replace("\n", " ").replace("BEG__", "").replace("__END", "")
            return text


        """Creates examples for the training and dev sets."""
        input_file = "{}{}1.0.json".format(folder, set)

        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]

        examples = []
        question_list, passage_list, answer_list, instance_list, ansercandidate_list = [], [], [], [], []
        for entry in input_data:
            entry = entry["document"]

            passage = clean_text(entry["context"])
            passage_list.append(passage)

            for qas in entry["qas"]:
                question = clean_text(qas["query"])
                question_list.append(question)
                for answer_entity in qas["answers"]:
                    answer = answer_entity["text"]
                    answer_list.append(answer)

            example = RCExample(
                guid="",
                passage=passage,
                question=question_list,
                answer=answer_list,
                instance=None
            )
            #examples.append(example)
            #print(example)
            #break

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "candidates": ansercandidate_list,
                "answers": answer_list}



class DuoRCProcessor(DataProcessor):
    """Processor for the BoolQ data set (GLUE version)."""
    def get_all_examples(self, data_dir):

        #questions and answers
        train_data = self._read_questions_examples(data_dir, "train")
        dev_data   = self._read_questions_examples(data_dir,  "dev")
        test_data   = self._read_questions_examples(data_dir,  "test")

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


    def _read_questions_examples(self, folder, set):
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
            for question_entity in entry["qa"]:
                question = question_entity["question"]
                answer = question_entity["answers"]

                question_list.append(question)
                answer_list+=answer

                example = RCExample(
                    guid=entry["id"],
                    passage=passage,
                    question=question,
                    answer=answer,
                    instance=instance
                )
                examples.append(example)

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}

class DROPProcessor(DataProcessor):
        """Processor for the BoolQ data set (GLUE version)."""

        def get_all_examples(self, data_dir):

            # questions and answers
            train_data = self._read_questions_examples(data_dir, "train")
            dev_data = self._read_questions_examples(data_dir, "dev")

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

        def _read_questions_examples(self, folder, set):
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

                    question_list.append(question)
                    answer_list.append(answer)

                    example = RCExample(
                        guid=instance,
                        passage=passage,
                        question=question,
                        answer=answer,
                        instance=instance
                    )
                    examples.append(example)
                #     print(example)
                # break

            return {"examples": examples,
                    "questions": question_list,
                    "passages": passage_list,
                    "instances": instance_list,
                    "answers": answer_list}




class AmazonQAProcessor(DataProcessor):
    """Processor for the BoolQ data set (GLUE version)."""
    def get_all_examples(self, data_dir):

        #questions and answers
        train_data = self._read_questions_examples(data_dir, "train")
        dev_data   = self._read_questions_examples(data_dir,  "val")

        examples = train_data["examples"] + dev_data["examples"]

        question_list = train_data["questions"] + dev_data["questions"]
        answer_list   = train_data["answers"]   + dev_data["answers"]
        passage_list  = train_data["passages"]  + dev_data["passages"]
        instance_list = set(train_data["instances"] + dev_data["instances"])

        return {"examples": [],
                #"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list,}


    def _read_questions_examples(self, folder, set):
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

                # example = RCExample(
                #     guid="_".join([set, str(qas_id)]),
                #     passage=passage,
                #     question=question,
                #     answer=answer,
                #     instance=asin,
                # )
                # examples.append(example)
                # print(example)
                # break


        return {"examples": [],
                #"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list,
                "candidates": candidates_list}

class QuasarProcessor(DataProcessor):


    def get_all_examples(self, data_dir):
        train_data = self._read_questions_examples(data_dir, "train")
        dev_data   = self._read_questions_examples(data_dir,  "dev")
        test_data  = self._read_questions_examples(data_dir,  "test")

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


    def _read_questions_examples(self, folder, set):

        """Creates examples for the training and dev sets."""
        examples = []
        question_list, passage_list, answer_list, instance_list = [], [], [], []


        #context_input_file = "{}contexts/short/{}_contexts.json.gz".format(folder, set)
        context_input_file = "{}contexts/long/{}_contexts.json.gz".format(folder, set)
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
                    passage=passage_dict[q_id],
                    question=question,
                    answer=answer,
                    instance=None
                )
                examples.append(example)
                #print(example)

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}


class RecipeQAProcessor(DataProcessor):
    def get_all_examples(self, data_dir):

        #questions and answers
        train_data = self._read_questions_examples(data_dir, "train")
        dev_data   = self._read_questions_examples(data_dir,  "val")
        test_data  = self._read_questions_examples(data_dir,  "test")

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


    def _read_questions_examples(self, folder, set):
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
                passage=passage,
                question=question,
                answer=answer,
                ansercandidate=ansercandidate,
                instance=None
            )
            examples.append(example)
            #print(example)
            # break

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "candidates": ansercandidate_list,
                "answers": answer_list}


class QAngarooProcessor(DataProcessor):
    """Processor for the BoolQ data set (GLUE version)."""
    def get_all_examples(self, data_dir):

        #questions and answers
        train_data = self._read_questions_examples(data_dir, "train")
        dev_data   = self._read_questions_examples(data_dir,  "dev")

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


    def _read_questions_examples(self, folder, set):
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
                passage=passage,
                question=question,
                answer=answer,
                instance=None,
            )
            examples.append(example)


        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list,
                "candidates": candidates_list}


class HotpotQAProcessor(DataProcessor):
    """Processor for the BoolQ data set (GLUE version)."""
    def get_all_examples(self, data_dir):

        #questions and answers
        train_data = self._read_questions_examples(data_dir, "train")
        dev_data   = self._read_questions_examples(data_dir,  "dev")
        test_data   = self._read_questions_examples(data_dir,  "test")

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


    def _read_questions_examples(self, folder, set):
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
            ansercandidate = entry["candidates"]
            answer = None
            if set is not "test":
                answer = entry["answer"]
                answer_list.append(answer)
            question_list.append(question)
            passage_list.append(passage)
            ansercandidate_list.append(ansercandidate)

            example = RCExample(
                guid=qas_id,
                passage=passage,
                question=question,
                answer=answer,
                ansercandidate=ansercandidate,
                instance=None
            )
            examples.append(example)

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "ansercandidate": ansercandidate_list,
                "answers": answer_list}


class TurkQAProcessor(DataProcessor):
    """Processor for the BoolQ data set (GLUE version)."""

    def get_all_examples(self, data_dir):
        examples = []
        """Read a SQuAD json file into a list of SquadExample."""
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
                        passage=passage,
                        question=question,
                        answer=answer,
                        instance=None
                    ))

                    answer_list.append(answer)
                    question_list.append(question)

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": [],
                "answers": answer_list}




class ShARCProcessor(DataProcessor):
    """Processor for the BoolQ data set (GLUE version)."""
    def get_all_examples(self, data_dir):

        #questions and answers
        train_data = self._read_questions_examples(data_dir, "train")
        dev_data   = self._read_questions_examples(data_dir,  "dev")

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


    def _read_questions_examples(self, folder, set):
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
                passage=passage,
                question=question,
                answer=answer,
                instance=instance
            )
            examples.append(example)

        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}


class NarrativeQAProcessor(DataProcessor):
    """Processor for the BoolQ data set (GLUE version)."""

    def get_all_examples(self, data_dir):
        examples = []
        question_list, answer_list, passage_list = [], [], []

        pre_example_passage_dict = {}
        with open(data_dir + 'third_party/wikipedia/summaries.csv', newline='') as csvfile:
            narrative_data = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(narrative_data)
            for row in narrative_data:
                id = row[0]
                passage = row[2]
                pre_example_passage_dict[id] = passage
                passage_list.append(passage)

        with open(data_dir + 'qaps.csv', newline='') as csvfile:
            narrative_data = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(narrative_data)
            i = 0
            for row in narrative_data:
                doc_id = row[0]
                question = row[2]
                answer_1 = row[3]
                answer_2 = row[4]

                question_list.append(question)
                answer_list.append(answer_1)
                answer_list.append(answer_2)

                example = RCExample(
                    guid="_".join([row[1], str(i)]),
                    passage=" ".join(pre_example_passage_dict[doc_id]),
                    question=question,
                    answer=[answer_1, answer_2],
                    instance=None
                )
                i+=1
                examples.append(example)


        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": [],
                "answers": answer_list}


class WikiMovieProcessor(DataProcessor):
    """Processor for the BoolQ data set (GLUE version)."""

    def get_all_examples(self, data_dir):

        # Data (passages)
        knowledge_data = self._read_passages(data_dir)

        #questions and answers
        train_data = self._read_questions_examples(data_dir, "train")
        dev_data   = self._read_questions_examples(data_dir,  "dev")
        test_data  = self._read_questions_examples(data_dir, "test")

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


    def _read_questions_examples(self, folder, set):
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


        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}


class SQuAD2Processor(DataProcessor):
    """Processor for the BoolQ data set (GLUE version)."""

    def get_examples(self, data_dir, subset):
        """See base class."""
        filename = os.path.join(data_dir, subset + "-v2.0.json")
        return self._read_examples(filename, subset)

    def get_all_examples(self, data_dir):
        train_data = self.get_examples(data_dir, "train")
        dev_data = self.get_examples(data_dir, "dev")

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



    def _read_examples(self, input_file, str_id):

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

                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question = qa["question"]
                    question = question.replace("??", "?")

                    is_impossible = qa["is_impossible"]
                    if not is_impossible:
                        answer = qa["answers"][0]["text"]
                        answer_list.append(answer)
                    else:
                        answer = None


                    question_list.append(question)


                    example = RCExample(
                        guid=qas_id,
                        passage=passage,
                        question=question,
                        answer=answer,
                        instance=instance
                    )
                    examples.append(example)


        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}


class SQuADProcessor(DataProcessor):
    """Processor for the BoolQ data set (GLUE version)."""

    def get_examples(self, data_dir, subset):
        """See base class."""
        filename = os.path.join(data_dir, subset + "-v1.1.json")
        return self._read_examples(filename, subset)

    def get_all_examples(self, data_dir):
        train_data = self.get_examples(data_dir, "train")
        dev_data = self.get_examples(data_dir, "dev")

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



    def _read_examples(self, input_file, str_id):

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

                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question = qa["question"]
                    question = question.replace("??", "?")
                    answer = qa["answers"][0]["text"]

                    question_list.append(question)
                    answer_list.append(answer)

                    example = RCExample(
                        guid=qas_id,
                        passage=passage,
                        question=question,
                        answer=answer,
                        instance=instance
                    )
                    examples.append(example)


        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}


class PubMedProcessor(DataProcessor):
    """Processor for the BoolQ data set (GLUE version)."""

    def get_all_examples(self, data_dir):
        filename = os.path.join(data_dir, "ori_pqal.json")
        return self._read_examples(filename, "")


    def get_labels(self):
        """See base class."""
        return [False, True, "Maybee"]


    def _read_examples(self, input_file, str_id):
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
                    passage=" ".join(passages),
                    question=question,
                    answer=answer,
                    instance=None
                )
                examples.append(example)


        return {"examples": examples,
                "questions": question_list,
                "passages": passage_list,
                "instances": instance_list,
                "answers": answer_list}


class BoolQProcessor(DataProcessor):
    """Processor for the BoolQ data set (GLUE version)."""

    def get_examples(self, data_dir, subset):
        """See base class."""
        filename = os.path.join(data_dir, subset + ".jsonl")
        return self._read_examples(filename, subset)

    def get_all_examples(self, data_dir):
        train_data = self.get_examples(data_dir, "train")
        dev_data = self.get_examples(data_dir,  "dev")
        test_data = self.get_examples(data_dir, "test")

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


    def _read_examples(self, input_file, str_id):
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
                    passage=json_entry["passage"],
                    question=json_entry["question"],
                    answer=answer,
                    instance=json_entry["title"]
                )
                examples.append(example)
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
    "drop": DuoRCProcessor,
    "clicr": CliCRProcessor,
    "babi": bAbIProcessor,
    #"wikiqa": WikiQAProcessor,
    "wdw": WhoDidWhatProcessor,
    "cnn": CNNDailyMailProcessor,
    "dailymail": CNNDailyMailProcessor,
    "msmarco": MsMarcoProcessor,
}

