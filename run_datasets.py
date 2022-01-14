import argparse

from utils import write_vocabulary, write_named_entities, get_vocabulary, update_ner, write_questions

from parameters import datapaths, dataset_names
from rc_zoo_utils import processors

def main():
    # print(get_data(["just text", "text with \ude03"]))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )

    args = parser.parse_args()
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    datapath = datapaths[args.task_name]
    data = processor.get_all_examples(datapath)

    #print("\tData collected")

    questions = data["questions"]
    passages  = data["passages"]
    answers   = data["answers"]
    instances = data["instances"]


    avg_q, avg_p, avg_a = 0,0,0
    vocabulary = set()
    named_entities = {}
    replace_token = "-"



    if len(questions) > 0:
        avg_q, vocab_q = get_vocabulary(questions)
        vocabulary.update(vocab_q)

    if len(passages) > 0:
        avg_p, vocab_p = get_vocabulary(passages)
        vocabulary.update(vocab_p)

    if len(answers) > 0:
        avg_a, vocab_a = get_vocabulary(answers)
        vocabulary.update(vocab_a)
    else:
        avg_a = replace_token

    avg_a_candidate = replace_token

    if "candidates" in data:
        cnadidatelist = data["candidates"]
        avg_a_candidate, vocab_a_c = get_vocabulary(cnadidatelist)
        vocabulary.update(vocab_a_c)
        avg_a_candidate = len(cnadidatelist)/len(questions)

    # print("AVG Q len:", avg_q)
    # print("AVG P len:", avg_p)
    # print("AVG A len:", avg_a)
    # print("VOCAB size:", len(vocabulary))
    # print("Number of Questions:", len(questions))
    # print("Number of Passages:", len(passages))
    # print("AVG Number of AnswerCandidate:", avg_a_candidate)

    len_instances = replace_token
    if len(instances) > 0:
        len_instances = len(instances)

    write_vocabulary(vocabulary, args.task_name)
    write_questions(questions, args.task_name)
    #write_named_entities(named_entities, args.task_name)

    dataset = dataset_names[args.task_name]
   # & # instances	& # passages &	# A/Q & AVG Q len	& AVG P len	 & AVG A len & Vcabulary Size
    print("&".join([str(x) for x in [dataset, len_instances, len(passages), avg_a_candidate, avg_q, avg_p, avg_a, len(vocabulary)]]))
    print("Number of questions:", len(questions))


if __name__ == "__main__":
    main()
