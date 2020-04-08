
import argparse
import re

from parameters import datapaths
from utils import write_vocabulary
from vocab_utils import processors

from spacy.lang.en import English

def main():
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
    vocabulary = processor.get_vocabulary_size(datapath)

    #print()
    print(args.task_name, "VOCAB size:", len(vocabulary))
    write_vocabulary(vocabulary, args.task_name)




if __name__ == "__main__":
    main()
