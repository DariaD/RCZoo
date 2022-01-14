import argparse


from rc_zoo_utils import processors
from stanza_processing import process_examples

from parameters import datapaths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--debug",
        default=False,
        type=bool,
        required=False,
        help="If true - processing only one example from dataset and save into debug file",
    )
    parser.add_argument(
        "--light",
        default=True,
        type=bool,
        required=False,
        help="If true - calculate statistics but do not save processed with Stanza files",
    )


    args = parser.parse_args()
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    datapath = datapaths[args.task_name]



    print("-------_start new dayaset ----------------")
    print("Dataset:", args.task_name)

    data = processor.get_all_examples(datapath, args.debug)

    print("\tData collected")

    examples = data["examples"]
    print("\tThe debug is",  args.debug)
    print("The number of examples is:", len(examples))

    process_examples(args.task_name, examples, args.debug, args.light)


    print("##########################")


if __name__ == "__main__":
    main()
