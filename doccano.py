import argparse
import copy
import json
import os
from random import shuffle
from decimal import Decimal

from tqdm import tqdm


def precess_doccano(input_path, output_directory, splits, maxlen):
    def single_entity(entity, relations):
        for i in relations:
            if i['from_id'] == entity['id'] or i['to_id'] == entity['id']:
                return False
        return True

    def find_ids(id: int, entities: list) -> list:
        for i in entities:
            if i['id'] == id:
                return list(range(i['start_offset'], i['end_offset']))
        raise Exception("Relation id not found in entities.")

    with open(input_path, "r", encoding="utf-8") as f:
        r = []
        for i in f.readlines():
            d = json.loads(i)
            d_new = {"sentence": list(d["text"][:maxlen]), "ner": []}
            relations_copied = copy.deepcopy(d["relations"])
            for j in d["entities"]:
                if single_entity(j, d["relations"]):
                    relations_copied.append({
                        "from_id": j['id'],
                        "to_id": j['id'],
                        "type": j['label']
                    })
            for j in relations_copied:
                if j['from_id'] == j['to_id']:
                    ids = find_ids(j['from_id'], d["entities"])
                    exceed_maxlen = False
                    for i in ids:
                        if i >= maxlen:
                            exceed_maxlen = True
                    if not exceed_maxlen:
                        d_new["ner"].append({
                            "index": ids,
                            "type": j['type'],
                        })
                else:
                    ids = find_ids(j['from_id'], d["entities"]) + find_ids(j['to_id'], d["entities"])
                    exceed_maxlen = False
                    for i in ids:
                        if i >= maxlen:
                            exceed_maxlen = True
                    if not exceed_maxlen:
                        d_new["ner"].append({
                            "index": ids,
                            "type": j['type']
                        })
            print(str(d['id']) + " ", end="")
            for j in d_new["ner"]:
                for k in j['index']:
                    print(d["text"][k], end="")
                print(" ", end="")
            print()
            # print(json.dumps(d_new,ensure_ascii = False))
            # break
            r.append(d_new)
        shuffle(r)
        train = r[:int(len(r) * splits[0])]
        dev = r[int(len(r) * splits[0]):int(len(r) * (splits[0] + splits[1]))]
        test = r[int(len(r) * (splits[0] + splits[1])):]
        print("all: {}, train: {}, dev: {}, test: {}".format(len(r), len(train), len(dev), len(test)))
        with open(os.path.join(output_directory, "all.json"), "w", encoding="utf-8") as f1:
            json.dump(r, f1, ensure_ascii=False, indent=4)
        with open(os.path.join(output_directory, "train.json"), "w", encoding="utf-8") as f2:
            json.dump(train, f2, ensure_ascii=False, indent=4)
        with open(os.path.join(output_directory, "dev.json"), "w", encoding="utf-8") as f3:
            json.dump(dev, f3, ensure_ascii=False, indent=4)
        with open(os.path.join(output_directory, "test.json"), "w", encoding="utf-8") as f4:
            json.dump(test, f4, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", default="input.jsonl", type=str,
                        help="Doccano relation output file path(jsonl).")
    parser.add_argument("-o", "--output_directory", default="data/example/", type=str,
                        help="output directory.")
    parser.add_argument("--splits", default=[0.75, 0.25, 0], type=float, nargs="*",
                        help="The ratio of samples in datasets. [0.6, 0.2, 0.2] means 60% samples used for training, 20% for evaluation and 20% for test.")
    parser.add_argument("--maxlen", default=510, type=int,
                        help="The max length of sentence.")
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        raise ValueError("Please input the correct path of doccano dataset.")
    if args.input_path.split(".")[-1] != "jsonl":
        raise ValueError("The Doccano dataset must be type of jsonl(relation type).")
    if not os.path.exists(args.output_directory):
        os.mkdir(args.output_directory)
    if len(args.splits) != 0 and len(args.splits) != 3:
        raise ValueError("Only []/ len(splits)==3 accepted for splits.")
    if args.maxlen < 0 or args.maxlen > 510:
        raise ValueError("The maxlen should be in [0,510].")


    def _check_sum(splits):
        return Decimal(str(splits[0])) + Decimal(str(splits[1])) + Decimal(
            str(splits[2])) == Decimal("1")


    if len(args.splits) == 3 and not _check_sum(args.splits):
        raise ValueError(
            "Please set correct splits, sum of elements in splits should be equal to 1."
        )

    precess_doccano(args.input_path, args.output_directory, args.splits, args.maxlen)
