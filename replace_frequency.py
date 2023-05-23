import json
import os

from tqdm import tqdm


def process(input_file, output_file, replace_json):
    def find_substring(s, sub, label):
        start = 0
        indexes = []
        while True:
            index = s.find(sub, start)
            if index == -1:
                break
            indexes.append({
                "label": label,
                "start_offset": index,
                "end_offset": index + len(sub)
            })
            start = index + 1
        return indexes

    if os.path.exists(output_file):
        os.remove(output_file)

    with open(input_file, "r") as f1:
        with open(output_file, 'a', encoding='utf-8') as f2:
            l = [i.strip() for i in f1.readlines()]
            for i in tqdm(l):
                entitie = []
                for type, words in zip(replace_json.keys(), replace_json.values()):
                    for word in words:
                        if find_substring(i, word, type) != []:
                            entitie += find_substring(i, word, type)
                f2.write(json.dumps({
                    "text": i,
                    "entities": entitie
                }, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    process("./data/changhai/data.txt",
            "./data/changhai/data_replaced.jsonl",
            {
                "消瘦": ["消瘦", "无消瘦"],
                "排便闲难": ["无排便困难"],
                "呕吐": ["呕吐", "不伴呕吐"],
                "黑便": ["黑便", "无黑便"],
                "恶心": ["恶心", "不伴恶心"],
                "腹部肿块": ["不伴腹部肿块"],
                "腹胀": ["无明显诱因出现腹痛腹胀", "无腹痛腹胀", "不伴腹痛腹胀", "不伴腹胀"],
                "便血": ["便血", "无明显诱因出现便血", "无便血", "不伴便血", "大便带血", "无明显诱因出现间断便血"],
                "腹痛": ["出现腹痛", "无腹痛", "不伴腹痛"],
                "里急后重": ["里急后重", "无里急后重"],
                "大便习惯和性状改变": ["不伴大便习惯和性状改变", "无大便性状改变", "伴大便习惯和性状改变"],
                "腹泻": ["无腹泻", "伴腹泻", "有腹泻"],
                "大便形状改变": ["大便不成形", "大便变细", "粘液便"],
            })
