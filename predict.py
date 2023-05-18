import argparse
import json
import re

import torch
import torch.autograd
from torch.utils.data import DataLoader

import config
import data_loader
import utils
from config import Config
from model import Model


class Trainer(object):
    def __init__(self, model, device, config):
        self.model = model
        self.model = self.model.to(device)
        self.device = device
        self.config = config

    def predict(self, predict_loader):
        self.model.eval()
        with torch.no_grad():
            for i, data_batch in enumerate(predict_loader):
                texts = data_batch[-1]
                data_batch = [data.to(self.device) for data in data_batch[:-1]]
                bert_inputs, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch
                outputs = self.model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
                outputs = torch.argmax(outputs, -1)
                # print(outputs)
                entities = predict_decode(outputs.cpu().numpy(), sent_length.cpu().numpy(), texts, self.config)
                return entities

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))


def predict_decode(outputs, length, texts, config):
    entities = []
    for index, (instance, l, text) in enumerate(zip(outputs, length, texts)):
        forward_dict = {}
        head_dict = {}
        ht_type_dict = {}
        for i in range(l):
            for j in range(i + 1, l):
                if instance[i, j] == 1:
                    if i not in forward_dict:
                        forward_dict[i] = [j]
                    else:
                        forward_dict[i].append(j)
        for i in range(l):
            for j in range(i, l):
                if instance[j, i] > 1:
                    ht_type_dict[(i, j)] = instance[j, i]
                    if i not in head_dict:
                        head_dict[i] = {j}
                    else:
                        head_dict[i].add(j)

        predicts = []

        def find_entity(key, entity, tails):
            entity.append(key)
            if key not in forward_dict:
                if key in tails:
                    predicts.append(entity.copy())
                entity.pop()
                return
            else:
                if key in tails:
                    predicts.append(entity.copy())
            for k in forward_dict[key]:
                find_entity(k, entity, tails)
            entity.pop()

        def convert_index_to_text(index, type):
            text = "-".join([str(i) for i in index])
            text = text + "-#-{}".format(type)
            return text

        for head in head_dict:
            find_entity(head, [], head_dict[head])
        predicts = set([convert_index_to_text(x, ht_type_dict[(x[0], x[-1])]) for x in predicts])
        tmp = (text,)
        for pre in predicts:
            pre = pre.split('-#-')
            # print(pre)
            # print(text)
            ind = pre[0].split('-')
            e = []
            for i in ind:
                e.append(text[int(i)])
            entity = e
            # print(config.vocab)
            entity_type = config.vocab.id2label[int(pre[1])]
            tmp += ((entity, entity_type, [int(i) for i in ind]),)
        entities.append(tmp)
    return entities


def predict(texts, tokenize_method, config, batch_size):
    texts = [tokenize_method(i.replace(" ", "")) for i in texts]

    # 这一步要在model之前创建，因为还有给config添加属性
    predict_dataset = data_loader.load_data_bert_predict(texts, config)
    predict_loader = DataLoader(dataset=predict_dataset,
                                batch_size=batch_size,
                                collate_fn=data_loader.collate_fn_predict,
                                shuffle=False,
                                num_workers=4,
                                drop_last=False)
    print("Building Model...")
    model = Model(config)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    trainer = Trainer(model, device, config)

    trainer.load(config.save_path)

    result = trainer.predict(predict_loader)
    return json.dumps([{"text": item[0], "entities": item[1:]} for item in result], ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/conll03.json')
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--dist_emb_size', type=int)
    parser.add_argument('--type_emb_size', type=int)
    parser.add_argument('--lstm_hid_size', type=int)
    parser.add_argument('--conv_hid_size', type=int)
    parser.add_argument('--bert_hid_size', type=int)
    parser.add_argument('--ffnn_hid_size', type=int)
    parser.add_argument('--biaffine_size', type=int)

    parser.add_argument('--dilation', type=str, help="e.g. 1,2,3")

    parser.add_argument('--emb_dropout', type=float)
    parser.add_argument('--conv_dropout', type=float)
    parser.add_argument('--out_dropout', type=float)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)

    parser.add_argument('--clip_grad_norm', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)

    parser.add_argument('--bert_name', type=str)
    parser.add_argument('--bert_learning_rate', type=float)
    parser.add_argument('--warm_factor', type=float)

    parser.add_argument('--use_bert_last_4_layers', type=int, help="1: true, 0: false")

    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    config = config.Config(args)

    logger = utils.get_logger(config.dataset)
    logger.info(config)
    config.logger = logger

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)


    def tokenize_word_en(text):
        # 将句号替换为空格+句号，这样句号就可以被当做单独的词处理
        text = re.sub(r'\.', ' .', text)
        # 将连续的空格替换为单个空格
        text = re.sub(r'\s+', ' ', text)
        # 分词
        tokens = text.split()
        return tokens


    def tokenize_char_cn(text):
        return list(text)


    texts = [
        # "高勇，男，中国国籍，无境外居留权。",
        # "现任浙江康恩贝健康产品有限公司总经理。",
        # "I am having horrible aching in my fingers , knees , hips & toes .",
        # "Muscle Pain in left arm and upper back."
        "现病史：患者于1月前无意中出现左下腹不适，无腹泻、黑便，无头晕、头痛、无恶心、呕吐等不适。遂就诊于新华医院素明分院，行肠镜检查示直肠息肉，横结肠瘤。病理示横结肠”符合上皮内瘤变，高级别，瘤变（腺瘤），今为进步手术治疗来我院就诊，门诊以“结肠瘤“收入院。自发病以来，病人精神状态良好，体力情况良好，食欲食量良好睡眠情况良好，体重无明显变化，大便正常，小便正常。",
        "现病史 : 患者 4月前 无明显诱 出现间断 腹痛、大便次数增多、可耐受。1天前再次腹痛、不能忍受、呈 阵发性加剧疼病无放射 、伴 频繁恶心、呕吐 、呕吐后腹病不减轻伴停止排便排气。 就诊于 当地医院 ，行保守治疗，小效果不佳。",
        "现病史：患者因大便带血、排便习惯及大便性状改变于2021-08-12至我院行结肠镜检查示：直肠癌？大肠多发息肉。",
        "主诉：腹泻1年。现病史：患者于2021年7月无明显诱因出现排便次数增多，少则每日3次，多则8次，粪便不成形或呈水样，无便血黑便等不适。"
    ]
    print(predict(texts,
                  # tokenize_word_en,
                  tokenize_char_cn,
                  config,
                  batch_size=255))
