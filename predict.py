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
        "主  诉：腹泻1年。现病史：患者于2021年7月无明显诱因出现排便次数增多，少则每日3次，多则8次，粪便不成形或呈水样，无便血黑便等不适。自服“诺氟沙星”等药物效果不佳，遂于2022年06月14日在江阴中医院行结肠镜检查示“结肠肿瘤”，病理提示“腺癌”。未行新辅助放化疗。今为进一步治疗来我院就诊，门诊以&quot;乙状结肠癌；直肠癌术后10年&quot;收入院。自发病以来，病人精神状态良好，体力情况良好，食欲食量良好，睡眠情况良好，体重在患病期间减少10kg，小便正常。",
        "主  诉：发现肠癌肝转移3年余。现病史：患者于2018年6月因“左下腹痛、大便变稀”起病，于6月27日我院查肠镜示：降结肠距肛缘45cm，有一溃疡型肿块，占肠腔一周，肠腔狭窄。病理示腺癌。肝脏MRI提示：肝内多发转移瘤（&gt;5个，最大3cm）。遂于7月2日、7月16日、7月30日、8月13行FOLFOX方案化疗4次（奥沙利铂200mg VD d1、5-Fu 3.25g civ 46h）。",
        "主  诉：结肠息肉切除术后2年余，排便习惯改变半月余。现病史：2年余前因便血于我院行肠镜检查示“乙状结肠进镜距肛缘30cm处有1枚2.5*3cm大小息肉，表面充血”，行结肠息肉EMR术，术后病检示“绒毛管状腺瘤，局部腺上皮高级别上皮内瘤变”。9月余前于当地医院复查行肠镜提示“横结肠中段、横结肠近脾曲、脾曲距肛门40cm及距肛门28cm结肠多发性息肉”，病检提示“降结肠绒毛状腺瘤，局部低级别上皮内瘤变”。半月余前无明显诱因地出现排便习惯改变，便秘、腹泻交替，伴左下腹疼痛，为阵发性隐痛，不剧烈，可自行缓解，无便血黑便、肛门坠胀、里急后重、恶心呕吐、头晕乏力、肛门停止排气排便等不适，于2021年12月15日在我院医院行电子结肠镜检查示：结肠肿物；结肠多发息肉。肠镜病检结果提示：绒毛状腺瘤伴高级别上皮内瘤变。今为进一步治疗来我院就诊，门诊以&quot;结肠肿物&quot;收入院。自发病以来，病人精神状态良好，体力情况良好，食欲食量良好，睡眠情况良好，体重无明显变化，小便正常。",
        "主  诉：结肠癌肝转移化疗后3月余。现病史：患者于2020-12-24晚无明显诱因下出现右下腹部疼痛，伴有呕吐，腹胀，肛门停止排气排便，无呕血，无发热畏寒等，2020-12-26至余姚市人民医院住院，血常规：白细胞15.21*10^9/l，GRAN%：92%。癌胚抗原9.1ug/l，CA125 74u/ml。头颅CT：老年脑改变，全腹部CT增强：回盲部及阑尾恶性肿瘤伴腹腔、腹膜后淋巴结肿大，肝右叶近膈顶转移瘤首先考虑，直肠乙状结肠壁增厚，水肿，回盲部下方包裹性积液考虑。经禁食、补液、抗感染、纠正电解质紊乱等治疗后好转，2021-1-5我院PET-CT：结肠回盲部区恶性肿瘤，周围淋巴结转移。腹主动脉旁淋巴结转移，肝脏转移瘤。结肠镜：升结肠占位，乙状结肠腔外压迫？病理：（升结肠）腺癌，（乙状结肠）溃疡。排除禁忌后,于2021-1-8起完成4次全身化疗（乐沙定 180mg d1 + 希罗达 1g 1/早 1.5g 1/晚 服用2周休1周），化疗后患者恢复可，此次为求进一步诊治，门诊拟“升结肠癌IV期（肝）”收治入院。"
    ]
    print(predict(texts,
                  # tokenize_word_en,
                  tokenize_char_cn,
                  config,
                  batch_size=255))
