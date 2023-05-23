# -*- coding: utf-8 -*-
# Install the Python Requests library:
# `pip install requests`
import base64
import hashlib
import json
import time

import requests

# base = "http://127.0.0.1:5020"
base = "https://inno.sh-sict.com"


def send_request_single(text):
    try:
        txt = "".join([i[0] for i in text])
        ts = str(int(round(time.time())))
        base64_encode = base64.b64encode((txt + ts).encode("utf-8"))
        md5 = hashlib.md5()
        md5.update(str(base64_encode, "utf-8").encode('utf-8'))
        md5_encode = md5.hexdigest().upper()
        print(str(ts))
        response = requests.post(
            url=base + "/disctn/ner/",
            headers={
                "Content-Type": "application/json;charset=UTF-8",
            },
            data=json.dumps({
                "txt": text,
                "timestamp": str(ts),
                "signiture": md5_encode
            })
        )
        print('Response HTTP Status Code: {status_code}'.format(
            status_code=response.status_code))
        print('Response HTTP Response Body: {content}'.format(
            content=json.dumps(json.loads(response.content.decode('UTF-8')), ensure_ascii=False)))
    except requests.exceptions.RequestException:
        print('HTTP Request failed')


if __name__ == '__main__':
    text = [
        # "主  诉：腹泻1年。现病史：患者于2021年7月无明显诱因出现排便次数增多，少则每日3次，多则8次，粪便不成形或呈水样，无便血黑便等不适。自服“诺氟沙星”等药物效果不佳，遂于2022年06月14日在江阴中医院行结肠镜检查示“结肠肿瘤”，病理提示“腺癌”。未行新辅助放化疗。今为进一步治疗来我院就诊，门诊以&quot;乙状结肠癌；直肠癌术后10年&quot;收入院。",
        # "主  诉：发现肠癌肝转移3年余。现病史：患者于2018年6月因“左下腹痛、大便变稀”起病，于6月27日我院查肠镜示：降结肠距肛缘45cm，有一溃疡型肿块，占肠腔一周，肠腔狭窄。病理示腺癌。肝脏MRI提示：肝内多发转移瘤（&gt;5个，最大3cm）。遂于7月2日、7月16日、7月30日、8月13行FOLFOX方案化疗4次（奥沙利铂200mg VD d1、5-Fu 3.25g civ 46h）。",
        # "主  诉：结肠息肉切除术后2年余，排便习惯改变半月余。现病史：2年余前因便血于我院行肠镜检查示“乙状结肠进镜距肛缘30cm处有1枚2.5*3cm大小息肉，表面充血”，行结肠息肉EMR术，术后病检示“绒毛管状腺瘤，局部腺上皮高级别上皮内瘤变”。",
        "主  诉：结肠癌肝转移化疗后3月余。现病史：患者于2020-12-24晚无明显诱因下出现右下腹部疼痛，伴有呕吐，腹胀，肛门停止排气排便，无呕血，无发热畏寒等，2020-12-26至余姚市人民医院住院，血常规：白细胞15.21*10^9/l，GRAN%：92%。癌胚抗原9.1ug/l，CA125 74u/ml。",
        "主  诉：胃癌术后5年余，发现吻合口复发4月余现病史：患者2017-7-20在我院行远端胃大部切除+残胃-空肠uncut roux-en-y吻合术，术后恢复可。近2年进食后出现上腹部饱胀不适，无腹痛腹泻、黑便，无寒战发热，无恶心呕吐，无便血，此次复查胃镜示：（2022-9-23 长海医院 05351178）吻合口溃疡（取病理）；内镜下胃石切碎术。取活检病理诊断：（2022-9-28 长海医院 2227555）吻合口低分化腺癌，部分印戒细胞癌。行PET-CT示：（2022-9-30 上海州信医学中心 2209300001）胃印戒细胞癌术后，吻合口增厚，FDG代谢增高，考虑复发；左上腹腹膜稍增厚，FDG轻度摄取，建议密切随访除外转移，食管裂孔疝。门诊以&quot;胃癌术后5年余，发现吻合口复发3周&quot;收住入院，拟行手术治疗。自发病以来，病人精神状态良好，体力情况良好，食欲食量良好，睡眠情况良好，体重无明显变化，大便正常，小便正常。"
    ]
    send_request_single(text)
