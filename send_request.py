# Install the Python Requests library:
# `pip install requests`
import base64
import hashlib
import json
import time

import requests

base = "http://127.0.0.1:5020"


# base = "https://inno.sh-sict.com"


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
        "现病史：患者于1月前无意中出现左下腹不适，无腹泻、黑便，无头晕、头痛、无恶心、呕吐等不适。遂就诊于新华医院素明分院，行肠镜检查示直肠息肉，横结肠瘤。病理示横结肠”符合上皮内瘤变，高级别，瘤变（腺瘤），今为进步手术治疗来我院就诊，门诊以“结肠瘤“收入院。自发病以来，病人精神状态良好，体力情况良好，食欲食量良好睡眠情况良好，体重无明显变化，大便正常，小便正常。",
        "现病史 : 患者 4月前 无明显诱 出现间断 腹痛、大便次数增多、可耐受。1天前再次腹痛、不能忍受、呈 阵发性加剧疼病无放射 、伴 频繁恶心、呕吐 、呕吐后腹病不减轻伴停止排便排气。 就诊于 当地医院 ，行保守治疗，小效果不佳。",
        "现病史：患者因大便带血、排便习惯及大便性状改变于2021-08-12至我院行结肠镜检查示：直肠癌？大肠多发息肉。",
        "主诉：腹泻1年。现病史：患者于2021年7月无明显诱因出现排便次数增多，少则每日3次，多则8次，粪便不成形或呈水样，无便血黑便等不适。"
    ]
    send_request_single(text)
