import argparse
import base64
import hashlib
import json
import time
from datetime import datetime

from flask import request, jsonify, Blueprint

from config import Config
from predict import predict
from flask import Flask

app = Flask(__name__)
app.secret_key = '1234567'

api = Blueprint('changhai', __name__)


@api.route('/')
def show():
    return 'This is changhai api.'


@api.route('/ner/', methods=['POST'], strict_slashes=False)
def recognize():
    if request.method == 'POST':
        # print(request.date)
        data = json.loads(request.get_data())
        txt = data['txt']
        timestamp = data['timestamp']
        sgt = data['signiture']
        if txt is None or txt is []:
            return jsonify({'success': False, 'description': {'error msg': 'invalid post body fields'}}), 500
        elif type(txt) == list:
            valid, msg = signiture(txt, timestamp, sgt)
            # print(valid, msg)
            if not valid:
                return jsonify({'success': False, 'description': {'error msg': msg}}), 500
            else:
                try:
                    result = processt_text(txt)
                    result = {'success': True,
                              'data': result,
                              'description': "success"}
                    return jsonify(result), 200

                except Exception as e:
                    print("/n******ERROR SRART******/n")
                    print(e)
                    print("----------txt----------")
                    print(txt)
                    print("/n*******ERROR END*******/n")
                    return jsonify({'success': False, 'description': {'error msg': str(e)}}), 500
        else:
            return jsonify({'success': False, 'description': {'error msg': "invalid type"}}), 500
    else:
        return jsonify({'success': False, 'description': {'error msg': 'Invalid methods'}}), 404


def signiture(txt, timestamp, signiture):
    ts1 = int(timestamp)
    ts2 = int(round(time.time()))
    dt1 = datetime.utcfromtimestamp(ts1)
    dt2 = datetime.utcfromtimestamp(ts2)
    # print(dt1)
    # print(dt2)
    # print((dt2 - dt1).total_seconds())
    if float((dt2 - dt1).total_seconds()) < 120.0:
        base64_encode = base64.b64encode(("".join([i[0] for i in txt]) + timestamp).encode("utf-8"))
        md5 = hashlib.md5()
        md5.update(str(base64_encode, 'UTF-8').encode('utf-8'))
        md5_encode = md5.hexdigest().upper()
        # print(md5_encode)
        # print(signiture)
        if signiture.upper() == md5_encode:
            return True, ""
        else:
            return False, "invalid signiture"
    else:
        return False, "Timestamp exceed"


def processt_text(txts):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/changhai.json')
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
    config = Config(args)
    result = json.loads(predict(txts,
                                lambda text: list(text),
                                config,
                                batch_size=255))

    for i, item in enumerate(result):
        entities = []
        for j, _ in enumerate(result[i]["entities"]):
            e = {"text": ''.join(result[i]["entities"][j][0]),
                 "category": result[i]["entities"][j][1]}
            if e not in entities:
                entities.append(e)
        result[i]["entities"] = entities
        result[i]["text"] = ''.join(result[i]["text"])
    return result


app.register_blueprint(api, url_prefix='/disctn')

if __name__ == '__main__':
    # from werkzeug.contrib.fixers import ProxyFix
    # app.wsgi_app = ProxyFix(app.wsgi_app)
    app.run(debug=True, port=5020, host='0.0.0.0')
