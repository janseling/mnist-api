# 加载库
from keras.models import load_model
from PIL import Image
import tensorflow as tf
import numpy as np
import hashlib
import base64
import keras
import flask
import time
import os

# 实例化 flask 
app = flask.Flask(__name__)

# 我们需要重新定义我们的度量函数，
# 从而在加载模型时使用它
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    keras.backend.get_session().run(tf.local_variables_initializer())
    return auc

# 加载模型，传入自定义度量函数
global graph
graph = tf.get_default_graph()
model = load_model('model.mlp.h5', custom_objects={'auc': auc})

# 将预测函数定义为一个端点
@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    params = flask.request.json
    if (params == None):
        params = flask.request.args

    # 若发现参数，则返回预测值
    if (params != None):
        image = base64.b64decode(params.get('img'))
        filename = 'uploads/' + time.strftime("%Y%m%d%H%M%S", time.localtime()) + hashlib.md5(image).hexdigest() + '.jpg'
        with open(filename, 'wb') as f:
            f.write(image)

        image = Image.open(filename)
        image = image.convert('L')
        image = image.resize((28, 28))
        image = np.array(image).astype(np.float32).reshape(784) / 255

        print('image shape ', image.shape)
        with graph.as_default():
            data["prediction"] = model.predict(np.array([image]))[0].tolist()
            data["success"] = True

    # 返回Jason格式的响应
    return flask.jsonify(data)    

# 启动Flask应用程序，允许远程连接
app.run(host='0.0.0.0', port=10001)
