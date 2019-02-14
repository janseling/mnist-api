# 加载库
import flask
import tensorflow as tf
import keras
from keras.models import load_model
from PIL import Image

# 实例化 flask 
UPLOAD_FOLDER = __dir__ + 'uploads'
print('upload floder : ' + UPLOAD_FOLDER)
app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 我们需要重新定义我们的度量函数，
# 从而在加载模型时使用它
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    keras.backend.get_session().run(tf.local_variables_initializer())
    return auc

# 加载模型，传入自定义度量函数
global graph
graph = tf.get_default_graph()
model = load_model('model.cnn.h5', custom_objects={'auc': auc})

# 将预测函数定义为一个端点
@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    params = flask.request.json
    if (params == None):
        params = flask.request.args

    # 若发现参数，则返回预测值
    if (params != None):
        x=pd.DataFrame.from_dict(params, orient='index').transpose()
        with graph.as_default():
            data["prediction"] = model.predict(x)
            data["success"] = True

    # 返回Jason格式的响应
    return flask.jsonify(data)    

# 启动Flask应用程序，允许远程连接
app.run(host='0.0.0.0', port='10001')
