import tensorflow as tf
from flask import Flask, request,jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from operator import itemgetter
import os


## Seperate model load##
# sep_root_path = "C:\\flask\\models\\models\\sep\\"
sep_root_path = "models\\"
sep_label_path = sep_root_path + 'labels.txt'

files = open(sep_label_path, "r", encoding="UTF-8")
labels = files.readlines()
sep_labels = []

for label in labels:
    sep_labels.append(label.strip('\n'))
files.close

sep_interpreter = tf.lite.Interpreter(model_path=sep_root_path + "model.tflite")
sep_interpreter.allocate_tensors()


def seperate():
    # Get input and output tensors.
    input_details = sep_interpreter.get_input_details()
    output_details = sep_interpreter.get_output_details()
    datagen = ImageDataGenerator(rescale=1. / 255)
    test_dir = 'uploaded'
    test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        shuffle=False,
        class_mode='categorical',
        batch_size=1)

    input_data = np.array(test_generator[0][0], dtype=np.float32)

    sep_interpreter.set_tensor(input_details[0]['index'], input_data)

    sep_interpreter.invoke()

    output_data = sep_interpreter.get_tensor(output_details[0]['index'])
    # print(*output_data)

    print_data = []
    list_print_data = []

    for index, value in enumerate(*output_data):
        list_print_data.append([index, value])

    print_data = sorted(list_print_data, key=itemgetter(1), reverse=True)

    # print(print_data)

    result = []
    for i in range(len(*output_data)):
        result.append(sep_labels[print_data[i][0]])

    return str(result[0])


def remo_credir():
    try:
        import shutil
        shutil.rmtree('uploaded/image')
        print()
    except:
        pass

    try:
        os.mkdir('uploaded/image')
    except:
        pass

app = Flask(__name__)

app.config['JSON_AS_ASCII'] = False # jsonify에서 한글사용
app.config['UPLOAD_FOLDER'] = 'uploaded\\image' #경로설정

@app.route('/API', methods=['POST','GET'])
def pred():
    if request.method == 'POST':
        remo_credir()
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        val1 = seperate()
        return jsonify({"1st":val1})
    if request.method == 'GET':
        return "get!"


if __name__ == '__main__':
    app.run()