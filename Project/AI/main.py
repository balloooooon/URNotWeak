from flask import Flask, request,jsonify
from werkzeug.utils import secure_filename
import os

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

@app.route('/AI/upload', methods=['POST','GET'])
def pred():
    if request.method == 'POST':
        remo_credir()
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        return "저장성공"
    if request.method == 'GET':
        return "get!"


if __name__ == '__main__':
    app.run()