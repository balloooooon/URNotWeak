from flask import Flask, request,jsonify
from werkzeug.utils import secure_filename
from flask import send_file
import os
# from models.e4e import predict

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
        # 입력받은 사용자 사진 저장
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        
        # GAN 적용
        
        # 입력받은 사용자 사진 삭제
        
        # 결과 이미지 반환
        image_path = 'result\\' + 'apple.jpg'
        # predict()
        return send_file(image_path, mimetype='image/jpeg')
    if request.method == 'GET':
        return "get!"


if __name__ == '__main__':
    app.run()