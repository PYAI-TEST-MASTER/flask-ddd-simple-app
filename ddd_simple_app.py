import os
import cv2
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from time import sleep

image_size = 28

UPLOAD_FOLDER = "uploads"
INTERMEDIATE_FOLODER = "intermediates"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


### model_vgg16 = load_model('./modelvgg16_64_18_adam.h5')#学習済みモデルをロード
model_vgg16_normal = load_model('./modelvgg16_64_18_adam_N.h5')#学習済みモデルをロード
#カスケード型分類器に使用する分類器のデータ（xmlファイル）を読み込み
HAAR_FILE = R"./haarcascade_eye_tree_eyeglasses.xml"
cascade = cv2.CascadeClassifier(HAAR_FILE)

@app.route('/', methods=['GET', 'POST'])

def upload_file():
    vgg16_count = 0
    vgg16_normal_count = 0
    vgg16_status = ''
    vgg16_normal_status = ''
    vgg16_drowsiness_level = ''
    vgg16_normal_drowsiness_level = ''
    history_on = 0
    if request.method == 'POST':
        if 'files' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['files']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        files = request.files.getlist('files')  # 'files'はinputタグのname属性
        for file in files:
            print("Start")
            def get_drowsiness_level(close_count ,totalcount):
                persent = close_count / totalcount * 100
                if persent >= 98:
                    return '居眠り'
                if persent >= 75:
                    return 'かなり眠たい'
                if persent >= 25:
                    return '眠たい'
                if persent >= 5:
                    return 'やや眠たい'
                return '覚醒'            
            # ここでファイルを処理する
            # 例: ファイルを保存する、画像解析を行うなど
            print(file)
            print(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            face = cv2.imread(filepath,0)
            os.remove(filepath)
            eye = cascade.detectMultiScale(face)
 
            #顔の座標を表示する

            if eye is None:
                print('CLOSE1_EYE')
                skip_flag = 1
            try:
                x,y,w,h = eye[0]
                skip_flag = 0
            except IndexError:
                print('CLOSE2_EYE')
                skip_flag = 1
            if skip_flag == 0:      
                # if eye[0] is None:
                #     print(eye[1])
                #     x,y,w,h = eye[1]
                # else:
                #     print(eye[0])
                #     x,y,w,h = eye[0]
                #顔部分を切り取る

                eye_cut = face[y-h//3:y+h*10//8, x-w//3:x+w*10//8]
    #            eye_cut = img[y:y+h, x:x+w]
    #            eye_cut = img[y-w//2:y+w//2, x:x+w]
                #白枠で顔を囲む
    #            x,y,w,h = eye[0]
                cv2.rectangle(face,(x-w//2,y-h//2),(x+w*10//8,y+h*10//8),(255,255,255),2)
    
                #cv2.rectangle(img,(x,y-w//2),(x+w,y+w//2),(255,255,255),2)
    
                #画像の出力
                filepath = "eye_"+file.filename
                filepath = os.path.join(INTERMEDIATE_FOLODER, filepath)
                if history_on == 1:
                    cv2.imwrite(filepath, eye_cut)
                filepath = 'face_'+file.filename
                filepath = os.path.join(INTERMEDIATE_FOLODER, filepath)
                if history_on == 1:
                    cv2.imwrite(filepath, face)
                #ヒストグラム平坦化
                eye_cut_hist = cv2.equalizeHist(eye_cut)
###1                cv2.imwrite(filepath, eye_cut_hist)        
                img_rgb = cv2.cvtColor(eye_cut_hist, cv2.COLOR_BGR2RGB)   
    #            img_rgb = cv2.cvtColor(eye_cut, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img_rgb, (82,82))
                img = img.astype('float32') / 255
                img = np.expand_dims(img, axis=0)
    #
    #           VGG16
    #
                # pred = model_vgg16.predict(img)
                # if np.argmax(pred) == 0:
                #     result_vgg16 = 'OPEN_EYE'    
                # else:
                #     result_vgg16 = 'CLOSE_EYE'
                # filepath = 'face_vgg16'+result_vgg16+file.filename
                # filepath = os.path.join(INTERMEDIATE_FOLODER, filepath)
                # cv2.imwrite(filepath, face)
                # pred_answer = 'vgg16:'+file.filename+ "は、" + result_vgg16
                # print(pred_answer)
                # sleep(1)
    #
    #           VGG16 NORMALIZATION
    #
                pred = model_vgg16_normal.predict(img)
                if np.argmax(pred) == 0:
                    result_vgg16_normal = 'OPEN_EYE'    
                else:
                    result_vgg16_normal = 'CLOSE_EYE'
                filepath = 'face_vgg16_normal'+result_vgg16_normal+file.filename
                filepath = os.path.join(INTERMEDIATE_FOLODER, filepath)
#                cv2.imwrite(filepath, face)
                pred_answer = 'vgg16_normal:'+file.filename+ "は、" + result_vgg16_normal
                print(pred_answer)
                sleep(1)
    # #           vgg16
    #             if result_vgg16 == "CLOSE_EYE":
    #                 vgg16_count = vgg16_count +1
    #                 vgg16_status += 'C'
    #             else:
    #                 vgg16_status += 'O'
    #             vgg16_drowsiness_level = get_drowsiness_level(vgg16_count,len(files))
    #           vgg16 normalization
                if result_vgg16_normal == "CLOSE_EYE":
                    vgg16_normal_count = vgg16_normal_count + 1
                    vgg16_normal_status += 'C'
                else:
                    vgg16_normal_status += 'O'
                vgg16_normal_drowsiness_level = get_drowsiness_level(vgg16_normal_count,len(files))
            else:
                # vgg16_count = vgg16_count +1
                # vgg16_status += 'C'
                vgg16_normal_count = vgg16_normal_count + 1
                vgg16_normal_status += 'C'
                # vgg16_drowsiness_level = get_drowsiness_level(vgg16_count,len(files))
                vgg16_normal_drowsiness_level = get_drowsiness_level(vgg16_normal_count,len(files))
            # VGG16_RESULT =                  "転移学習VGG16(非正規)：クローズ数／枚数　"+ str(vgg16_count)+"/"+ str(len(files))+"   " + vgg16_drowsiness_level +vgg16_status[:20]
            VGG16_NORMAL_RESULT =           "■転移学習VGG16(正規化)  ：クローズ数／枚数　"+ str(vgg16_normal_count)+"/"+ str(len(files))+"   ★判定結果："+ vgg16_normal_drowsiness_level+"　　下記は各画像の判定結果(O:OPEN_EYE/C:CLOSE_EYE)"

#        return render_template("index.html",answer=VGG16_RESULT,answer2 = VGG16_NORMAL_RESULT) 
        return render_template("index.html",answer = VGG16_NORMAL_RESULT ,answer2 = vgg16_normal_status) 

    return render_template("index.html",answer="")
# if __name__ == "__main__":
#      app.run()
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)
