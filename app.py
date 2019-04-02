# -*- coding: utf-8 -*-

import os
import io
import time
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from werkzeug import secure_filename
from PIL import Image
app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'PNG', 'JPG', 'jpeg', 'JPEG'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def overlay(background_image, overlay_image, point):
        # OpenCV形式の画像をPIL形式に変換(α値含む)

        # 背景画像
        rgb_background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
        pil_rgb_background_image = Image.fromarray(rgb_background_image)
        pil_rgba_background_image = pil_rgb_background_image.convert('RGBA')
        # オーバーレイ画像
        cv_rgb_overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGRA2RGBA)
        pil_rgb_overlay_image = Image.fromarray(cv_rgb_overlay_image)
        pil_rgba_overlay_image = pil_rgb_overlay_image.convert('RGBA')

        # composite()は同サイズ画像同士が必須のため、合成用画像を用意
        pil_rgba_bg_temp = Image.new('RGBA', pil_rgba_background_image.size,
                                    (255, 255, 255, 0))
        # 座標を指定し重ね合わせる
        pil_rgba_bg_temp.paste(pil_rgba_overlay_image, point, pil_rgba_overlay_image)
        result_image = Image.alpha_composite(pil_rgba_background_image, pil_rgba_bg_temp)

        # OpenCV形式画像へ変換
        cv_result_image = cv2.cvtColor(np.asarray(result_image), cv2.COLOR_RGBA2BGRA)

        return cv_result_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send', methods=['GET', 'POST'])
def send():
    if request.method == 'POST':
        original_image = request.files['image_file']

        # 許可していないファイルを弾く
        if original_image and allowed_file(original_image.filename):
            filename = secure_filename(original_image.filename)
        else:
            return ''' <p>取り扱いのできないファイル形式です</p> '''

        # BytesIOで読み込んでOpenCVで扱える型にする
        f = original_image.stream.read()
        bin_data = io.BytesIO(f)
        file_bytes = np.asarray(bytearray(bin_data.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # 元となる画像を保存する
        original_image_url = os.path.join(app.config['UPLOAD_FOLDER'], 'original_'+filename)
        cv2.imwrite(original_image_url, image)

        # OpenCV標準のカスケードファイルを利用して顔認識
        cascade_path = "/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        facerect = cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))

        # 顔認識で顔が見つかったら、笑い男を被せる
        if len(facerect) > 0:
            smile = cv2.imread('smile.png', cv2.IMREAD_UNCHANGED)

            for x, y, w, h in facerect:
                size = (w, h)
                point = (x, y)
                resize_smile = cv2.resize(smile, size)

                image = overlay(image, resize_smile, point)

        # 笑い男を被せた画像を保存する
        smile_image_url = os.path.join(app.config['UPLOAD_FOLDER'], 'smile_'+filename)
        cv2.imwrite(smile_image_url, image)

        return render_template('index.html', original_image_url=original_image_url, smile_image_url=smile_image_url)

    else:
        return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.debug = True
    app.run()
