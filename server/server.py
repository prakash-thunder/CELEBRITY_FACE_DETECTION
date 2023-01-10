from flask import Flask,request,render_template,flash
import numpy as np
import cv2
import pywt
import pickle
from werkzeug.utils import secure_filename
def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)   
    imArray /= 255;
    # compute coefficients 
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0;  

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)

    return imArray_H
def test_image_fun(img1):
    face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    img=cv2.imread(str(img1))
    # pred_img=cv2.resize(face_img,(32,32))
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        face_img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
        roi_color=face_img[y:y+h,x:x+w]
        img_har = w2d(roi_color,'db1',5)
    scalled_img_har = cv2.resize(img_har, (32, 32))
    combined_img = np.vstack((roi_color.reshape(32*32*3,1),scalled_img_har.reshape(32*32,1)))
    test_img=combined_img.T
    return test_img
model = pickle.load(open(r"C:\Users\praka\Visual Studio CODE\CELEBRITIES_FACE_DETECTION\saved_model.pickle", 'rb'))
app=Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')
@app.route("/predict",methods=['POST'])
def predict():
    if request.method=="POST":
        f=request.files['img']
        f.save(secure_filename(f.filename))
        a=print(type(f.filename))
        Actual=a.split('.')[0]
        img1=cv2.imread(f.filename)
        face_cascade=cv2.CascadeClassifier(r'C:\Users\praka\Visual Studio CODE\CELEBRITIES_FACE_DETECTION\haarcascade_frontalface_default.xml')
        # pred_img=cv2.resize(face_img,(32,32))
        gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:
            face_img=cv2.rectangle(img1,(x,y),(x+w,y+h),(0,0,255),3)
            roi_color=face_img[y:y+h,x:x+w]
        img_har = w2d(roi_color,'db1',5)
        roi_color=cv2.resize(roi_color,(32,32))
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((roi_color.reshape(32*32*3,1),scalled_img_har.reshape(32*32,1)))
        test_img=combined_img.T
        # # test=test_image_fun(img1)
        # cv2.imshow('s',roi_color)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        celebritiess_labels_dict=['none','mira_bai_chanu', 'messi', 'virat', 'pradeep_narwal']
        pred=model.predict(test_img)[0]
        print(pred)
        pred_face=celebritiess_labels_dict[pred]
        print(pred_face)
        return render_template('index.html',data1=pred_face,data2=Actual)
    else:
        'something went wrong'
if __name__=="__main__":
    app.run(debug=True)