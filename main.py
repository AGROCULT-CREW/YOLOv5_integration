import cv2
import uuid
import torch
from flask import Flask, request, make_response
import os

from logic import extract_img, get_prediction

app = Flask(__name__)

# create a python dictionary for your models d = {<key>: <value>, <key>: <value>, ..., <key>: <value>}
dictOfModels = {}
# create a list of keys to use them in the select part of the html code
listOfKeys = []
for r, d, f in os.walk("models_train"):
    for file in f:
        if ".pt" in file:
            # example: file = "model1.pt"
            # the path of each model: os.path.join(r, file)
            dictOfModels[os.path.splitext(file)[0]] = torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.join(r, file), force_reload=True)
            # you would obtain: dictOfModels = {"model1" : model1 , etc}
    for key in dictOfModels :
        listOfKeys.append(key)     # put all the keys in the listOfKeys
    print(listOfKeys)


@app.route('/', methods=['POST'])
def predict():
    print(request)
    file = extract_img(request)
    img_bytes = file.read()
    # choice of the model
    model = dictOfModels['best']
    results = get_prediction(img_bytes, model)
    results.render()
    # encoding the resulting image and return it
    for img in results.imgs:
        RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_arr = cv2.imencode('.jpg', RGB_img)[1]
        
        # img_id = uuid.uuid4()
        # cv2.imwrite(str(img_id) + '.jpg', RGB_img)
        response = make_response(im_arr.tobytes())
        response.headers['Content-Type'] = 'image/jpg'
    # return your image with boxes and labels
    return response


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=7777, debug=True)
