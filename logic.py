import io
from PIL import Image
from werkzeug.exceptions import BadRequest


def get_prediction(img_bytes, model):
    img = Image.open(io.BytesIO(img_bytes))
    img = img.resize((640, 640))
    img = img.convert("L")
    # inference
    results = model(img, size=640)  
    return results


def extract_img(request):
    # checking if image uploaded is valid
    if 'file' not in request.files:
        raise BadRequest("Missing file parameter!")
    file = request.files['file']
    if file.filename == '':
        raise BadRequest("Given file is invalid")
    return file