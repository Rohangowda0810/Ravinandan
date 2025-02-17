import os
import requests
from keras.models import load_model
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# URLs of the split model parts on GitHub
part1_url = 'https://github.com/yourusername/yourrepo/raw/main/model_part1'  # Replace with actual URL
part2_url = 'https://github.com/yourusername/yourrepo/raw/main/model_part2'  # Replace with actual URL

# Function to download and merge the split model parts
def download_and_merge_model():
    model_path = 'model.h5'
    
    # Download the model parts from GitHub
    response1 = requests.get(part1_url)
    response2 = requests.get(part2_url)
    
    if response1.status_code == 200 and response2.status_code == 200:
        # Merge the parts
        with open(model_path, 'wb') as f:
            f.write(response1.content)
            f.write(response2.content)
        print("Model parts downloaded and merged into 'model.h5'.")
    else:
        print("Error downloading model parts.")
        return False
    
    return True

# Ensure model is downloaded and merged
if not os.path.exists('model.h5'):
    if not download_and_merge_model():
        raise FileNotFoundError("Unable to download or merge model parts.")

# Load the model
model = load_model('model.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

def getResult(image_path):
    img = load_img(image_path, target_size=(225, 225))
    x = img_to_array(img)
    x = x.astype('float32') / 255.0
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)[0]
    return predictions

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        
        # Ensure the uploads directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        f.save(file_path)
        
        predictions = getResult(file_path)
        predicted_label = labels[np.argmax(predictions)]
        
        return str(predicted_label)
    
    return None

if __name__ == '__main__':
    app.run(debug=True)
