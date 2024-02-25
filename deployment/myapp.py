#flask --app myapp --debug run
import json
import requests
import cv2
import base64
from io import BytesIO
import numpy as np
from tensorflow.keras.preprocessing import image as image_utils
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap

# Configure the required models here and the docker ports where the Tensorflow serving is running.
#  http://localhost:8501/v1/models/model:predict
mymodel_config = [
        {"name" : "unet", "port":8501},
        {"name" : "vgg16", "port":8501},
        {"name" : "resnet", "port":8501},
    ]

def prediction_handler(orig_image_resized, y_pred, model_cfg):

    # TO write to file only
    matplotlib.use('agg')
    y_pred = np.argmax(y_pred, axis=-1) + 1
    y_pred = y_pred.squeeze()

   # Define your custom colors for each label
    colors = ['cyan', 'yellow', 'magenta', 'green', 'blue', 'black', 'white']
    # Create a ListedColormap
    cmap = ListedColormap(colors)

    # Create a figure
    fig, ax = plt.subplots()
    # Display the predictions using the specified colormap
    cax = ax.imshow(y_pred, cmap=cmap, vmin=1, vmax=7, alpha=0.5)
    
    # Create colorbar and set ticks and ticklabels
    cbar = plt.colorbar(cax, ticks=np.arange(1, 8))
    cbar.set_ticklabels(['Urban', 'Agriculture', 'Range Land', 'Forest', 'Water', 'Barren', 'Unknown'])
    
    model_name = model_cfg["name"] + '.png'
    genfilename = os.path.join(app.config['GEN_FOLDER'], model_name)    

    plt.savefig(genfilename) 
    plt.close(fig)
    
    return model_name

def prediction_handler_1(orig_image_resized, y_pred, model_cfg):
    print("YPRED - before", y_pred)

    y_pred = np.argmax(y_pred, axis=-1) + 1
    y_pred = y_pred.squeeze()

    print("YPRED - after argmax", y_pred)
    # Define your custom colors for each label
    colors = ['cyan', 'yellow', 'magenta', 'green', 'blue', 'black', 'white']
    # Create a ListedColormap
    cmap = ListedColormap(colors)


    norm = plt.Normalize(vmin=y_pred.min(), vmax=y_pred.max())
    y_pred = y_pred.reshape(-1)
    y_pred = cmap(norm(y_pred))

    fig = Figure()
    ax = fig.subplots()
    ax.plot(y_pred)
    buf = BytesIO()
    fig.savefig(buf, format="png")
    #fig_result_img = base64.b64encode(buf.getbuffer()).decode("ascii")
    fig_result_img = base64.b64encode(buf.getbuffer())
    fig_result_img = f"<img src='data:image/png;base64,{fig_result_img}'/>"


    return fig_result_img

def load_and_scale_image(image_path):
    image = image_utils.load_img(image_path, color_mode="rgb", target_size=(224,224))
    return image

def model_handler(img_file, model_config):
    print("img_file : ", img_file)

    image = load_and_scale_image(img_file)
    image = image_utils.img_to_array(image)
    image = image.reshape(1,224,224,3) 
    image = image / 255

    data = json.dumps({ 
        "instances": image.tolist()
    })
    modelname = str(model_config["name"])
    port = str(model_config["port"])
    headers = {"content-type": "application/json"}
    url = 'http://localhost:'+port+'/v1/models/'+modelname+':predict'

    print("URL is :", url)
    response = requests.post(url, data=data, headers=headers)
    print("Response: ", response)
    dict_response = json.loads(response.content)
    
    return (image,dict_response["predictions"])


#flask --app myapp --debug run
from flask import Flask, request, render_template,session

import os
from flask import Flask, flash, request, redirect, url_for


ALLOWED_EXTENSIONS = { 'jpg', 'jpeg'}
app = Flask(__name__)
app.secret_key = 'dataguru'

app.config['UPLOAD_FOLDER'] = './uploads'
app.config['GEN_FOLDER'] = './gen'


# -----------
# THe files that are allowed for upload
# -----------
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# -----------
# The root path handler
# -----------
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    result_images = []
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            #filename = secure_filename(file.filename)
            filename = file.filename
            fullfilename = os.path.join(app.config['UPLOAD_FOLDER'], filename)    
            file.save(fullfilename)
            session['img_upload_name'] = filename          
            return redirect(url_for('upload_file', name=filename))

    # Get the filename and  if there are valid filenames, then proceed
    uploaded_img_name = session.get('img_upload_name', None)
    if uploaded_img_name:
        fullfilename = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_img_name) 
        myresults = []
        # Go through each available models 
        for modelcfg in mymodel_config:   
            orig_image_resized,result = model_handler(fullfilename, modelcfg)
            genfile_name = prediction_handler(orig_image_resized, result, modelcfg)
            myresults.append( {"model": modelcfg["name"], "filename": genfile_name, "result" : result}  )
        return render_template('result.html', user_image=uploaded_img_name, prediction_text='Some Prediction', results = myresults)
    else:
        return render_template('result.html', user_image="blank.png", prediction_text='No Prediction')
        
## -----------
# The handler for the uploaded file    
# -----------    
@app.route('/uploads/<filename>')
def send_uploaded_file(filename=''):
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)

## -----------
# The handler the geenrated file   
# -----------  
@app.route('/gen/<filename>')
def send_generated_file(filename=''):
    from flask import send_from_directory
    return send_from_directory(app.config['GEN_FOLDER'],filename)


if __name__ == "__main__":
    app.config['SESSION_TYPE'] = 'filesystem'
    session.init_app(app)    
    app.run(debug=True)