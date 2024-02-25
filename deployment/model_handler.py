
import json
import requests
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image as image_utils
import matplotlib.pyplot as plt

def load_and_scale_image(image_path):
    image = image_utils.load_img(image_path, color_mode="rgb", target_size=(224,224))
    return image

def predict(image):
    sample_image_resized = resize_image(image, input_shape)
    y_pred = ensemble_predict(sample_image_resized)
    y_pred = get_predictions(y_pred).squeeze()

    # Create a figure without saving it to a file
    fig, ax = plt.subplots()
    cax = ax.imshow(y_pred, cmap='viridis', vmin=1, vmax=7)

    # Convert the figure to a PIL Image
    image_buffer = io.BytesIO()
    plt.savefig(image_buffer, format='png')
    image_buffer.seek(0)
    image_pil = Image.open(image_buffer)

    # Close the figure to release resources
    plt.close(fig)

    return image_pil

def convert(json_response):
    # Extract text from JSON
    response = json.loads(json_response.text)

    # Interpret bitstring output
    response_string = response["predictions"][0]["b"]
    print("Base64 encoded string: " + response_string[:10] + " ... " + response_string[-10:])

    # Decode bitstring
    encoded_response_string = response_string.encode("utf-8")
    response_image = base64.b64decode(encoded_response_string)
    print("Raw bitstring: " + str(response_image[:10]) + " ... " + str(response_image[-10:]))

    # Save inferred image
    with open("images/sinusoidal.png", "wb") as output_file:
        output_file.write(response_image)
    
def main():
    image = load_and_scale_image('./images/988205_sat.jpg')
    plt.imshow(image, cmap='gray')
    image = image_utils.img_to_array(image)
    image = image.reshape(1,224,224,3) 
    image = image / 255
    print(image.shape)

    data = json.dumps({ 
        "instances": image.tolist()
    })

    headers = {"content-type": "application/json"}
    response = requests.post('http://localhost:8501/v1/models/model:predict', data=data, headers=headers)
    with open("response.log", mode='wb') as localfile:     
        localfile.write(response.content) 
    # prediction = response.json()['predictions'][0]

    #prediction = response.json()["predictions"][0]["b64"]
    #png_str = base64.b64decode(prediction)

    #result = int(response.json()['predictions'][0][0])
    #response_json = response.json()
    #print(response_json["predictions"][0])
    #print("Mn Result :", response.content)
    #convert(response)
    # Save inferred image
    #with open("images/mn.png", "wb") as output_file:
    #    output_file.write(png_str)
    


if __name__ == "__main__":
  main()