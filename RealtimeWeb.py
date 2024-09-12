from flask import Flask, render_template
import os

# Set the root and VIIRS folders
root_folder = 'DataRepository/RealTimeIncoming_results/DavisCreekFire/'
prediction_folder = 'results/'
viirs_folder = 'VIIRS/'

# Specify the static folder where your images are stored
app = Flask(__name__, static_folder=root_folder)

def get_image_files(folder):
    image_files = os.listdir(root_folder+folder)
    image_files = [folder+file for file in image_files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    image_files.sort(reverse=True)
    return list(enumerate(image_files, start=1))

@app.route('/')
def index():
    enumerated_images = get_image_files(prediction_folder)
    # Pass the folder as static_url to handle different image locations
    return render_template('index_realtime.html', title="Davis Creek Fire", enumerated_images=enumerated_images, static_url=root_folder, viirs=False)

@app.route('/viirs')
def viirs():
    enumerated_images = get_image_files(viirs_folder)
    return render_template('index_realtime.html', title="Davis Creek Fire - VIIRS", enumerated_images=enumerated_images, static_url=viirs_folder, viirs=True)

if __name__ == '__main__':
    # app.run(debug=False,host='0.0.0.0', port=8085)
    app.run(debug=True, port=8085) # for local only
