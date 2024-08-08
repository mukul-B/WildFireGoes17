from flask import Flask, render_template
import os

folder = 'DataRepository/RealTimeIncoming_results//ParkFire/results'
# app = Flask(__name__,static_folder='DataRepository/reference_data/Dixie/compare')
app = Flask(__name__,static_folder=folder)

@app.route('/')
def index():
    # Folder containing images
    # image_folder = 'DataRepository/reference_data/Dixie/compare'
    image_folder = folder

    # Get the list of image files in the folder
    image_files = os.listdir(image_folder)

    # Filter only image files (assuming they have common extensions like .jpg, .png, etc.)
    image_files = [file for file in image_files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    image_files.sort(reverse=True)
    # Render the HTML template and pass the list of image files to it
    enumerated_images = list(enumerate(image_files, start=1))
    return render_template('index_realtime.html', enumerated_images=enumerated_images)

if __name__ == '__main__':
    app.run(debug=True, port=8080)




