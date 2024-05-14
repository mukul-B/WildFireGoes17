from flask import Flask, render_template
import os

app = Flask(__name__,static_folder='reference_data/Dixie/compare')

@app.route('/')
def index():
    # Folder containing images
    image_folder = 'reference_data/Dixie/compare'

    # Get the list of image files in the folder
    image_files = os.listdir(image_folder)

    # Filter only image files (assuming they have common extensions like .jpg, .png, etc.)
    image_files = [file for file in image_files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

    # Render the HTML template and pass the list of image files to it
    return render_template('index.html', image_files=image_files)

if __name__ == '__main__':
    app.run(debug=True)
