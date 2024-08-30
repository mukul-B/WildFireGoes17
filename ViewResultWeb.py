from flask import Flask, render_template, request, url_for
import os

# Define the root folder where your subfolders and images are located
root_folder = 'DataRepository/reference_data_everything_closeDate/compare'

# Initialize the Flask app with the root folder as the static folder
app = Flask(__name__, static_folder=root_folder)

# Number of images to display per page
IMAGES_PER_PAGE = 200

@app.route('/')
def index():
    return show_images('')

@app.route('/folder/<path:folder_name>')
def show_images(folder_name):
    # Construct the full path to the selected folder
    folder_path = os.path.join(root_folder, folder_name)

    # List all subfolders in the current folder
    subfolders = [f.name for f in os.scandir(folder_path) if f.is_dir()]
    subfolders.sort()

    # Get the list of image files in the selected folder
    image_files = os.listdir(folder_path)

    # Filter only image files
    image_files = [file for file in image_files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    image_files.sort()

    # Pagination logic
    page = int(request.args.get('page', 1))
    start = (page - 1) * IMAGES_PER_PAGE
    end = start + IMAGES_PER_PAGE
    paginated_images = image_files[start:end]

    # Enumerate the images for display in the template
    enumerated_images = list(enumerate(paginated_images, start=start + 1))

    # Determine if there are previous/next pages
    has_prev = page > 1
    has_next = end < len(image_files)

    # Render the index.html template
    return render_template('index.html', 
                           subfolders=subfolders, 
                           enumerated_images=enumerated_images, 
                           folder_name=folder_name,
                           page=page, 
                           has_prev=has_prev, 
                           has_next=has_next)

if __name__ == '__main__':
    app.run(debug=True, port=8081)
