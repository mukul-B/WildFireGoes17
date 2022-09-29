import os

from matplotlib import pyplot as plt

from GlobalValues import data_dir, goes_folder, viirs_folder, compare, logs, testing_dir, training_dir, GOES_ndf


def prepareDir(location, product):
    site_dir = location
    product_dir = product

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists(data_dir + '/' + site_dir):
        os.mkdir(data_dir + '/' + site_dir)
    if not os.path.exists(data_dir + '/' + site_dir + '/' + goes_folder):
        os.mkdir(data_dir + '/' + site_dir + '/' + goes_folder)
    if not os.path.exists(data_dir + '/' + site_dir + '/' + goes_folder + '/' + product_dir):
        os.mkdir(data_dir + '/' + site_dir + '/' + goes_folder + '/' + product_dir)
    if not os.path.exists(data_dir + '/' + site_dir + '/' + goes_folder + '/' + product_dir + '/tif'):
        os.mkdir(data_dir + '/' + site_dir + '/' + goes_folder + '/' + product_dir + '/tif')
    if not os.path.exists(data_dir + '/' + site_dir + '/' + viirs_folder):
        os.mkdir(data_dir + '/' + site_dir + '/' + viirs_folder)

    if not os.path.exists(data_dir + '/' + site_dir + '/' + compare):
        os.mkdir(data_dir + '/' + site_dir + '/' + compare)

    if not os.path.exists(logs):
        os.mkdir(logs)
    if not os.path.exists(training_dir):
        os.mkdir(training_dir)

    if not os.path.exists(testing_dir):
        os.mkdir(testing_dir)

    if not os.path.exists(GOES_ndf):
        os.mkdir(GOES_ndf)

def plot_sample(col,titles):
    num = len(col)
    fig, axs = plt.subplots(1, num, constrained_layout=True)
    for i, j in enumerate(col):
        axs[i].imshow(j)
        axs[i].set_title(titles[i])
    plt.show()