import os

from GlobalValues import data_dir, goes_folder, viirs_folder, compare, logs, training


def prepareDir(location, product):
    site_dir = location
    print("hello")
    product_dir = product

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists(data_dir + '/' + site_dir):
        os.mkdir(data_dir + '/' + site_dir)
    if not os.path.exists(data_dir + '/' + site_dir + '/' + goes_folder):
        os.mkdir(data_dir + '/' + site_dir + '/' + goes_folder)
    if not os.path.exists(data_dir + '/' + site_dir + '/' + goes_folder + '/' + product_dir):
        os.mkdir(data_dir + '/' + site_dir + '/' + goes_folder + '/' + product_dir)
    if not os.path.exists(data_dir + '/' + site_dir + '/' + viirs_folder):
        os.mkdir(data_dir + '/' + site_dir + '/' + viirs_folder)

    if not os.path.exists(data_dir + '/' + site_dir + '/' + compare):
        os.mkdir(data_dir + '/' + site_dir + '/' + compare)

    if not os.path.exists(logs):
        os.mkdir(logs)
    if not os.path.exists(data_dir + '/' + site_dir + '/' + training):
        os.mkdir(data_dir + '/' + site_dir + '/' + training)
