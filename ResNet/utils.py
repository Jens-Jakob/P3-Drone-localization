# utils.py
import os

def get_drone_images(drone_folder_path):
    """ Get all drone images (only .png files) """
    return [os.path.join(drone_folder_path, img) for img in os.listdir(drone_folder_path) if
            img.endswith('.png') and not img.startswith('._')]

def get_satellite_images(satellite_folder_path):
    """ Get all satellite images (only .png files) """
    return [os.path.join(satellite_folder_path, img) for img in os.listdir(satellite_folder_path) if
            img.endswith('.png') and not img.startswith('._')]
