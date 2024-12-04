# similarity_calculator.py
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(input_features, folder_features):
    """ Compare the input image's features with images in the folder """
    similarities = []
    for features in folder_features:
        similarity = cosine_similarity([input_features], [features])
        similarities.append(similarity[0][0])
    return similarities
