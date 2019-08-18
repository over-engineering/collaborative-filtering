import numpy as np
from similarity import adjusted_cosine_similarity, pearson_correlation, get_similarity_matrix
from recommendation import weighted_sum, prediction

user_list = ["Chang", "Chan", "jmpark", "Ruby", "suji kang", "Cold New User", "Hot New User", "Chang's soul mate"]
restaurant_list = ["아비꼬", "롱타임노씨", "피맥하우스", "오신 매운갈비찜", "오빠닭", "바나나 피자"]

test_data = np.array([
    [2.5, 3.5, 3.0, 3.5, 2.5, 3.0],
    [3.0, 3.5, 1.5, 5.0, 3.5, 3.0],
    [2.5, 3.5, 0, 3.5, 0, 4.0],
    [0, 3.5, 3.0, 4.0, 2.5, 4.5],
    [1.0, 0, 3.0, 2.0, 0, 1.5],
    [0, 0, 0, 0, 0, 0],
    [1.0, 0, 2.5, 3.5, 3.5, 4.5],
    [2.5, 3.5, 3.0, 3.5, 2.5, 3.0]
])

similarity_matrix = get_similarity_matrix(test_data, target="item")
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print("@@@@@Item-based Similarity Matrix@@@@@@@@@@@@@@@@@@@@@@")
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print(similarity_matrix)
print("\n\n")

result = prediction(test_data, target="item")
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print("@@@@@Item-based Predicted Rating Matrix@@@@@@@@@@@@@@@@")
print(result)
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print("\n\n")

similarity_matrix = get_similarity_matrix(test_data, target="user")
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print("@@@@@User-based Similarity Matrix@@@@@@@@@@@@@@@@@@@@@@")
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print(similarity_matrix)
print("\n\n")

result = prediction(test_data, target="user")
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print("@@@@@User-based Predicted Rating Matrix@@@@@@@@@@@@@@@@")
print(result)
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print("\n\n")
