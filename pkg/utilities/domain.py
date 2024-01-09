import numpy as np

def cosine_similarity(array1, array2):
  dot_product = np.dot(array1, array2)
  norm_array1 = np.linalg.norm(array1)
  norm_array2 = np.linalg.norm(array2)
  similarity = dot_product / (norm_array1 * norm_array2)
  return similarity

def meanOfSimilarity(arrays):
  if(len(arrays) > 1):
    similarities = []
    for i in range(len(arrays)-1):
      for j in range(i+1, len(arrays)):
        similarity = cosine_similarity(arrays[i], arrays[j])
        similarities.append(similarity)

    mean_similarity = np.mean(similarities)
    return mean_similarity
  else:
    return 1