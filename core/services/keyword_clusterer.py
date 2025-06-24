
from sentence_transformers import SentenceTransformer
import sys
import os
import numpy as np
# get current working directory
sys.path.append(os.getcwd())

model = SentenceTransformer('all-MiniLM-L6-v2')

with open("output-keywords.txt", "r") as f:
    keywords = [line.strip() for line in f]

# sentences are encoded by calling model.encode()
embeddings = model.encode(keywords)

# embeddings in a array of numpy vectors

numpy_embeddings = np.array(embeddings)




