# AI University & Supervisor Matching - Simple Demo Code
# Author: Syad Ali Raza
# Description:
# A minimal working example that matches student interests to universities
# using semantic similarity (Sentence Transformers).

from sentence_transformers import SentenceTransformer, util

# Load a small pre-trained model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample dataset (you can expand later)
universities = {
    "University of Oxford": "quantum physics, condensed matter, superconductivity, photonics",
    "MIT": "machine learning, artificial intelligence, robotics, deep learning",
    "University of Toronto": "computer vision, NLP, computational linguistics",
    "ETH Zurich": "autonomous systems, AI safety, reinforcement learning",
    "National University of Singapore": "data science, cloud computing, cybersecurity"
}

# Get student profile input
student_interests = input("Enter your research interests: ")

# Encode all text into embeddings
student_embedding = model.encode(student_interests)
scores = {}

for uni, description in universities.items():
    uni_embedding = model.encode(description)
    similarity = util.cos_sim(student_embedding, uni_embedding).item()
    scores[uni] = similarity

# Sort universities by similarity score
ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

print("\nTop Recommended Universities:\n")
for uni, score in ranked_results:
    print(f"{uni}  -> Match Score: {score:.2f}")
