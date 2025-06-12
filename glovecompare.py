#!/usr/bin/python3
"""
cosine-similarity of input words to GloVe: Global Vectors
for Word Representation
"""

import argparse
import math

import numpy as np


def load_glove() -> dict[str, np.ndarray]:
    embeddings = {}
    with open("./data/glove.6B.300d.txt", "r") as f:
        for line in f:
            word, vec = line.split(" ", maxsplit=1)
            embeddings[word] = np.fromstring(vec, sep=" ", dtype=np.float32, count=300)

    return embeddings


def cosine_similarity(u, v):
    cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return max(-1.0, min(1.0, cos_theta))


def angle(cos_theta):
    """Returns the angle in radians"""
    return math.acos(cos_theta)


def topk_similar(embeds, word: str, k: int = 10):
    v = embeds[word]
    all_words = np.array(list(embeds.keys()))
    mat = np.stack(embeds.values())
    sims = mat @ v / (np.linalg.norm(mat, axis=1) * np.linalg.norm(v))
    top_idx = sims.argsort()[-k - 1 :][::-1]  # drop self
    for i in top_idx:
        if all_words[i] != word:
            yield all_words[i], float(sims[i])


def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(
        description="A simple program to calculate the square of a number."
    )

    # Add arguments
    parser.add_argument("word1", type=str, help="The first word to compare")
    parser.add_argument("word2", type=str, help="The second word to compare")

    args = parser.parse_args()

    embeddings = load_glove()
    cos_theta = cosine_similarity(embeddings[args.word1], embeddings[args.word2])
    theta = math.degrees(angle(cos_theta))
    print(
        f"Cosine Similarity between {args.word1} and {args.word2}: cosine: {cos_theta} angle: {theta}"
    )

    topk = [x for x, _ in list(topk_similar(embeddings, args.word1))]
    print(f"topk_similar {args.word1}: {topk}")
    topk = [x for x, _ in list(topk_similar(embeddings, args.word2))]
    print(f"topk_similar {args.word2}: {topk}")


if __name__ == "__main__":
    main()
