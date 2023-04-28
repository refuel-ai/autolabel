from refuel_oracle.confidence import ConfidenceCalculator
import pickle as pkl
import sklearn
import random

matches = pkl.load(open("matches.pkl", "rb"))
confidences = pkl.load(open("confidences.pkl", "rb"))
random_confidences = [random.random() for _ in range(len(matches))]

print(matches, confidences)
print(ConfidenceCalculator.compute_auroc(matches, confidences, plot=True))
ConfidenceCalculator.plot_data_distribution(
    matches, confidences, "real.png", save_data=False
)

print(ConfidenceCalculator.compute_auroc(matches, random_confidences, plot=False))
ConfidenceCalculator.plot_data_distribution(
    matches, random_confidences, "random.png", save_data=False
)
