import pickle as pkl

from refuel_oracle.confidence import ConfidenceCalculator

ConfidenceCalculator.plot_data_distribution(
    match=pkl.load(open("matches.pkl", "rb")),
    confidence=pkl.load(open("confidences.pkl", "rb")),
)
