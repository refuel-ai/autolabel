import pickle as pkl

from autolabel.confidence import ConfidenceCalculator

ConfidenceCalculator.plot_data_distribution(
    match=pkl.load(open("matches.pkl", "rb")),
    confidence=pkl.load(open("confidences.pkl", "rb")),
)
