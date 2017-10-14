MAX_RESULTS = 5
NUM_HM_LEVEL = 3
CERTAINTY_LEVELS = 5

RESULTS_CHOICES = [
	(i, str(i)) for i in range(1, MAX_RESULTS+1)
]

HM_CHOICES = [
	(0, "None"),
	(1, "Medium"),
	(2, "Local")
]

CERTAINTY_CHOICES = [
	(i, str(i)) for i in range(1, CERTAINTY_LEVELS+1)
]