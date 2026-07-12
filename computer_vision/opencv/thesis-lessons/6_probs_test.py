import numpy as np

# 1. Simulate 3 raw unscaled outputs(logits) from an AI model
# Index 0 = Keyboard, Index 1 = Mouse, Index 2 = Empty Desk
# 1000 items in trained data = 1000 logits
mock_logits = np.array([2.0, 5.0, 1.0])

# 2. Execute the Softmax logic
# If an AI model gets highly confident, it can easily emit a logit score of 1000.0.
# if that score was exponentially increase the code will exceed our computer's limit and
# crash and returns unreadable errors like [NaN, NaN, NaN]
# so by subtracting each logits to the highest logit, it ensures that the highest number 
# would be 0 so when it gets exponentiated the value would be 1
exp_scores = np.exp(mock_logits - np.max(mock_logits)) 

# Normalizing simply means dividing each individual exponential 
# score by the total sum of all the scores combined
probs = exp_scores / exp_scores.sum()

# 3. Print out the results to examine the mathematical location
print("="*40)
print(f"Raw model logits: {mock_logits}")
print(f"Calculated softmax probabilities: {probs}")
print(f"Sum of all probabilities: {np.sum(probs):.1f}")
print(f"Winning Class index position: {np.argmax(probs)}")
print("="*40)
