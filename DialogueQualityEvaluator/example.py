from dialoguequalityevaluator import *


# input: List of input sentences. type: list(str)
# output: List of candidate reply messages. type: list(str)
def pseudo_model(input_logs):
    return [input_logs[0]]

# Run test.
score = evaluate_dialogue_quality_of_model("./test.txt", pseudo_model)
print(f"Score: {score}")
