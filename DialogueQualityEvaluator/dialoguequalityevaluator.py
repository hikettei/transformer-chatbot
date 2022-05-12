import torch
from torchtext.data.metrics import bleu_score
from tqdm import tqdm
import random


def evaluate_dialogue_quality_of_model(
        test_data_file_path,
        predict_function,
        max_reference_logs=7,
        min_reference_logs=1,
        random_seed=20060317,
        skip_probability=0.9,
        return_summary_text=False
):
    random.seed(random_seed)
    print(f"Loading file {test_data_file_path}")
    with open(test_data_file_path, mode='r') as f:
        test_data = f.read()
    test_data = test_data.split("\n")

    print(f"Loaded test data that contains {len(test_data)} utterances.")

    total_scores = []
    weights = []
    counter = 0
    bar1 = tqdm(range(min_reference_logs, max_reference_logs+1), position=0)
    for r in bar1:
        tqdm.write(f"Evaluating reply that takes {r} utterance...")
        subtotal_score = 0
        subcounter = 0
        bar2 = tqdm(range(r+1, len(test_data)-1), position=1)
        for n in bar2:
            if random.random() >= skip_probability:
                try:
                    input_data = test_data[n-r-1:n-1]
                    results = predict_function(input_data)

                    score = bleu_score([list(s)
                                for s in results], [[list(test_data[n])]])
                    subtotal_score += score
                    subcounter += 1
                    counter + 1
                except:
                    pass
        tqdm.write(f"In {r} reference uttearances, score: {subtotal_score / subcounter}")
        # average of sub loop.
        total_scores.append(subtotal_score / subcounter)
        weights.append(1/r)
    # generete result summary graph
    normalized = total_scores
    normalized = [s-min(normalized) for s in normalized]
    normalized = [s/max(normalized) for s in normalized]
    normalized = [int(s*32) for s in normalized]
    summary = ""

    tqdm.write("Summary of performance each reference utterances.")
    tqdm.write("Utter.,Score, Bar")
    for r, n in enumerate(normalized):
        star = "*"
        no_star = "-"
        summary += f"  {r+1}   : {total_scores[r]:.4f} {star*n+no_star*(32-n)}\n"
        
    tqdm.write(summary)

    # weighted average
    finaly_score = [s * w for s, w in zip(total_scores, weights)]
    finaly_score = sum(finaly_score) / len(finaly_score)
    
    tqdm.write(f"finaly score = {finaly_score}")
    
    if return_summary_text:
        return finaly_score, summary
    else:
        return finaly_score


