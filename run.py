import argparse
import os
import re
import numpy as np
import mlx.core as mx  # Import mlx core for array manipulations
from train import TransformerLM, load_dataset
from tqdm import tqdm
from boloco.boloco import SOS_TOKEN, EVAL_TOKEN, EOS_TOKEN


# Function to parse hyperparameters from the file name
def parse_model_name(file_name):
    """
    Parse hyperparameters from the model file name.
    Example file name: transformer_ptb_b32_cs128_blocks2_dim256_heads2_lr0.001.npz
    """
    match = re.match(
        r".*/(?P<dataset>[a-zA-Z0-9_\-]+)_b(?P<batch_size>\d+)_cs(?P<context_size>\d+)_"
        r"blocks(?P<num_blocks>\d+)_dim(?P<dim>\d+)_heads(?P<num_heads>\d+)_it\d+_"
        r"lr(?P<lr>[0-9.\-e]+).npz",
        file_name,
    )
    if not match:
        raise ValueError(
            "Invalid model file name format. Cannot extract hyperparameters."
        )

    # Extract hyperparameters from the file name
    params = match.groupdict()
    return {
        "dataset": params["dataset"],
        "batch_size": int(params["batch_size"]),
        "context_size": int(params["context_size"]),
        "num_blocks": int(params["num_blocks"]),
        "dim": int(params["dim"]),
        "num_heads": int(params["num_heads"]),
        "learning_rate": float(params["lr"]),
    }


def get_npz_file():
    for file in os.listdir():
        if file.endswith(".npz"):
            return file
    return None


def process_input(input_text, model, vocab, id_to_word):
    """
    Tokenize input text, pass it through the model, and return the single-token prediction.
    """
    # Tokenize the input
    token_ids = [
        vocab[word] if word in vocab else vocab["<unk>"] for word in input_text.split()
    ]

    # Prepare the input as a batch of 1 sequence
    input_array = np.array([token_ids], dtype=np.int32)  # Shape: (1, sequence_length)
    input_mx_array = mx.array(input_array)  # Convert to mx.array

    # Run the model
    output_logits = model(input_mx_array)

    # Extract the prediction for the last token
    predicted_token_id = mx.argmax(output_logits[0][-1]).tolist()
    predicted_word = id_to_word[predicted_token_id]

    return predicted_word, token_ids, predicted_token_id


def evaluate_model(eval_file, model, vocab, id_to_word):
    """
    Evaluate the model on a dataset loaded from a .txt file.
    """
    if not os.path.exists(eval_file):
        raise FileNotFoundError(f"Evaluation file {eval_file} not found.")

    with open(eval_file, "r") as f:
        lines = f.readlines()

    correct = 0
    total = 0

    with tqdm(total=len(lines), desc="Evaluatiing") as pbar:
        for line in lines:
            line = line.strip()
            if not line:
                pbar.update(1)
                continue

            # match = re.match(r"<\|in\|> (.*?) <\|out\|> (.*?) <\|end\|>", line)
            match = re.match(f"{SOS_TOKEN} (.*?) {EVAL_TOKEN} (.*?) {EOS_TOKEN}", line)
            if not match:
                pbar.update(1)
                continue

            input_text, expected_output = match.groups()
            # augmented_input = f"<|in|> {input_text} <|out|>"
            augmented_input = f"{SOS_TOKEN} {input_text} {EVAL_TOKEN}"

            predicted_word, _, _ = process_input(
                augmented_input, model, vocab, id_to_word
            )

            is_correct = predicted_word == expected_output
            correct += is_correct
            total += 1

            # Update progress bar with the current accuracy
            accuracy = correct / total if total > 0 else 0.0
            pbar.set_postfix(accuracy=f"{accuracy * 100:.2f}%")
            pbar.update(1)

    print(f"Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Load and infer with a Transformer LM."
    )
    parser.add_argument(
        "--model_file",
        type=str,
        required=False,
        help="Path to the model file (.npz) with hyperparameters in its name.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["interactive", "eval"],
        required=True,
        help="Mode of operation: 'interactive' or 'eval'.",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        required=False,
        help="Path to the evaluation dataset file (required for eval mode).",
    )

    args = parser.parse_args()

    # Extract hyperparameters from the file name
    model_file = args.model_file
    if model_file is None:
        model_file = get_npz_file()
        if model_file is None:
            print("No .npz file found in the current directory.")
            exit(1)

    params = parse_model_name(model_file)
    print(f"Parsed Hyperparameters: {params}")

    # Load the vocabulary and dataset based on the parsed dataset name
    vocab, train, valid, test = load_dataset(params["dataset"])

    # Initialize the model with parsed parameters
    model = TransformerLM(
        len(vocab),
        params["num_blocks"],
        params["dim"],
        params["num_heads"],
        checkpoint=False,
    )

    # Load pre-trained weights
    model.load_weights(model_file)
    model.eval()

    # Reverse vocabulary (to map token IDs back to words)
    id_to_word = {v: k for k, v in vocab.items()}

    if args.mode == "interactive":
        print("Transformer Language Model Inference")
        print("Type a sentence and press Enter to see the model's prediction.")
        print("Type 'exit' to quit the program.")
        while True:
            user_input = input("Enter a sentence: ").strip()
            if user_input.lower() == "exit":
                print("Exiting...")
                break

            augmented_user_input = f"{SOS_TOKEN} {user_input.upper()} {EVAL_TOKEN}"
            predicted_sentence, token_ids, predicted_ids = process_input(
                augmented_user_input, model, vocab, id_to_word
            )

            print(f"Augmented Input Sentence: {augmented_user_input}")
            print(f"Input Sentence: {user_input}")
            print(f"Token IDs: {token_ids}")
            print(f"Predicted Token IDs: {predicted_ids}")
            print(f"Predicted Sentence: {predicted_sentence}")
            print("-" * 50)
    elif args.mode == "eval":
        if not args.eval_file:
            print("Evaluation file is required in eval mode.")
            exit(1)
        evaluate_model(args.eval_file, model, vocab, id_to_word)
