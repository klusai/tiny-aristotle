# BoLoCo: Boolean Logic Corpus
# This script generates a dataset of Boolean logic expressions, evaluates them, and splits them into training, validation, and test datasets.
# BoLoCo stands for Boolean Logic Corpus, aiming to provide a structured dataset for logical operations research and machine learning applications.

import argparse
import random
import os
import logging  # Add logging import
import time  # Add time import

# Define tokens used in the dataset
SOS_TOKEN = "<s>"  # Start of sequence token
EVAL_TOKEN = "<eval/>"  # Evaluation token separating expression and result
EOS_TOKEN = "</s>"  # End of sequence token
ERROR_TOKEN = "<err/>"  # Error token for invalid expressions

logger = logging.getLogger("BoLoCo")  # Add logger


def generate_logic_expressions(n):
    start_time = time.time()  # Start timing
    logger.info(f"Generating logic expressions with up to {n} tokens.")
    literals = ["T", "F"]  # Literals representing True and False
    operators = ["AND", "OR"]  # Logical operators

    # Recursive function to build valid expressions
    def build(current, tokens_left, expect_literal, open_parens):
        if tokens_left == 0:
            # Add only complete expressions (balanced parentheses, no dangling operators)
            if open_parens == 0 and not expect_literal:
                valid_expressions.add(" ".join(current))
            return

        # Add literals if expecting a literal
        if expect_literal:
            for literal in literals:
                build(current + [literal], tokens_left - 1, False, open_parens)

        # Add operators if expecting an operator
        if not expect_literal and current and current[-1] not in ("(", "AND", "OR"):
            for operator in operators:
                build(current + [operator], tokens_left - 1, True, open_parens)

        # Add opening parenthesis if expecting a literal
        if expect_literal:
            build(current + ["("], tokens_left - 1, True, open_parens + 1)

        # Add closing parenthesis if there are open ones and the last token is not an operator
        if (
            open_parens > 0
            and not expect_literal
            and current
            and current[-1] not in ("(", "AND", "OR")
        ):
            build(current + [")"], tokens_left - 1, False, open_parens - 1)

    valid_expressions = set()
    for length in range(1, n + 1):
        build([], length, True, 0)  # Start expecting a literal or '('

    logger.info(
        f"Generated {len(valid_expressions)} expressions in {time.time() - start_time:.2f} seconds."
    )
    return valid_expressions


# Evaluates a logic expression using Python's eval function
def eval_expression(expression):
    """
    Evaluate a logic expression.

    Args:
        expression (str): The logic expression to evaluate.

    Returns:
        bool: The result of the evaluated expression.
    """
    # Replace logical operators with Python equivalents
    expression = expression.replace("AND", "and").replace("OR", "or")

    # Replace literals with Python equivalents
    expression = expression.replace("T", "True").replace("F", "False")

    # Evaluate the expression
    try:
        return eval(expression)
    except Exception as e:
        print(f"Error evaluating expression: {expression}. Error: {e}")
        return None


def save_dataset(dataset, name):
    start_time = time.time()  # Start timing
    logger.info(f"Saving dataset to {name}.txt with {len(dataset)} expressions.")
    """
    Save the dataset to a file with the specified name.

    Args:
        dataset (list): The dataset containing logic expressions.
        name (str): The name of the file to save the dataset.
    """
    with open(f"{name}.txt", "w") as f:
        for expr in sorted(dataset):
            eval_result = eval_expression(expr)
            if eval_result is True:
                eval_string = "T"
            elif eval_result is False:
                eval_string = "F"
            else:
                eval_string = ERROR_TOKEN
            f.write(f"{SOS_TOKEN} {expr} {EVAL_TOKEN} {eval_string} {EOS_TOKEN}\n")
    logger.info(f"Saved dataset in {time.time() - start_time:.2f} seconds.")


def count_tokens(dataset):
    """
    Count the total number of tokens in the dataset, expressed as millions.

    Args:
        dataset (list): The dataset containing logic expressions.

    Returns:
        float: The total number of tokens in the dataset, expressed as millions (mtokens).
    """
    return sum(len(expr.split()) for expr in dataset)


def format_output_filename(
    max_tokens, seed, set_name, output_dir, expr_size, tokens, ratio
):
    # based on this: f"{args.output_dir}/boloco_train_se{args.seed}_ex{train_size}_mtks{count_mtokens(train_set):.0f}_ra{args.train_ratio*100:.0f}",
    return f"{output_dir}/boloco-{set_name}-mt{max_tokens}_se{seed}_ex{expr_size}_tks{tokens:.0f}_ra{ratio*100:.0f}"


def main(args):
    logging.basicConfig(level=logging.INFO)  # Configure logger

    logger.info("Starting BoLoCo dataset generation.")

    start_time = time.time()  # Start timing

    random.seed(args.seed)
    max_tokens = args.max_tokens
    logger.info(f"Generating expressions with max tokens: {max_tokens}")
    expressions = list(generate_logic_expressions(max_tokens))
    random.shuffle(expressions)
    logger.info("Shuffling expressions.")

    # Split dataset into training, validation, and test sets
    train_size = int(len(expressions) * args.train_ratio)
    validate_size = int(len(expressions) * args.validate_ratio)
    test_size = len(expressions) - train_size - validate_size

    train_set = expressions[:train_size]
    validate_set = expressions[train_size : train_size + validate_size]
    test_set = expressions[train_size + validate_size :]

    logger.info(
        f"Splitting dataset into train, validate, and test sets with ratios {args.train_ratio}, {args.validate_ratio}, {args.test_ratio}."
    )
    # Create output directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save datasets with names reflecting BoLoCo and seed/ratios
    logger.info("Saving training set.")
    save_dataset(
        train_set,
        format_output_filename(
            args.max_tokens,
            args.seed,
            "train",
            args.output_dir,
            train_size,
            count_tokens(train_set),
            args.train_ratio,
        ),
    )
    logger.info("Saving validation set.")
    save_dataset(
        validate_set,
        format_output_filename(
            args.max_tokens,
            args.seed,
            "validate",
            args.output_dir,
            validate_size,
            count_tokens(validate_set),
            args.validate_ratio,
        ),
    )
    logger.info("Saving test set.")
    save_dataset(
        test_set,
        format_output_filename(
            args.max_tokens,
            args.seed,
            "test",
            args.output_dir,
            test_size,
            count_tokens(test_set),
            args.test_ratio,
        ),
    )
    logger.info(
        f"BoLoCo dataset generation completed in {time.time() - start_time:.2f} seconds."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Boolean logic expressions and split into BoLoCo datasets."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="boloco_datasets",
        help="Directory to save the BoLoCo datasets.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=17,
        help="Maximum number of tokens in the logic expression.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling the expressions.",
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.7, help="Ratio of training set."
    )
    parser.add_argument(
        "--validate_ratio", type=float, default=0.15, help="Ratio of validation set."
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.15, help="Ratio of test set."
    )

    args = parser.parse_args()

    main(args)
