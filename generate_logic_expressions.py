import argparse
import random

SOS_TOKEN = "<s>"
EVAL_TOKEN = "=>"
EOS_TOKEN = "</s>"
ERROR_TOKEN = "<err/>"


def generate_logic_expressions(n):
    """
    Generate all valid logic expressions up to n tokens.

    Args:
        n (int): The maximum number of tokens in the logic expression.

    Returns:
        set: A set containing all unique valid logic expressions as strings.
    """

    literals = ["T", "F"]
    operators = ["AND", "OR"]

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
        f.write(f"Total valid expressions: {len(dataset)}\n")


def main(args):
    random.seed(args.seed)
    max_tokens = args.max_tokens
    expressions = list(generate_logic_expressions(max_tokens))
    random.shuffle(expressions)

    train_size = int(len(expressions) * args.train_ratio)
    validate_size = int(len(expressions) * args.validate_ratio)
    test_size = len(expressions) - train_size - validate_size

    train_set = expressions[:train_size]
    validate_set = expressions[train_size : train_size + validate_size]
    test_set = expressions[train_size + validate_size :]

    save_dataset(train_set, f"train_{args.seed}_{args.train_ratio}")
    save_dataset(validate_set, f"validate_{args.seed}_{args.validate_ratio}")
    save_dataset(test_set, f"test_{args.seed}_{args.test_ratio}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate logic expressions and split into datasets."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=5,
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
