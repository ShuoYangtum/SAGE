import argparse
import torch
import time
from sage import SAGE


def build_parser():
    parser = argparse.ArgumentParser(description="Train SAGE and run column imputation.")
    parser.add_argument("--target-column", type=str, required=True, help="Column to impute in test CSV.")
    parser.add_argument("--train-path", type=str, default="train.csv", help="Training CSV path.")
    parser.add_argument("--test-path", type=str, default="test.csv", help="Test CSV path.")
    parser.add_argument("--output-path", type=str, default="../output.csv", help="Output CSV path.")
    parser.add_argument("--model-name", type=str, default="gpt2", help="HF model name.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--use-lora", action="store_true", help="Enable LoRA fine-tuning.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic torch backend for reproducibility.")
    parser.add_argument("--checkpoint-path", type=str, default="best_generator_model.pt", help="Path to save best model checkpoint.")

    # Fit arguments
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=50)
    parser.add_argument("--val-ratio", type=float, default=0.001)
    parser.add_argument("--early-stopping-rounds", type=int, default=5)
    parser.add_argument("--mi-n-bins", type=int, default=10)
    parser.add_argument(
        "--mi-strategy",
        type=str,
        default="fd",
        choices=["fd", "quantile", "uniform", "kmeans"],
        help="Discretization strategy for MI preprocessing. 'fd' follows paper default."
    )
    parser.add_argument(
        "--mi-threshold",
        type=float,
        default=None,
        help="MI threshold for selecting relevant context. If omitted, use median MI from training set."
    )
    parser.add_argument("--constrain-string-values", action="store_true")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-sample-num", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument(
        "--drop-columns",
        type=str,
        default="",
        help="Comma-separated columns to drop before training."
    )

    # Imputation arguments
    parser.add_argument("--max-new-tokens-per-value", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--disable-final-constraints", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    device = torch.device(args.device)
    
    drop_columns = [c.strip() for c in args.drop_columns.split(",") if c.strip()]
    apply_final_constraints = not args.disable_final_constraints

    generator = SAGE(model_name=args.model_name, device=device, use_lora=args.use_lora) # Qwen/Qwen3-0.6B-Base meta-llama/Llama-3.2-1B
    a = time.time()
    generator.fit(
        args.train_path,
        max_sample_num=args.max_sample_num,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        max_length=args.max_length,
        shuffle=args.shuffle,
        early_stopping_rounds=args.early_stopping_rounds,
        mi_threshold=args.mi_threshold,
        mi_n_bins=args.mi_n_bins,
        mi_strategy=args.mi_strategy,
        constrain_string_values=args.constrain_string_values,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        drop=drop_columns,
        seed=args.seed,
        deterministic=args.deterministic,
        checkpoint_path=args.checkpoint_path,
        load_best_at_end=True
    )

    # generator.model.load_state_dict(torch.load("best_generator_model.pt")) 

    result = generator.imputation(
        args.test_path,
        args.target_column,
        max_new_tokens_per_value=args.max_new_tokens_per_value,
        temperature=args.temperature,
        mi_threshold=args.mi_threshold,
        apply_final_constraints=apply_final_constraints,
        save_path=args.output_path
    )

    # Print training time and results
    training_time = time.time() - a
    print(f"Training time: {training_time:.2f} seconds")
    
    # Print evaluation results
    if result['task_type'] == 'regression':
        print(f"MSE: {result.get('mse', 'N/A')}")
        print(f"MAE: {result.get('mae', 'N/A')}")
        print(f"RMSE: {result.get('rmse', 'N/A')}")
    else:
        print(f"Accuracy: {result.get('accuracy', 'N/A')}")
        print(f"Error Rate: {result.get('error_rate', 'N/A')}")

if __name__=='__main__':
    main()