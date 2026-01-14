## Wrapper for mace.cli.run_train.main ##

from mace.cli.run_train import train_main
from scripts.eval_horm import evaluate_hessian_on_horm_dataset

if __name__ == "__main__":
    args = train_main()
    
    df_results, aggregated_results = evaluate_hessian_on_horm_dataset(
        args=args,
        max_samples=1000,
    )