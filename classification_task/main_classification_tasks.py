# main.py

import argparse
from tasks.classification_task import SST2ClassificationTask, SUBJClassificationTask
from language_model.gpt_model import GPTModel  # This would be implemented for your specific model
from agent import PromptOptimizationAgent
from utils import Config


def main():
    """
    Main function to run the prompt optimization process.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Prompt Optimization with MCTS")
    parser.add_argument("--task_name", type=str, choices=["sst2", "subj"], required=True,
                        help="Classification task name")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the dataset")
    parser.add_argument("--init_prompt", type=str, required=True,
                        help="Initial prompt to optimize")
    parser.add_argument("--pred_model", type=str, default="gpt-3.5-turbo",
                        help="Model used for predictions")
    parser.add_argument("--train_size", type=int, default=70,
                        help="Number of training examples")
    parser.add_argument("--eval_size", type=int, default=50,
                        help="Number of evaluation examples")
    parser.add_argument("--iteration_num", type=int, default=10,
                        help="Number of MCTS iterations")
    parser.add_argument("--depth_limit", type=int, default=6,
                        help="Maximum depth of the MCTS search")
    parser.add_argument("--expand_width", type=int, default=3,
                        help="Number of actions to expand per node")
    parser.add_argument("--exploration_weight", type=float, default=1.0,
                        help="Exploration weight for UCB formula")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Create configuration object
    config = Config()
    config.iteration_num = args.iteration_num
    config.depth_limit = args.depth_limit
    config.expand_width = args.expand_width
    config.exploration_weight = args.exploration_weight
    config.seed = args.seed
    
    # Initialize task
    if args.task_name == "sst2":
        task = SST2ClassificationTask(args.data_dir)
    else:  # subj
        task = SUBJClassificationTask(args.data_dir)
    
    # Load dataset
    train_data, eval_data, _ = task.load_task_dataset(
        train_size=args.train_size,
        eval_size=args.eval_size,
        test_size=0,
        seed=args.seed
    )
    
    # Initialize models
    pred_model = GPTModel(model_name=args.pred_model)
    optim_model = GPTModel(model_name=args.pred_model)  # Using same model for both
    
    # Initialize agent
    agent = PromptOptimizationAgent(task, pred_model, optim_model, config)
    
    # Run optimization
    print(f"Initial prompt: {args.init_prompt}")
    optimized_prompt = agent.optimize(args.init_prompt, train_data, eval_data)
    print(f"Optimized prompt: {optimized_prompt}")
    
    # Evaluate on the evaluation set
    print("Evaluating optimized prompt...")
    # This would perform evaluation in a real implementation
    
    print("Done!")


if __name__ == "__main__":
    main()