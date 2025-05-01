# search_algo/mcts.py

import numpy as np
import math
import random
from collections import defaultdict

class MCTSNode:
    """
    Node class for Monte Carlo Tree Search.
    Each node represents a prompt state in the optimization process.
    """
    def __init__(self, prompt, parent=None, parent_action=None):
        self.prompt = prompt  # The current prompt text
        self.parent = parent  # Parent node
        self.parent_action = parent_action  # Action that led to this node
        self.children = []  # Child nodes
        self.visits = 0  # Number of visits to this node
        self.reward = 0  # Accumulated reward
        self.untried_actions = []  # List of actions not yet tried
        
    def add_child(self, prompt, action):
        """Add a child node with the given prompt and action."""
        child = MCTSNode(prompt, parent=self, parent_action=action)
        self.children.append(child)
        return child
        
    def update(self, reward):
        """Update node statistics with a new reward."""
        self.visits += 1
        self.reward += reward
        
    def is_fully_expanded(self):
        """Check if all possible actions from this node have been explored."""
        return len(self.untried_actions) == 0
    
    def best_child(self, exploration_weight=1.0):
        """
        Select the best child according to UCB formula.
        Higher exploration_weight encourages exploration over exploitation.
        """
        if not self.children:
            return None
            
        # Use UCB (Upper Confidence Bound) formula to balance exploration and exploitation
        ucb_values = [
            (child.reward / child.visits) + exploration_weight * math.sqrt(2 * math.log(self.visits) / child.visits)
            for child in self.children
        ]
        return self.children[np.argmax(ucb_values)]
    
    def rollout_policy(self, actions):
        """Select an action for rollout simulation."""
        return random.choice(actions)


class MCTS:
    """
    Monte Carlo Tree Search implementation for prompt optimization.
    Uses classification accuracy as the reward metric.
    """
    def __init__(self, task, model, iterations=10, exploration_weight=1.0, depth_limit=6, expand_width=3):
        self.task = task  # Task instance for evaluation
        self.model = model  # Language model for generating responses
        self.iterations = iterations  # Number of MCTS iterations
        self.exploration_weight = exploration_weight  # UCB exploration parameter
        self.depth_limit = depth_limit  # Maximum search depth
        self.expand_width = expand_width  # Number of actions to expand per node
        
    def search(self, initial_prompt, eval_dataset):
        """
        Run the MCTS algorithm to find the best prompt.
        Returns the best prompt found.
        """
        root = MCTSNode(initial_prompt)
        
        for _ in range(self.iterations):
            # Select a node to expand
            node = self.select(root)
            
            # If we reached depth limit, go back to selection
            if self.get_depth(node) >= self.depth_limit:
                continue
                
            # Generate error feedback and create new prompts
            if not node.is_fully_expanded() and node.visits > 0:
                node = self.expand(node)
            
            # Simulation: evaluate the new prompt
            reward = self.simulate(node, eval_dataset)
            
            # Backpropagate the results
            self.backpropagate(node, reward)
        
        # Return the best prompt (highest average reward)
        best_child = None
        best_value = float('-inf')
        
        for child in root.children:
            child_value = child.reward / child.visits
            if child_value > best_value:
                best_value = child_value
                best_child = child
        
        return best_child.prompt if best_child else root.prompt
    
    def select(self, node):
        """
        Select a node to expand using the UCB formula.
        Traverse the tree until reaching a node that has untried actions or is a leaf.
        """
        current = node
        depth = 0
        
        # While node is fully expanded and not a leaf
        while current.is_fully_expanded() and current.children and depth < self.depth_limit:
            current = current.best_child(self.exploration_weight)
            depth += 1
        
        # If this node hasn't been visited yet, initialize its untried actions
        if current.visits == 0:
            # Generate potential actions (error feedback) for this node
            error_samples = self.get_error_samples(current.prompt)
            current.untried_actions = error_samples
        
        return current
    
    def expand(self, node):
        """
        Expand the node by adding a child node with a new prompt.
        The new prompt is generated based on an untried action (error feedback).
        """
        if not node.untried_actions:
            return node
        
        # Pick a random untried action
        action = node.untried_actions.pop(0)
        
        # Generate a new prompt based on error feedback
        new_prompt = self.generate_new_prompt(node.prompt, action)
        
        # Add the new child node
        child = node.add_child(new_prompt, action)
        
        return child
    
    def simulate(self, node, eval_dataset):
        """
        Evaluate the prompt using classification accuracy on the evaluation dataset.
        """
        # Get prompt
        prompt = node.prompt
        
        # Run the model with this prompt on the evaluation dataset
        accuracy = self.evaluate_prompt(prompt, eval_dataset)
        
        return accuracy
    
    def backpropagate(self, node, reward):
        """
        Update statistics for all nodes from the selected node to the root.
        """
        current = node
        while current:
            current.update(reward)
            current = current.parent
    
    def get_depth(self, node):
        """Calculate the depth of a node in the tree."""
        depth = 0
        current = node
        while current.parent:
            depth += 1
            current = current.parent
        return depth
    
    def get_error_samples(self, prompt):
        """
        Sample data points where the model makes errors with the current prompt.
        Use these as the basis for generating error feedback.
        """
        # This would use the task and model to sample misclassified examples
        # For this simplified version, we'll just return dummy actions
        return [f"Error sample {i}" for i in range(self.expand_width)]
    
    def generate_new_prompt(self, prompt, error_feedback):
        """
        Generate a new prompt based on error feedback.
        In a real implementation, this would use the optimization model to create
        a new prompt that addresses the errors.
        """
        # This is a simplified placeholder - in real implementation, 
        # we would use the language model to generate a new prompt
        return f"{prompt} (improved based on: {error_feedback})"
    
    def evaluate_prompt(self, prompt, eval_dataset):
        """
        Evaluate a prompt by measuring its classification accuracy on the evaluation dataset.
        """
        # Run the model on the dataset with the given prompt
        correct = 0
        total = len(eval_dataset)
        
        for example in eval_dataset:
            # In a real implementation, we would:
            # 1. Format the input with the prompt and example
            # 2. Run the model to get a prediction
            # 3. Check if the prediction matches the label
            
            # For this simplified version, we'll simulate accuracy
            # In real implementation, we would actually run the model and count correct predictions
            # prediction = self.model.predict(prompt, example)
            # is_correct = self.task.is_correct(prediction, example['label'])
            # if is_correct:
            #     correct += 1
            
            # Simulated accuracy for illustration
            is_correct = np.random.random() > 0.5  # Simulated prediction
            if is_correct:
                correct += 1
        
        accuracy = correct / total
        return accuracy


# agent.py

class PromptOptimizationAgent:
    """
    Agent that uses MCTS to optimize prompts for a classification task.
    """
    def __init__(self, task, pred_model, optim_model, config):
        self.task = task
        self.pred_model = pred_model  # Model for prediction (to evaluate prompts)
        self.optim_model = optim_model  # Model for optimization (to generate new prompts)
        self.config = config
        
    def optimize(self, initial_prompt, train_data, eval_data):
        """
        Run the prompt optimization process using MCTS.
        """
        mcts = MCTS(
            task=self.task,
            model=self.pred_model,
            iterations=self.config.iteration_num,
            exploration_weight=self.config.exploration_weight,
            depth_limit=self.config.depth_limit,
            expand_width=self.config.expand_width
        )
        
        optimized_prompt = mcts.search(initial_prompt, eval_data)
        
        return optimized_prompt