# tasks/classification_task.py

class ClassificationTask:
    """
    A base class for classification tasks used with the MCTS prompt optimization.
    """
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.train_data = None
        self.eval_data = None
        self.test_data = None
        
    def load_task_dataset(self, train_size, eval_size, test_size, shuffle=True, seed=42):
        """
        Load and split the dataset into train, eval, and test sets.
        """
        # Load data from the dataset path
        # In a real implementation, this would load from JSON or other format
        
        # For this simplified version, we'll create dummy data
        all_data = self._load_dummy_data()
        
        # Split the data
        if shuffle:
            import random
            random.seed(seed)
            random.shuffle(all_data)
        
        self.train_data = all_data[:train_size]
        self.eval_data = all_data[train_size:train_size+eval_size]
        self.test_data = all_data[train_size+eval_size:train_size+eval_size+test_size]
        
        return self.train_data, self.eval_data, self.test_data
    
    def _load_dummy_data(self):
        """Create dummy data for demonstration."""
        return [
            {"text": f"Example {i}", "label": i % 2} 
            for i in range(200)
        ]
    
    def build_forward_prompts(self, prompt, examples):
        """
        Format inputs for the model using the given prompt and examples.
        """
        formatted_inputs = []
        for example in examples:
            formatted_input = f"{prompt}\n\nText: {example['text']}\n\nLabel:"
            formatted_inputs.append((formatted_input, example))
        return formatted_inputs
    
    def clean_response(self, response):
        """
        Extract the predicted label from the model's response.
        """
        # In a real implementation, this would parse the model's output
        # For this simplified version, we just return the response
        return response.strip()
    
    def is_correct(self, prediction, label):
        """
        Check if a prediction is correct against the true label.
        """
        # Convert prediction string to the same format as label (e.g., int)
        try:
            prediction_value = int(prediction)
            return prediction_value == label
        except:
            return False
    
    def cal_metric(self, predictions, examples):
        """
        Calculate classification accuracy for a batch of predictions.
        """
        correct = 0
        for pred, example in zip(predictions, examples):
            if self.is_correct(pred, example['label']):
                correct += 1
        return correct / len(examples)


# For a specific classification task like SUBJ (subjectivity classification)
class SUBJClassificationTask(ClassificationTask):
    """
    Implementation for the SUBJ dataset (subjective vs objective classification).
    """
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
    
    def _load_dummy_data(self):
        """
        In a real implementation, this would load the SUBJ dataset.
        For demonstration, we create dummy examples.
        """
        return [
            {"text": f"This is a {'subjective' if i % 2 else 'objective'} statement example {i}", 
             "label": i % 2} 
            for i in range(200)
        ]


# For a specific classification task like SST-2 (sentiment classification)
class SST2ClassificationTask(ClassificationTask):
    """
    Implementation for the SST-2 dataset (sentiment classification).
    """
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
    
    def _load_dummy_data(self):
        """
        In a real implementation, this would load the SST-2 dataset.
        For demonstration, we create dummy examples.
        """
        return [
            {"text": f"This is a {'positive' if i % 2 else 'negative'} sentiment example {i}", 
             "label": i % 2} 
            for i in range(200)
        ]