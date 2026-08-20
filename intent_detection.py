import os
from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np

def main():
    # 1. Define sample intent dataset (Few-shot example)
    # Intent mapping: 0 = greeting, 1 = book_flight, 2 = customer_support
    train_data = {
        "text": [
            "Hello there!", "Hi, good morning", "Hey, anyone available?",
            "I need to book a flight to New York", "Can I reserve a plane ticket?", "Look for flights to London next Tuesday",
            "My account is locked", "I need help with my password", "Can I speak to support?"
        ],
        "label": [0, 0, 0, 1, 1, 1, 2, 2, 2]
    }
    
    test_data = {
        "text": [
            "Hey", 
            "Search flights to Paris", 
            "I can't log into my profile"
        ],
        "label": [0, 1, 2]
    }

    # Convert to Hugging Face Dataset format
    train_dataset = Dataset.from_dict(train_data)
    test_dataset = Dataset.from_dict(test_data)

    # 2. Load a lightweight Sentence Transformer model from Hugging Face
    # 'all-MiniLM-L6-v2' is only ~90MB and highly accurate for semantic similarity
    print("Loading lightweight model...")
    model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # 3. Configure lightweight training arguments for laptop CPU execution
    args = TrainingArguments(
        batch_size=4,
        num_epochs=1,  # 1 epoch is usually enough for SetFit contrastive learning
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
    )

    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # 5. Train the model
    print("Training intent classifier...")
    trainer.train()

    # 6. Evaluate accuracy
    metrics = trainer.evaluate()
    print(f"\nEvaluation Metrics: {metrics}")

    # 7. Test inference
    label_map = {0: "greeting", 1: "book_flight", 2: "customer_support"}
    unseen_phrases = [
        "Good afternoon!",
        "Are there any available seats to Tokyo tomorrow?",
        "Reset my login pin code please"
    ]
    
    print("\n--- Running Inference ---")
    preds = model.predict(unseen_phrases)
    for phrase, pred in zip(unseen_phrases, preds):
        print(f"Phrase: '{phrase}' -> Intent: {label_map[int(pred)]}")

if __name__ == "__main__":
    main()
