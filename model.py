from datasets import load_dataset
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Using a global variable for generator (which will be imported
# by the app). Bad practice.
generator = None


tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
tokenizer.pad_token = tokenizer.eos_token

def my_tokenize(text):
  # Tokenize the text
  tokenized_text = tokenizer(text['text'], truncation=True, padding="max_length", max_length=512)

  # Put them under labels
  tokenized_text["labels"] = tokenized_text["input_ids"].copy()
  return tokenized_text

def prepare_model():

    # Using the anthropic synthetic dataset to train.
    dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    df = pd.DataFrame(dataset)

    # For now, let us drop the rejected dataset,
    # Let us quickly train on the chosen dataset.
    df.drop('rejected', inplace=True, axis=1)

    # Rename 'chosen' to 'text', something generic.
    df = df.rename(columns={'chosen': 'text'})

    # Get the dataset ready (tokenize it)
    small_dataset = Dataset.from_pandas(df.head(1000))
    tokenized_small_dataset = small_dataset.map(my_tokenize, batched=True)

    # Using tiny-gpt2 model
    model = AutoModelForCausalLM.from_pretrained("./model")

    # Go for training
    training_args = TrainingArguments(
        output_dir="./tiny-gpt2-finetuned-convo",
        per_device_train_batch_size=8,
        num_train_epochs=2,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=100,
        fp16=False,  # Enable if you use GPU
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_small_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

