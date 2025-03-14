import numpy as np
import pandas as pd 
from transformers import BertTokenizer, Trainer, BertForSequenceClassification, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification, ModernBertForSequenceClassification
from datasets import Dataset
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# tested in transformers==4.18.0, pytorch==1.7.1 
import torch
import transformers
torch.__version__, transformers.__version__

torch.cuda.is_available()


df = pd.read_csv('financial_phrases_all_updated.txt', delimiter='\t', names=['sentence', 'label', 'sentiment'])
df.head()

print(df['label'].value_counts())
df_train = df[df['label'] == 'train']
df_test = df[df['label'] == 'test']
df_train, df_val = train_test_split(df_train, stratify=df_train['sentiment'], test_size=0.1, random_state=42)

model = ModernBertForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base",num_labels=3)
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base", max_length=512)

dataset_train = Dataset.from_pandas(df_train)
dataset_val = Dataset.from_pandas(df_val)
dataset_test = Dataset.from_pandas(df_test)

dataset_train = dataset_train.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=128), batched=True)
dataset_val = dataset_val.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=128), batched=True)
dataset_test = dataset_test.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length' , max_length=128), batched=True)

dataset_train = dataset_train.remove_columns(['label']).rename_column('sentiment', 'label')
dataset_val = dataset_val.remove_columns(['label']).rename_column('sentiment', 'label')
dataset_test = dataset_test.remove_columns(['label']).rename_column('sentiment', 'label')

print(dataset_train)

dataset_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label']) #dataset_train.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
dataset_val.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
dataset_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy' : accuracy_score(predictions, labels)}

args = TrainingArguments(
        output_dir = 'temp/',
        evaluation_strategy = 'epoch',
        save_strategy = 'epoch',
        learning_rate=3e-4,
        per_device_train_batch_size=72,
        per_device_eval_batch_size=72,
        num_train_epochs=20,
        weight_decay=1e-5,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy'
)

trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=args,                  # training arguments, defined above
        train_dataset=dataset_train,         # training dataset
        eval_dataset=dataset_val,            # evaluation dataset
        compute_metrics=compute_metrics
)

trainer.train()   

model.eval()
trainer.predict(dataset_test).metrics

trainer.save_model('ModernBERT-sentiment/')