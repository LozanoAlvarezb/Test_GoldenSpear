import os
from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    EvalPrediction
)

from datasets import (
    load_metric,
    load_from_disk
)

import numpy as np

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import wandb

import logging
logger = logging.getLogger(__name__)

# def get_data():
#     try:
        
#         return dataset

#     except Exception as error:
#         logger.exception("Coudn't load dataset")



def compute_metrics(p: EvalPrediction):
  """
  Calcula las metricas de la lista metrics.
  Cada predicción de los modelos [MODEL_NAME]ForSequenceClassification es un 
  array de dimensiones == n_labels. Para compararlo con los labels se clasifica
  cada ejemplo en base a la clase mas probable 
  """
  acc = load_metric("accuracy")
  preds = np.argmax(p.predictions, axis=1)
  return acc.compute(predictions=preds, references=p.label_ids)


@hydra.main(config_path='configs', config_name='default')
def train(cfg: DictConfig):

    # wandb.init(config=cfg)
    # # Access all hyperparameter values through wandb.config
    # cfg = wandb.config

    # Simple logging of the configuration
    logger.info(OmegaConf.to_yaml(cfg))

    args = TrainingArguments(**cfg.training)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.checkpoint)
    model_pretrained = AutoModelForSequenceClassification.from_pretrained(cfg.model.checkpoint, num_labels=5)

    dataset = load_from_disk(f"{get_original_cwd()}/data")
    encoded_dataset = dataset.map(lambda example: tokenizer(example["text"],truncation=True), batched=True)

    trainer = Trainer(
        model_pretrained,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["dev"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    predictions = trainer.predict(encoded_dataset["test"])
    

    # Log confusion matrix in different formats
    wandb.log({"test/confusion_mat" : wandb.plot.confusion_matrix(probs=None,
                            preds=predictions.label_ids, y_true=np.argmax(predictions.predictions, axis=1),
                            class_names=list(range(1,6)))})
    wandb.log({"test/confusion_mat":wandb.sklearn.confusion_matrix(predictions.label_ids+1, np.argmax(predictions.predictions, axis=1)+1)})




if __name__ == "__main__":
    train()
