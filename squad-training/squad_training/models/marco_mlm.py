import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelWithLMHead, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl
from squad_training.datasets.mlm_dataset import MLMDataset
from squad_training.datasets.mlm_dataset_utils import collate, mask_tokens


class MarcoMLM(pl.LightningModule):

    def __init__(self, args):
        super(MarcoMLM, self).__init__()

        self.args = args

        self.model = AutoModelWithLMHead.from_pretrained(
            self.args.bert_model_type)
        self.model.train()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.bert_model_type)

        self.epoch_counter = 0

    def forward(self, input_ids, masked_lm_labels, attention_masks):

        # Just feed stuff into Bert and return the given loss
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_masks, masked_lm_labels=masked_lm_labels)
        # print('ouputus: ', outputs)
        loss = outputs[0]

        return loss

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_masks = batch['attention_masks']

        input_ids, masked_lm_labels = mask_tokens(
            input_ids, self.tokenizer, self.args)
        loss = self(input_ids=input_ids, masked_lm_labels=masked_lm_labels,
                    attention_masks=attention_masks)

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_masks = batch['attention_masks']

        input_ids, masked_lm_labels = mask_tokens(
            input_ids, self.tokenizer, self.args)
        loss = self(input_ids=input_ids, masked_lm_labels=masked_lm_labels,
                    attention_masks=attention_masks)
        # print(loss)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}

        # Save model checkpoint
        self.epoch_counter = self.epoch_counter + 1
        epoch_save_dir = '{}{}/'.format(self.args.model_save_path,
                                        self.epoch_counter)
        os.mkdir(epoch_save_dir)
        self.model.save_pretrained(epoch_save_dir)

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    # def test_step(self, batch, batch_idx):
    #     input_ids = batch['input_ids']
    #     attention_masks = batch['attention_masks']

    #     input_ids, masked_lm_labels = mask_tokens(input_ids, self.tokenizer, self.args)
    #     loss = self(input_ids=input_ids, masked_lm_labels=masked_lm_labels, attention_masks=attention_masks)

    #     return {'test_loss': loss}

    # def test_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
    #     tensorboard_logs = {'test_loss': avg_loss}
    #     return {'test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        adam = AdamW([p for p in self.parameters() if p.requires_grad],
                     lr=self.args.learning_rate, eps=1e-08)

        scheduler = get_linear_schedule_with_warmup(
            adam, num_warmup_steps=(17800), num_training_steps=17810*10)

        return [adam], [{"scheduler": scheduler, "interval": "step"}]

    def prepare_data(self):
        self.train_dataset = MLMDataset(
            self.tokenizer, self.args.train_file, self.args)
        self.eval_dataset = MLMDataset(
            self.tokenizer, self.args.eval_file, self.args)
        # self.test_dataset = MLMDataset(
        #     self.tokenizer, self.args.test_file, self.args)

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.args.batch_size, collate_fn=collate)

        return train_dataloader

    def val_dataloader(self):
        eval_dataloader = DataLoader(
            self.eval_dataset, batch_size=self.args.batch_size, collate_fn=collate)

        return eval_dataloader

    # def test_dataloader(self):
    #     test_dataloader = DataLoader(
    #         self.test_dataset, batch_size=self.args.batch_size, collate_fn=collate)

    #     return test_dataloader
