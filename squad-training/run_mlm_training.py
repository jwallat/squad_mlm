import os
import sys
from argparse import ArgumentParser
from dotmap import DotMap
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.profiler import AdvancedProfiler
from squad_training.models.marco_mlm import MarcoMLM


def main(cli_args):

    seed_everything(24)

    args = DotMap()
    args.bert_model_type = 'bert-base-uncased'
    args.lowercase = 'uncased' in args.bert_model_type
    args.train_file = cli_args.train_file
    args.eval_file = cli_args.eval_file
    # args.test_file = '../data/eval_v2.1_public.json'
    args.learning_rate = cli_args.learning_rate
    args.batch_size = cli_args.batch_size
    args.model_save_path = cli_args.model_save_path
    args.training_epochs = 10
    args.mlm_probability = 0.15

    # ----------------------------------------------------------------------

    mlm_model = MarcoMLM(args)

    checkpoint_callback = ModelCheckpoint(
        filepath=cli_args.model_save_path,  # '/home/wallat/msmarco-models/models/mlm/'
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )

    if cli_args.use_wandb_logging:
        print('If you are having issues with wandb, make sure to give the correct python executable to --python_executable')
        sys.executable = cli_args.python_executable
        logger = WandbLogger(project=cli_args.wandb_project_name,
                             name=cli_args.wandb_run_name)
    else:
        logger = TensorBoardLogger("{}/tb_logs".format(args.output_dir))

    trainer = Trainer.from_argparse_args(
        cli_args, checkpoint_callback=checkpoint_callback, early_stop_callback=True, logger=logger)

    trainer.fit(mlm_model)
    trainer.save_checkpoint(os.path.join(
        cli_args.model_save_path, 'trained_checkpoint'))
    mlm_model.model.save_pretrained(cli_args.model_save_path)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser = Trainer.add_argparse_args(parser)

    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)

    parser.add_argument('--train_file', required=True, default=None)
    parser.add_argument('--eval_file', required=True, default=None)
    parser.add_argument('--model_save_path', required=True, default=None)

    parser.add_argument('--use_wandb_logging', default=False, action='store_true',
                        help='Use this flag to use wandb logging. Otherwise we will use the pytorch-lightning tensorboard logger')
    parser.add_argument('--wandb_project_name', required='--use_wandb_logging' in sys.argv, type=str,
                        help='Name of wandb project')
    parser.add_argument('--wandb_run_name', default='',
                        type=str, help='Name of wandb run')
    parser.add_argument('--python_executable', required='--use_wandb_logging' in sys.argv, type=str, default='/usr/bin/python3',
                        help='Some cluster environments might require to set the sys.executable for wandb to work')

    args = parser.parse_args()

    main(args)
