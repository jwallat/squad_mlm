from dotmap import DotMap
from pytorch_lightning import Trainer
from squad_training.models.marco_mlm import MarcoMLM


def main():
    args = DotMap()
    args.bert_model_type = 'bert-base-uncased'
    args.lowercase = 'uncased' in args.bert_model_type
    args.train_file = '../data/train-v2.0.json'
    args.eval_file = '../data/dev-v2.0.json'
    # args.test_file = '../data/eval_v2.1_public.json'
    args.learning_rate = 5e-5
    args.batch_size = 1
    args.training_epochs = 1
    args.mlm_probability = 0.15

    # General flow is

    # Instantiate lightning model
    # Todo:
    ''' 
    Load dataset files
    Write marco_cloze dataset
    lightning module
    trainer
    '''
    mlm_model = MarcoMLM(args)

    # Do lightning trainer with the prefered args
    trainer = Trainer(gups=1, fast_dev_run=False, max_epochs=args.training_epochs,
                      progress_bar_refresh_rate=1, profiler=True)

    trainer.fit(mlm_model)


if __name__ == '__main__':
    main()
