from squad_training.models.marco_mlm import MarcoMLM
from dotmap import DotMap


def main():

    args = DotMap()
    args.bert_model_type = 'bert-base-uncased'

    mlm_model = MarcoMLM.load_from_checkpoint(
        checkpoint_path="/home/wallat/squad_mlm/models/mlm_old/_ckpt_epoch_31.ckpt")
    mlm_model.model.save_pretrained(
        '/home/wallat/squad_mlm/models/stable_old/')


if __name__ == '__main__':
    main()
