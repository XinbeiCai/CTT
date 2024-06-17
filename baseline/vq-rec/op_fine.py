import argparse
from logging import getLogger
import optuna
import torch
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, init_logger, set_color

from vqrec import VQRec
from utils import create_dataset
from trainer import VQRecTrainer

# 固定参数
fixed_params = {
    'n_layers': 2,
    'n_heads': 2,
    'inner_size': 256,
    'loss_type': 'CE',
    'hidden_act': 'gelu',
    'layer_norm_eps': 1e-12,
    'hidden_size': 300,
    'plm_size': 768,
    'sinkhorn_iter': 3,
    'reassign_steps': 5,
    'eval_batch_size': 1024,
    'code_dim': 32,
    'code_cap': 256,
    'initializer_range': 0.02,
    'temperature': 0.07,
    'fake_idx_ratio': 0.75
}

def finetune(model_name, dataset, pretrained_file='', finetune_mode='', **kwargs):
    # configurations initialization
    props = [f'props/{model_name}.yaml', 'props/finetune.yaml']
    print(props)

    # configurations initialization
    config = Config(model=VQRec, dataset=dataset, config_file_list=props, config_dict=kwargs)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = VQRec(config, train_data.dataset).to(config['device'])
    model.pq_codes = model.pq_codes.to(config['device'])

    # Load pre-trained model
    if pretrained_file != '':
        checkpoint = torch.load(pretrained_file)
        logger.info(f'Loading from {pretrained_file}')
        logger.info(f'Transfer [{checkpoint["config"]["dataset"]}] -> [{dataset}]')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        if finetune_mode == 'fix_enc':
            logger.info('[Fine-tune mode] Fix Seq Encoder!')
            for param in model.position_embedding.parameters():
                param.requires_grad = False
            for param in model.trm_encoder.parameters():
                param.requires_grad = False
    logger.info(model)

    # trainer loading and initialization
    trainer = VQRecTrainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return best_valid_score

def objective(trial):
    params = {
        'hidden_dropout_prob': trial.suggest_uniform('hidden_dropout_prob', 0.1, 0.5),
        'attn_dropout_prob': trial.suggest_uniform('attn_dropout_prob', 0.1, 0.5)
    }

    params.update(fixed_params)

    best_valid_score = finetune(args.m, args.d, pretrained_file=args.p, finetune_mode=args.f, **params)

    return best_valid_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, default='VQRec', help='Model name')
    parser.add_argument('-d', type=str, default='CDs_and_Vinyl', help='Dataset name')
    parser.add_argument('-p', type=str, default='pretrained/VQRec-FHCKM-300-20230315.pth',
                        help='Pre-trained model path')
    parser.add_argument('-f', type=str, default='fix_enc', help='Fine-tune mode')
    parser.add_argument('-n_trials', type=int, default=50, help='Number of optimization trials')
    parser.add_argument('-timeout', type=int, default=3600, help='Time limit for the optimization in seconds')
    args, unparsed = parser.parse_known_args()
    print(args)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
