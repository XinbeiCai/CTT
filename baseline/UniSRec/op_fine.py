import argparse
import optuna
import torch
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, init_logger, get_trainer, set_color
from unisrec import UniSRec
from data.dataset import UniSRecDataset
from logging import getLogger

fixed_params = {
    'n_layers': 2,
    'n_heads': 2,
    'hidden_size': 300,
    'inner_size': 256,
    'adaptor_layers': [768, 300],
    'plm_size': 768,
    'n_exps': 8,
    'temperature': 0.07,
    'adaptor_dropout_prob': 0.2,
    'lambda': 1e-3,
    'item_drop_ratio': 0.2,
    'item_drop_coefficient': 0.5,
    'initializer_range': 0.02,
    'layer_norm_eps': 1e-12
}


def finetune(dataset, pretrained_file, fix_enc=True, **kwargs):
    # configurations initialization
    props = ['props/UniSRec.yaml', 'props/finetune.yaml']
    print(props)

    # configurations initialization
    config = Config(model=UniSRec, dataset=dataset, config_file_list=props, config_dict=kwargs)
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = UniSRecDataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = UniSRec(config, train_data.dataset).to(config['device'])

    # Load pre-trained model
    if pretrained_file:
        checkpoint = torch.load(pretrained_file)
        logger.info(f'Loading from {pretrained_file}')
        logger.info(f'Transfer [{checkpoint["config"]["dataset"]}] -> [{dataset}]')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        if fix_enc:
            logger.info('Fixing encoder parameters.')
            for param in model.position_embedding.parameters():
                param.requires_grad = False
            for param in model.trm_encoder.parameters():
                param.requires_grad = False
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

    logger.info(set_color('Best valid score: ', 'yellow') + f'{best_valid_result}')
    logger.info(set_color('Test result: ', 'yellow') + f'{test_result}')

    weight_file_path = kwargs.get('weight_file_path', 'Movies_and_TV.pth')

    torch.save(model.state_dict(), weight_file_path)
    logger.info(f'Model weights saved to {weight_file_path}')

    return best_valid_score


def objective(trial):
    params = {
        'hidden_dropout_prob': trial.suggest_uniform('hidden_dropout_prob', 0.1, 0.5),
        'attn_dropout_prob': trial.suggest_uniform('attn_dropout_prob', 0.1, 0.5)
    }

    params.update(fixed_params)

    best_valid_score = finetune(args.d, pretrained_file=args.p, fix_enc=args.f, **params)

    return best_valid_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='Movies_and_TV', help='Dataset name')
    parser.add_argument('-p', type=str, default='saved/UniSRec-FHCKM-300.pth', help='Pre-trained model path')
    parser.add_argument('-f', type=bool, default=True, help='Fix encoder parameters')
    parser.add_argument('-n_trials', type=int, default=50, help='Number of optimization trials')
    parser.add_argument('-timeout', type=int, default=3600, help='Time limit for the optimization in seconds')
    args, unparsed = parser.parse_known_args()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
