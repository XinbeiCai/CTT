import os
import argparse
import time
import numpy as np
import torch
import random
from tqdm import tqdm
from utils import data_partition, WarpSampler, evaluate_valid, evaluate_test
from model import Model
import optuna

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


parser = argparse.ArgumentParser(description="Configurations.")
parser.add_argument('--target_domain', default='Movies_and_TV') #, required=True
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--maxlen', type=int, default=50, help='Maximum length of sequences.')
parser.add_argument('--hidden_units', type=int, default=64, help='i.e. latent vector dimensionality.')
parser.add_argument('--num_blocks', type=int, default=2, help='Number of self-attention blocks.')
parser.add_argument('--num_heads', type=int, default=1, help='Number of heads for attention.')
parser.add_argument('--num_layers', type=int, default=2, help='Number of GRU layers.')
parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs.')
parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout Rate.')
parser.add_argument('--l2_emb', type=float, default=0.001)
parser.add_argument('--random_seed', type=int, default=2024)
parser.add_argument('--early_stop_epoch', type=int, default=20)
parser.add_argument('--save_model', type=int, default=1, help='Whether to save the torch model.')
parser.add_argument('--eva_interval', type=int, default=50, help='Number of epoch interval for evaluation.')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--eps', type=float, default=1e-12)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--optimize', type=bool, default=True, help="Use Optuna for hyperparameter optimization")

args = parser.parse_args()
dir = './log/%s' % args.target_domain
if not os.path.isdir(dir):
    os.makedirs(dir)

log_file = os.path.join(dir, 'log_GRU4Rec_%s.txt' % (args.target_domain))



def train_model(args):
    with open(log_file, 'a') as f:
        print('\n--------------------------------------', file=f)
        print('args', file=f)
        print('\n'.join([str(k) + ': ' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]),
              file=f)
        print('\n'.join([str(k) + ': ' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
        print('--------------------------------------\n', file=f)

    dataset = data_partition(args.target_domain)
    [user_train_target_domain, user_valid_target_domain, user_test_target_domain,
     usernum, itemnum_target_domain,
     user_neg_target_domain] = dataset

    num_batch = len(user_train_target_domain) // args.batch_size

    # seed random seed
    setup_seed(args.random_seed)

    sampler = WarpSampler(user_train_target_domain, usernum, itemnum_target_domain,
                          batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)

    model = Model(itemnum_target_domain, args).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass  # just ignore those failed init layers

    model.train()
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb;

    if args.inference_only:
        model.eval()
        t_test = evaluate_test(model, dataset, args)
        print('test (HR@10: %.4f, NDCG@10: %.4f, HR@5: %.4f, NDCG@5: %.4f)' % (
        t_test[1], t_test[0], t_test[3], t_test[2]))

    bce_criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    T = 0.0
    t0 = time.time()

    with tqdm(total=args.num_epochs * num_batch, desc=f"Training", leave=False, ncols=100) as pbar:
        for epoch in range(epoch_start_idx, args.num_epochs + 1):
            if args.inference_only: break  # just to decrease identition
            for step in range(num_batch):
                pbar.update(1)
                pbar.set_postfix({"Epoch": epoch})

                [u, seq_target, pos, neg] = [np.array(result) for result in sampler.next_batch()]
                pos_logits_sasrec, neg_logits_sasrec = model(seq_target, pos, neg)
                pos_labels = torch.ones(pos_logits_sasrec.shape, device=args.device)
                neg_labels = torch.zeros(neg_logits_sasrec.shape, device=args.device)
                adam_optimizer.zero_grad()
                indices = np.where(pos != 0)

                loss = bce_criterion(pos_logits_sasrec[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits_sasrec[indices], neg_labels[indices])

                for param in model.item_emb.parameters():
                    loss += args.l2_emb * torch.norm(param)
                loss.backward()
                adam_optimizer.step()

            # valid
            if epoch % args.eva_interval == 0:
                model.eval()
                t1 = time.time() - t0
                T += t1
                t_valid = evaluate_valid(model, dataset, args)
                print('epoch:%d, valid ( HR@10: %.4f, NDCG@10: %.4f, HR@5: %.4f, NDCG@5: %.4f)), time: %f(s)' %
                      (epoch, t_valid[1], t_valid[0], t_valid[3], t_valid[2], T))

                with open(log_file, 'a') as f:
                    print('epoch:%d, valid ( HR@10: %.4f, NDCG@10: %.4f, HR@5: %.4f, NDCG@5: %.4f)), time: %f(s)' %
                          (epoch, t_valid[1], t_valid[0], t_valid[3], t_valid[2], T), file=f)

                t_test = evaluate_test(model, dataset, args)

                print('test (HR@10: %.4f, NDCG@10: %.4f, HR@5: %.4f, NDCG@5: %.4f)' %
                      (t_test[1], t_test[0], t_test[3], t_test[2]))

                with open(log_file, 'a') as f:
                    print('test (HR@10: %.4f, NDCG@10: %.4f, HR@5: %.4f, NDCG@5: %.4f)' %
                          (t_test[1], t_test[0], t_test[3], t_test[2]), file=f)

                t0 = time.time()

                model.train()
                #
                if epoch == args.num_epochs and args.save_model == 1:
                    fname = 'GRU4Rec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                    fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads,
                                         args.hidden_units,
                                         args.maxlen)
                    torch.save(model.state_dict(), os.path.join(dir, fname))

                    return t_valid[1]

    sampler.close()


def objective(trial, args):
    optuna_args = argparse.Namespace(
        hidden_units=trial.suggest_categorical('hidden_units', [16, 32, 64, 128, 256]),
        num_layers=trial.suggest_categorical('num_layers', [1, 2, 3]),
        batch_size=trial.suggest_categorical('batch_size', [64, 128, 256]),
        maxlen=trial.suggest_categorical('maxlen', [50, 60, 70, 80, 90, 100]),
        dropout_rate=trial.suggest_categorical('dropout_rate', [0.3, 0.4, 0.5]),
        l2_emb = trial.suggest_float('l2_emb', 1e-3, 1e-2),
    )

    args_dict = vars(args)
    args_dict.update(vars(optuna_args))
    args = argparse.Namespace(**args_dict)

    best_score = train_model(args)

    return best_score


if __name__ == '__main__':
    if args.optimize:
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, args), n_trials=50)
        print("Best trial:")
        print(study.best_trial.params)
    else:
        train_model(args)

    print("Done")

