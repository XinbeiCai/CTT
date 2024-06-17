import os
import time
import torch
import argparse
from tqdm import tqdm

from model import SASRec, SSL
from utils import *
import optuna


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    os.environ['PYTHONHASHSEED'] = str(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--target_domain', required=True)
parser.add_argument('--source_domain', required=True)
parser.add_argument('--train_dir', default="default")
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=500, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--alpha', default=0.5, type=float)
parser.add_argument('--beta', default=1.0, type=float)
parser.add_argument('--gamma', default=0.25, type=float)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.001, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)

SEED = 2020
setup_seed(SEED)

args = parser.parse_args()



def train_model(args):
    if not os.path.isdir(args.target_domain + '_' + args.train_dir):
        os.makedirs(args.target_domain + '_' + args.train_dir)
    with open(os.path.join(args.target_domain + '_' + args.train_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
    f.close()

    dataset = data_partition(args.target_domain, args.source_domain)

    [user_train1, user_valid1, user_test1, usernum, itemnum1, user_neg1, user_train2, user_valid2, user_test2, itemnum2,
     time1, time2] = dataset
    num_batch = len(user_train1) // args.batch_size
    cc = 0.0
    for u in user_train1:
        cc += len(user_train1[u])
    print('average sequence length: %.2f' % (cc / len(user_train1)))
    print(usernum, '--', itemnum1)

    f = open(os.path.join(args.target_domain + '_' + args.train_dir, 'logs_0.4.txt'), 'w')

    sampler = WarpSampler(user_train1, user_train2, time1, time2, usernum, itemnum1, batch_size=args.batch_size,
                          maxlen=args.maxlen, n_workers=3)
    model = SASRec(user_train1, usernum, itemnum1 + itemnum2, args).to(
        args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass

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

            pdb.set_trace()

    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))

    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    T = 0.0
    t0 = time.time()

    with tqdm(total=args.num_epochs * num_batch, desc=f"Training", leave=False, ncols=100) as pbar:
        for epoch in range(epoch_start_idx, args.num_epochs + 1):
            if args.inference_only: break  # just to decrease identition
            # for step in tqdm(range(num_batch), desc=f"Epoch {epoch}/{args.num_epochs}", leave=True, ncols=100):
            for step in range(num_batch):
                pbar.update(1)
                pbar.set_postfix({"Epoch": epoch})
                u, seq, pos, neg, seq2, mask = sampler.next_batch()
                u, seq, pos, neg, seq2, mask = np.array(u), np.array(seq), np.array(pos), np.array(neg), np.array(
                    seq2), np.array(mask)

                pos_logits, neg_logits, con_loss, con_loss2, con_loss3 = model(u, seq, seq2, pos, neg, mask)
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape,
                                                                                                       device=args.device)

                adam_optimizer.zero_grad()
                indices = np.where(pos != 0)
                loss1 = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss1 += bce_criterion(neg_logits[indices], neg_labels[indices])

                loss = loss1 + args.alpha * con_loss + args.beta * con_loss2 + args.gamma * con_loss3
                for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
                loss.backward()

                adam_optimizer.step()
                # print("loss in epoch {} iteration {}: {}".format(epoch, step,
                #                                                  loss.item()))

            if epoch % 50 == 0 or (epoch % 10 == 0 and epoch > 450):
                model.eval()
                t1 = time.time() - t0
                T += t1
                print('Evaluating', end='')
                t_test = evaluate(model, dataset, args)
                t_valid = evaluate_valid(model, dataset, args)
                print(
                    'epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f,NDCG@5: %.4f, HR@5: %.4f), test (NDCG@10: %.4f, HR@10: %.4f, NDCG@5: %.4f, HR@5: %.4f)'
                    % (epoch, T, t_valid[0], t_valid[1], t_valid[2], t_valid[3], t_test[0], t_test[1], t_test[2],
                       t_test[3]))

                f.write(str(t_valid) + ' ' + str(t_test) + '\n')
                f.flush()
                t0 = time.time()
                model.train()

            if epoch == args.num_epochs:
                folder = args.target_domain + '_' + args.train_dir
                fname = 'MGCL.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units,
                                     args.maxlen)
                torch.save(model.state_dict(), os.path.join(folder, fname))
                return t_valid[1]

    f.close()
    sampler.close()
    print("Done")




def objective(trial, args):
    optuna_args = argparse.Namespace(
        hidden_units=trial.suggest_categorical('hidden_units', [16, 32, 64, 128, 256]),
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

