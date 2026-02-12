import argparse
import os
import csv
from utils.logger import Logger
from data_process.dataset import FedDatasetCV, get_dataset_mat
# import sys
import numpy as np
import torch
import wandb
import scipy.io as io
# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./")))
from models.fpnn import *
from models.fpnn_fedavg_api import FedAvgAPI
from models.fpnn_fed_trainer import MyModelTrainer as MyModelTrainerFPNN


from data.custom_partiton import load_multiple_fold_files

def add_args(p_parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    p_parser.add_argument('--model', type=str, default='fed_fpnn', metavar='N',
                          help='neural network used in training')

    p_parser.add_argument('--dataset', type=str, default='iris', metavar='N',  # FIXME: gsad, wine
                          help='dataset used for training')

    p_parser.add_argument('--fs', type=str, default='sum', metavar='N',
                          help='firing strength layer')

    p_parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                          help='how to partition the dataset on local workers')

    p_parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                          help='partition alpha (default: 0.5)')

    p_parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                          help='input batch size for training (default: 64)')

    p_parser.add_argument('--optimizer', type=str, default='adam',
                          help='SGD with momentum; adam')

    p_parser.add_argument('--criterion', type=str, default='bce',
                          help='the loss function')

    p_parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                          help='learning rate (default: 0.001)')

    p_parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    p_parser.add_argument('--epochs', type=int, default=15, metavar='EP',
                          help='how many epochs will be trained locally')

    p_parser.add_argument('--n_client', type=int, default=5, metavar='NN',
                          help='number of workers in a distributed cluster')

    p_parser.add_argument('--n_client_per_round', type=int, default=5, metavar='NN',
                          help='number of workers')

    p_parser.add_argument('--comm_round', type=int, default=100,
                          help='how many round of communications we shoud use')

    p_parser.add_argument('--milestone', type=int, default=10,
                          help='the tag that local rules suit their own rules after certain round of communications')

    p_parser.add_argument('--frequency_of_the_test', type=int, default=1,
                          help='the frequency of the algorithms')

    p_parser.add_argument('--gpu', type=int, default=0,
                          help='gpu')

    p_parser.add_argument('--n_rule', type=int, default=10,
                          help='rule number')
    p_parser.add_argument('--n_rule_min', type=int, default=10,
                          help='rule number')

    p_parser.add_argument('--n_kernel', type=int, default=5,
                          help='Cov kernel number')

    p_parser.add_argument('--hidden_dim', type=int, default=512,
                          help='the output dim of the EEG encoder')

    p_parser.add_argument('--dropout', type=float, default=0.25, metavar='DR',
                          help='dropout rate (default: 0.025)')
    p_parser.add_argument(
        "--nl",
        type=float,
        default=0.0,
        help="noise level on dataset corruption",
    )
    p_parser.add_argument('--alpha', default=0., type=float,
                          help='mixup interpolation coefficient (default: 1)')

    p_parser.add_argument('--n_kfolds', type=int, default=10,
                          help='The number of k_fold cross validation')

    p_parser.add_argument('--f1_average', type=str, default='macro',
                          help='F1 averaging method (macro, micro, weighted)')

    p_parser.add_argument('--partition_dir', type=str, default='data',
                          help='Base directory containing dataset subfolders')
    p_parser.add_argument('--partition_dataset', type=str, default=None,
                          help='Dataset folder/name override for partition files')

    p_parser.add_argument('--b_debug', type=int, default=1,
                          help='set 1 to change to debug mode')

    p_parser.add_argument('--b_norm_dataset', type=int, default=1,
                          help='set 1 to change to debug mode')

    p_args = p_parser.parse_args()
    return p_args


def create_model(p_args):
    p_args.logger.info("======create_model.======")
    p_model: torch.nn.Module = None
    if p_args.model == "fpnn":
        local_rule_idxs = np.arange(p_args.n_rule)
        p_model: torch.nn.Module = FPNN(p_args, local_rule_idxs)
    elif p_args.model == "fed_fpnn":
        # client_idx_list = np.arange(p_args.n_client)
        # initiate the global rule list
        # golobal_rule_list = [RuleR(p_args.n_fea, p_args.n_class, client_idx_list) for _ in range(p_args.n_rule)]
        local_rule_idxs = np.arange(p_args.n_rule)
        # global_fs_layer = FSLayer(p_args.n_fea, p_args.dropout)
        # p_model: torch.nn.Module = FedFPNNR(p_args.n_class, global_fs_layer, local_rule_idxs, golobal_rule_list)
        p_model: torch.nn.Module = FPNN(p_args, local_rule_idxs)

    return p_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter parse of this project")
    args = add_args(parser)
    args.logger = Logger(True, args.dataset, args.model)
    args.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

    # Dataset configuration
    args.logger.info(f"========================={args.model}========================")
    args.logger.info(f"dataset : {args.dataset}")
    args.logger.info(f"device : {args.device}")
    args.logger.info(f"batch size : {args.batch_size}")
    args.logger.info(f"epoch number : {args.epochs}")
    args.logger.info(f"rule number : {args.n_rule}")

    global_train_acc_tsr = torch.zeros(args.comm_round, args.n_kfolds).to(args.device)
    global_train_loss_tsr = torch.zeros(args.comm_round, args.n_kfolds).to(args.device)
    global_test_loss_tsr = torch.zeros(args.comm_round, args.n_kfolds).to(args.device)
    global_test_acc_tsr = torch.zeros(args.comm_round, args.n_kfolds).to(args.device)
    global_train_f1_tsr = torch.zeros(args.comm_round, args.n_kfolds).to(args.device)
    global_test_f1_tsr = torch.zeros(args.comm_round, args.n_kfolds).to(args.device)

    global_train_rule_count = torch.zeros(args.comm_round, args.n_rule, args.n_kfolds).to(args.device)
    global_train_rule_contr = torch.zeros(args.comm_round, args.n_rule, args.n_kfolds).to(args.device)

    local_train_rule_count = torch.zeros(args.comm_round, args.n_client, args.n_rule, args.n_kfolds).to(args.device)
    local_train_rule_contr = torch.zeros(args.comm_round, args.n_client, args.n_rule, args.n_kfolds).to(args.device)

    local_train_acc_tsr = torch.zeros(args.comm_round, args.n_client, args.n_kfolds).to(args.device)
    local_train_loss_tsr = torch.zeros(args.comm_round, args.n_client, args.n_kfolds).to(args.device)
    local_test_loss_tsr = torch.zeros(args.comm_round, args.n_client, args.n_kfolds).to(args.device)
    local_test_acc_tsr = torch.zeros(args.comm_round, args.n_client, args.n_kfolds).to(args.device)
    local_test_f1_tsr = torch.zeros(args.comm_round, args.n_client, args.n_kfolds).to(args.device)

    # Tensors for global test set evaluation (each client on all test data)
    client_global_test_f1_tsr = torch.zeros(args.n_client, args.n_kfolds).to(args.device)
    client_global_test_acc_tsr = torch.zeros(args.n_client, args.n_kfolds).to(args.device)

    from data.load_dataset import read_json_dataset
    from pathlib import Path
    datasetName = args.dataset # "iris"
    alpha = args.partition_alpha
    nclients = args.n_client
    n_rep = args.n_kfolds
    i = 0
    task = "C"  # Classification

    dataset_folder = args.partition_dataset or datasetName
    dataset_base = Path(args.partition_dir) / dataset_folder
    base_prefix = f"{dataset_folder}_a{alpha}_n{nclients}"

    dataset_path = dataset_base / f"{base_prefix}_f{i + 1}-{n_rep}.json"
    client_partitions, data, partitions = read_json_dataset(datasetName, dataset_path)
    inputs = data.data
    targets = data.target
    n_class = len(np.unique(targets))

    partition_files = []
    for i in range(n_rep):
        p = dataset_base / f"{base_prefix}_f{i + 1}-{n_rep}.json"
        partition_files.append(str(p))


    partition = load_multiple_fold_files(partition_files, targets, args)
    args.partition = partition
    dataset = FedDatasetCV(inputs, targets, n_class, task, args.dataset, p_args=args,
                           fed_kfold_partition=partition, require_partition=True)

    for cv_idx in range(args.n_kfolds):
    # for cv_idx in range(1):
        args.cv = cv_idx
        args.logger.war(f"=====k_fold: {cv_idx + 1}=======")

        # load data
        # args.dataset = "wine"
        # dataset: FedDatasetCV = get_dataset_mat(args.dataset, args)
        dataset.set_current_folds(cv_idx)

        # save category number
        args.n_class = dataset.n_class
        args.n_fea = dataset.n_fea
        args.tag = f"{args.model}_{args.n_rule}_{args.n_client}_{args.n_client_per_round}_{args.partition_method}" \
                   f"_{args.partition_alpha}" \
                   f"_{args.nl}_{args.criterion}_{args.lr}"
        # if args.partition_method == 'homo':
        #     args.tag = f"{args.model}_{args.n_rule}_{args.partition_method}_{args.nl}_{args.criterion}_{args.lr}"
        # else:
        #     args.tag = f"{args.model}_{args.n_rule}_{args.n_client}_{args.n_client_per_round}_{args.partition_method}" \
        #             f"_{args.partition_alpha}" \
        #             f"_{args.nl}_{args.criterion}_{args.lr}"
        if not args.b_debug:
            wandb.init(
                project=f"FederatedFPNN-{args.dataset}",
                name=str(args.model) +
                     "-r" + str(args.n_rule) +
                     "-c" + str(args.n_client) +
                     "-p" + str(args.n_client_per_round) +
                     '-' + str(args.partition_method)
                     + str(args.partition_alpha) +
                     "-nl" + str(args.nl) +
                     "-" + args.criterion +
                     "-lr" + str(args.lr) +
                     "-cv" + str(cv_idx + 1),
                config=args
            )

        # create model.
        model = create_model(args)
        model_trainer = MyModelTrainerFPNN(model, args)
        # args.logger.info(model)

        # federated method
        fedavgAPI = FedAvgAPI(dataset, model_trainer, args)
        metrics_list = fedavgAPI.train()

        for commu_idx in range(args.comm_round):
            metrics = metrics_list[commu_idx]
            global_train_acc_tsr[commu_idx, cv_idx] = metrics['training_acc']
            global_train_loss_tsr[commu_idx, cv_idx] = metrics['training_loss']
            global_test_loss_tsr[commu_idx, cv_idx] = metrics['test_loss']
            global_test_acc_tsr[commu_idx, cv_idx] = metrics['test_acc']
            global_train_f1_tsr[commu_idx, cv_idx] = metrics['training_f1']
            global_test_f1_tsr[commu_idx, cv_idx] = metrics['test_f1']
            for rule_idx in torch.arange(args.n_rule):
                global_train_rule_count[commu_idx, rule_idx, cv_idx] = metrics[f"rule{rule_idx + 1}_count"]
                global_train_rule_contr[commu_idx, rule_idx, cv_idx] = metrics[f"rule{rule_idx + 1}_contr"]
                for client_idx in range(args.n_client):
                    local_train_rule_count[commu_idx, client_idx, rule_idx, cv_idx] = \
                        metrics[f"client{client_idx + 1}_rule{rule_idx + 1}_count"]
                    local_train_rule_contr[commu_idx, client_idx, rule_idx, cv_idx] = \
                        metrics[f"client{client_idx + 1}_rule{rule_idx + 1}_contr"]
            for client_idx in range(args.n_client):
                local_train_acc_tsr[commu_idx, client_idx, cv_idx] = metrics[f"client{client_idx + 1}_train_acc"]
                local_train_loss_tsr[commu_idx, client_idx, cv_idx] = metrics[f"client{client_idx + 1}_train_loss"]
                local_test_acc_tsr[commu_idx, client_idx, cv_idx] = metrics[f"client{client_idx + 1}_test_acc"]
                local_test_loss_tsr[commu_idx, client_idx, cv_idx] = metrics[f"client{client_idx + 1}_test_loss"]
                local_test_f1_tsr[commu_idx, client_idx, cv_idx] = metrics[f"client{client_idx + 1}_test_f1"]

        # After training, evaluate each client on the combined global test set
        args.logger.war(f"===== Evaluating clients on global test set for fold {cv_idx + 1} =====")
        global_f1, global_acc = fedavgAPI._evaluate_clients_on_global_testset()
        client_global_test_f1_tsr[:, cv_idx] = global_f1
        client_global_test_acc_tsr[:, cv_idx] = global_acc

        if not args.b_debug:
            wandb.finish()

    save_dict = dict()
    save_dict["global_test_acc_tsr"] = global_test_acc_tsr.cpu().numpy()
    save_dict["global_test_f1_tsr"] = global_test_f1_tsr.cpu().numpy()
    # local_test_acc_personalized_tsr[commu_idx, client_idx, cv_idx] = metrics[f"client{client_idx + 1}_test_acc_personalized"]
    # local_test_loss_personalized_tsr[commu_idx, client_idx, cv_idx] = metrics[f"client{client_idx + 1}_test_loss_personalized"]
    save_dict["global_train_acc_tsr"] = global_train_acc_tsr.cpu().numpy()
    save_dict["global_train_f1_tsr"] = global_train_f1_tsr.cpu().numpy()
    save_dict["global_train_rule_count"] = global_train_rule_count.cpu().numpy()
    save_dict["global_train_rule_contr"] = global_train_rule_contr.cpu().numpy()

    save_dict["local_test_acc_tsr"] = local_test_acc_tsr.cpu().numpy()
    save_dict["local_test_loss_tsr"] = local_test_loss_tsr.cpu().numpy()
    save_dict["local_train_loss_tsr"] = local_train_loss_tsr.cpu().numpy()
    save_dict["local_train_acc_tsr"] = local_train_acc_tsr.cpu().numpy()
    save_dict["local_train_rule_count"] = local_train_rule_count.cpu().numpy()
    save_dict["local_train_rule_contr"] = local_train_rule_contr.cpu().numpy()
    save_dict["local_test_f1_tsr"] = local_test_f1_tsr.cpu().numpy()

    # Add global test evaluation results
    save_dict["client_global_test_f1_tsr"] = client_global_test_f1_tsr.cpu().numpy()
    save_dict["client_global_test_acc_tsr"] = client_global_test_acc_tsr.cpu().numpy()

    final_round = args.comm_round - 1
    final_f1_per_fold = global_test_f1_tsr[final_round].detach().cpu().numpy()
    final_f1_mean = float(final_f1_per_fold.mean())
    final_f1_std = float(final_f1_per_fold.std())

    final_acc_per_fold = global_test_acc_tsr[final_round].detach().cpu().numpy()
    final_acc_mean = float(final_acc_per_fold.mean())
    final_acc_std = float(final_acc_per_fold.std())

    # Calculate global test metrics (averaged across all clients and folds)
    final_f1_global_per_fold = client_global_test_f1_tsr.mean(dim=0).detach().cpu().numpy()
    final_f1_global_mean = float(final_f1_global_per_fold.mean())
    final_f1_global_std = float(final_f1_global_per_fold.std())

    final_acc_global_per_fold = client_global_test_acc_tsr.mean(dim=0).detach().cpu().numpy()
    final_acc_global_mean = float(final_acc_global_per_fold.mean())
    final_acc_global_std = float(final_acc_global_per_fold.std())

    # In this setup, each rule uses all antecedents (n_fea).
    antecedents_mean = float(args.n_fea)
    antecedents_std = 0.0

    # Count how many rules were used per client (nonzero counts) on the final round.
    final_rule_counts = (local_train_rule_count[final_round] > 0).sum(dim=1)
    final_rule_counts_np = final_rule_counts.detach().cpu().numpy().reshape(-1)
    rules_mean = float(final_rule_counts_np.mean())
    rules_std = float(final_rule_counts_np.std())

    save_dict["final_f1_mean"] = final_f1_mean
    save_dict["final_f1_std"] = final_f1_std
    save_dict["final_acc_mean"] = final_acc_mean
    save_dict["final_acc_std"] = final_acc_std
    save_dict["final_f1_global_mean"] = final_f1_global_mean
    save_dict["final_f1_global_std"] = final_f1_global_std
    save_dict["final_acc_global_mean"] = final_acc_global_mean
    save_dict["final_acc_global_std"] = final_acc_global_std
    save_dict["antecedents_mean"] = antecedents_mean
    save_dict["antecedents_std"] = antecedents_std
    save_dict["rules_mean"] = rules_mean
    save_dict["rules_std"] = rules_std

    args.logger.info(f"Final F1 (mean±std over folds): {final_f1_mean:.4f} ± {final_f1_std:.4f}")
    args.logger.info(f"Final Acc (mean±std over folds): {final_acc_mean:.4f} ± {final_acc_std:.4f}")
    args.logger.info(f"Final F1 Global (mean±std over folds): {final_f1_global_mean:.4f} ± {final_f1_global_std:.4f}")
    args.logger.info(f"Final Acc Global (mean±std over folds): {final_acc_global_mean:.4f} ± {final_acc_global_std:.4f}")
    args.logger.info(f"Antecedents per rule (mean±std): {antecedents_mean:.2f} ± {antecedents_std:.2f}")
    print(f"Final F1 (mean±std over folds): {final_f1_mean:.4f} ± {final_f1_std:.4f}")
    print(f"Final Acc (mean±std over folds): {final_acc_mean:.4f} ± {final_acc_std:.4f}")
    print(f"Final F1 Global (mean±std over folds): {final_f1_global_mean:.4f} ± {final_f1_global_std:.4f}")
    print(f"Final Acc Global (mean±std over folds): {final_acc_global_mean:.4f} ± {final_acc_global_std:.4f}")
    print(f"Antecedents per rule (mean±std): {antecedents_mean:.2f} ± {antecedents_std:.2f}")

    save_file_name = "fed" + str(args.dataset) + "-r" + str(args.n_rule) + "-c" + str(args.n_client) \
                     + "-p" + str(args.n_client_per_round) + '-' + str(args.partition_method)\
                     + str(args.partition_alpha) + "-nl" + str(args.nl) + "-" + args.criterion\
                     + "-lr" + str(args.lr) + ".mat"

    data_save_dir = f"./results"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    data_save_file = f"{data_save_dir}/{save_file_name}"

    csv_file_name = save_file_name.replace(".mat", "_client_fold.csv")
    csv_file_path = f"{data_save_dir}/{csv_file_name}"
    with open(csv_file_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Dataset", "NClients", "Alpha", "Folds", "Fold", "Method", "client", "Rules",
                        "AntecedentsMean", "AntecedentsStd", "f1", "acc", "f1_global_test", "acc_global_test"])
        method = "fedavg_fpnn"
        for fold_idx in range(args.n_kfolds):
            for client_idx in range(args.n_client):
                rules = int((local_train_rule_count[final_round, client_idx, :, fold_idx] > 0).sum().item())
                f1_val = float(local_test_f1_tsr[final_round, client_idx, fold_idx].item())
                acc_val = float(local_test_acc_tsr[final_round, client_idx, fold_idx].item())
                f1_global = float(client_global_test_f1_tsr[client_idx, fold_idx].item())
                acc_global = float(client_global_test_acc_tsr[client_idx, fold_idx].item())
                writer.writerow([
                    args.dataset,
                    args.n_client,
                    args.partition_alpha,
                    args.n_kfolds,
                    fold_idx + 1,
                    method,
                    client_idx + 1,
                    rules,
                    antecedents_mean,
                    antecedents_std,
                    f1_val,
                    acc_val,
                    f1_global,
                    acc_global,
                ])

    io.savemat(data_save_file, save_dict)
    print(save_dict)
    print(f"Wrote CSV: {csv_file_path}")


