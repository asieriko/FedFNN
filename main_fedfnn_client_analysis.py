import argparse
import os
import csv
from pathlib import Path
from utils.logger import Logger
from data_process.dataset import FedDatasetCV
# import sys
import numpy as np
import torch
import wandb
import scipy.io as io
# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./")))
from models.fpnn import *
from models.fedfpnn_fedavg_api import FedAvgAPI
from data_process.custom_partiton import load_multiple_fold_files
from data_process.load_dataset import read_json_dataset


def add_args(p_parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    p_parser.add_argument('--model', type=str, default='fed_fpnn_csm', metavar='N',
                          help='neural network used in training')

    p_parser.add_argument('--dataset', type=str, default='iris', metavar='N',  # FIXME: wifi
                          help='dataset used for training')

    p_parser.add_argument('--fs', type=str, default='l2', metavar='N',
                          help='firing strength layer')

    p_parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                          help='how to partition the dataset on local workers')

    p_parser.add_argument('--partition_alpha', type=float, default=0.1, metavar='PA',
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

    p_parser.add_argument('--n_client', type=int, default=5, metavar='NN',
                          help='number of workers in a distributed cluster')

    p_parser.add_argument('--n_client_per_round', type=int, default=5, metavar='NN',
                          help='number of workers')

    p_parser.add_argument('--comm_round', type=int, default=200,
                          help='how many round of communications we shoud use')

    p_parser.add_argument('--milestone', type=int, default=10,
                          help='the tag that local rules suit their own rules after certain round of communications')

    p_parser.add_argument('--epochs', type=int, default=15, metavar='EP',
                          help='how many epochs will be trained locally')

    p_parser.add_argument('--frequency_of_the_test', type=int, default=1,
                          help='the frequency of the algorithms')

    p_parser.add_argument('--gpu', type=int, default=0,
                          help='gpu')

    p_parser.add_argument('--n_rule', type=int, default=15,
                          help='rule number')
    p_parser.add_argument('--n_rule_min', type=int, default=5,
                          help='rule number')
    p_parser.add_argument('--n_rule_max', type=int, default=8,
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

    p_parser.add_argument('--n_kfolds', type=int, default=5,
                          help='The number of k_fold cross validation')

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
    elif p_args.model == "fed_fpnn_csm":
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

    # Rule activity from the explicit client rule status (rules_idx_list).
    server_active_rule_count_tsr = torch.zeros(args.n_kfolds).to(args.device)
    client_active_rule_count_tsr = torch.zeros(args.n_client, args.n_kfolds).to(args.device)
    client_rule_active_mask_tsr = torch.zeros(args.n_client, args.n_rule, args.n_kfolds, dtype=torch.bool)
    server_rule_active_mask_tsr = torch.zeros(args.n_rule, args.n_kfolds, dtype=torch.bool)
    rule_structure_per_fold = []

    dataset_name = args.dataset
    alpha = args.partition_alpha
    n_clients = args.n_client
    n_rep = args.n_kfolds
    task = "C"

    dataset_folder = args.partition_dataset or dataset_name
    dataset_base = Path(args.partition_dir) / dataset_folder
    base_prefix = f"{dataset_folder}_a{alpha}_n{n_clients}"

    dataset_path = dataset_base / f"{base_prefix}_f1-{n_rep}.json"
    _, data, _ = read_json_dataset(dataset_name, dataset_path)
    inputs = data.data
    targets = data.target
    n_class = len(np.unique(targets))

    partition_files = []
    for fold_idx in range(n_rep):
        partition_file = dataset_base / f"{base_prefix}_f{fold_idx + 1}-{n_rep}.json"
        partition_files.append(str(partition_file))

    partition = load_multiple_fold_files(partition_files, targets, args)
    args.partition = partition
    dataset = FedDatasetCV(
        inputs,
        targets,
        n_class,
        task,
        args.dataset,
        p_args=args,
        fed_kfold_partition=partition,
        require_partition=True,
    )
    n_para = 0
    for cv_idx in range(args.n_kfolds):
        args.cv = cv_idx
        args.logger.war(f"=====k_fold: {cv_idx + 1}=======")

        dataset.set_current_folds(cv_idx)
        args.n_class = dataset.n_class
        args.n_fea = dataset.n_fea

        args.tag = f"{args.dataset}_{args.model}_analysis_r{args.n_rule}" \
                   f"c{args.n_client}p{args.n_client_per_round}" \
                   f"_{args.partition_method}{args.partition_alpha}" \
                   f"_nl{args.nl}_{args.criterion}_lr{args.lr}_e{args.epochs}cr{args.comm_round}"

        if not args.b_debug:
            wandb.init(
                project=f"FederatedFPNN-{args.dataset}",
                name=f"{args.tag}_cv{cv_idx + 1}",
                config=args,
            )

        global_model = create_model(args)
        if cv_idx == 0:
            n_para = sum(param.numel() for param in global_model.parameters())
            args.logger.war(f"parameter amount : {n_para}")

        fedavgAPI = FedAvgAPI(dataset, global_model, args)
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

        # Explicit active rules from local rule index lists (status-based activity).
        active_rule_mask = torch.zeros(args.n_rule, dtype=torch.bool)
        for client_idx, client in enumerate(fedavgAPI.client_list):
            active_rules_raw = np.array(client.rules_idx_list, dtype=int).reshape(-1)
            valid_mask = (active_rules_raw >= 0) & (active_rules_raw < args.n_rule)
            active_rules = np.unique(active_rules_raw[valid_mask])
            if active_rules.size > 0:
                client_rule_active_mask_tsr[client_idx, active_rules, cv_idx] = True
            client_active_rule_count_tsr[client_idx, cv_idx] = int(
                client_rule_active_mask_tsr[client_idx, :, cv_idx].sum().item()
            )
            active_rule_mask |= client_rule_active_mask_tsr[client_idx, :, cv_idx]
        server_rule_active_mask_tsr[:, cv_idx] = active_rule_mask
        server_active_rule_count_tsr[cv_idx] = int(server_rule_active_mask_tsr[:, cv_idx].sum().item())

        fold_rule_structure = []
        global_state = fedavgAPI.global_model.cpu().state_dict()
        for rule_idx in range(args.n_rule):
            key_prefix = f"rule_{rule_idx}"
            fold_rule_structure.append({
                "proto": global_state[f"{key_prefix}.antecedent_layer.proto"].detach().cpu().numpy().reshape(-1),
                "var": global_state[f"{key_prefix}.antecedent_layer.var"].detach().cpu().numpy().reshape(-1),
                "consq_weight": global_state[f"{key_prefix}.consequent_layer.consq_layers.weight"].detach().cpu().numpy(),
                "consq_bias": global_state[f"{key_prefix}.consequent_layer.consq_layers.bias"].detach().cpu().numpy(),
            })
        rule_structure_per_fold.append(fold_rule_structure)

        if not args.b_debug:
            wandb.finish()

    final_round = args.comm_round - 1

    # Rule usage based on winner-takes-all counts (argmax fs).
    server_used_rules_per_fold = (global_train_rule_count[final_round] > 0).sum(dim=0)
    client_used_rules_per_fold = (local_train_rule_count[final_round] > 0).sum(dim=1)

    server_total_rules = int(args.n_rule)

    # Active-rule statistics from client status vectors.
    server_rules_mean = float(server_active_rule_count_tsr.float().mean().item())
    server_rules_std = float(server_active_rule_count_tsr.float().std(unbiased=False).item())
    client_rules_mean = float(client_active_rule_count_tsr.float().mean().item())
    client_rules_std = float(client_active_rule_count_tsr.float().std(unbiased=False).item())
    client_rules_mean_each = client_active_rule_count_tsr.float().mean(dim=1)
    client_rules_std_each = client_active_rule_count_tsr.float().std(dim=1, unbiased=False)

    # Averages for winner-takes-all usage counters.
    server_used_rules_mean = float(server_used_rules_per_fold.float().mean().item())
    server_used_rules_std = float(server_used_rules_per_fold.float().std(unbiased=False).item())
    client_used_rules_mean = float(client_used_rules_per_fold.float().mean().item())
    client_used_rules_std = float(client_used_rules_per_fold.float().std(unbiased=False).item())
    client_used_rules_mean_each = client_used_rules_per_fold.float().mean(dim=1)
    client_used_rules_std_each = client_used_rules_per_fold.float().std(dim=1, unbiased=False)

    # In this setup, each active rule uses all antecedents (n_fea).
    server_antecedents_mean = float(args.n_fea)
    server_antecedents_std = 0.0
    client_antecedents_mean = float(args.n_fea)
    client_antecedents_std = 0.0
    client_antecedents_mean_each = torch.full((args.n_client,), float(args.n_fea)).to(args.device)
    client_antecedents_std_each = torch.zeros(args.n_client).to(args.device)

    client_f1_per_fold = local_test_f1_tsr[final_round]
    client_acc_per_fold = local_test_acc_tsr[final_round]
    client_f1_mean_each = client_f1_per_fold.mean(dim=1)
    client_f1_std_each = client_f1_per_fold.std(dim=1, unbiased=False)
    client_acc_mean_each = client_acc_per_fold.mean(dim=1)
    client_acc_std_each = client_acc_per_fold.std(dim=1, unbiased=False)

    final_f1_per_fold = global_test_f1_tsr[final_round].detach().cpu().numpy()
    final_f1_mean = float(final_f1_per_fold.mean())
    final_f1_std = float(final_f1_per_fold.std())
    final_acc_per_fold = global_test_acc_tsr[final_round].detach().cpu().numpy()
    final_acc_mean = float(final_acc_per_fold.mean())
    final_acc_std = float(final_acc_per_fold.std())

    save_dict = {
        "global_train_acc_tsr": global_train_acc_tsr.cpu().numpy(),
        "global_train_loss_tsr": global_train_loss_tsr.cpu().numpy(),
        "global_test_loss_tsr": global_test_loss_tsr.cpu().numpy(),
        "global_test_acc_tsr": global_test_acc_tsr.cpu().numpy(),
        "global_train_f1_tsr": global_train_f1_tsr.cpu().numpy(),
        "global_test_f1_tsr": global_test_f1_tsr.cpu().numpy(),
        "global_train_rule_count": global_train_rule_count.cpu().numpy(),
        "global_train_rule_contr": global_train_rule_contr.cpu().numpy(),
        "local_train_rule_count": local_train_rule_count.cpu().numpy(),
        "local_train_rule_contr": local_train_rule_contr.cpu().numpy(),
        "local_train_acc_tsr": local_train_acc_tsr.cpu().numpy(),
        "local_train_loss_tsr": local_train_loss_tsr.cpu().numpy(),
        "local_test_loss_tsr": local_test_loss_tsr.cpu().numpy(),
        "local_test_acc_tsr": local_test_acc_tsr.cpu().numpy(),
        "local_test_f1_tsr": local_test_f1_tsr.cpu().numpy(),
        "server_total_rules": server_total_rules,
        "server_active_rule_count_tsr": server_active_rule_count_tsr.cpu().numpy(),
        "client_active_rule_count_tsr": client_active_rule_count_tsr.cpu().numpy(),
        "server_rule_active_mask_tsr": server_rule_active_mask_tsr.cpu().numpy(),
        "client_rule_active_mask_tsr": client_rule_active_mask_tsr.cpu().numpy(),
        "server_used_rules_per_fold": server_used_rules_per_fold.detach().cpu().numpy(),
        "client_used_rules_per_fold": client_used_rules_per_fold.detach().cpu().numpy(),
        "server_rules_mean": server_rules_mean,
        "server_rules_std": server_rules_std,
        "client_rules_mean": client_rules_mean,
        "client_rules_std": client_rules_std,
        "server_used_rules_mean": server_used_rules_mean,
        "server_used_rules_std": server_used_rules_std,
        "client_used_rules_mean": client_used_rules_mean,
        "client_used_rules_std": client_used_rules_std,
        "server_antecedents_mean": server_antecedents_mean,
        "server_antecedents_std": server_antecedents_std,
        "client_antecedents_mean": client_antecedents_mean,
        "client_antecedents_std": client_antecedents_std,
        "client_rules_mean_each": client_rules_mean_each.detach().cpu().numpy(),
        "client_rules_std_each": client_rules_std_each.detach().cpu().numpy(),
        "client_used_rules_mean_each": client_used_rules_mean_each.detach().cpu().numpy(),
        "client_used_rules_std_each": client_used_rules_std_each.detach().cpu().numpy(),
        "client_antecedents_mean_each": client_antecedents_mean_each.detach().cpu().numpy(),
        "client_antecedents_std_each": client_antecedents_std_each.detach().cpu().numpy(),
        "client_f1_mean_each": client_f1_mean_each.detach().cpu().numpy(),
        "client_f1_std_each": client_f1_std_each.detach().cpu().numpy(),
        "client_acc_mean_each": client_acc_mean_each.detach().cpu().numpy(),
        "client_acc_std_each": client_acc_std_each.detach().cpu().numpy(),
        "final_f1_mean": final_f1_mean,
        "final_f1_std": final_f1_std,
        "final_acc_mean": final_acc_mean,
        "final_acc_std": final_acc_std,
    }

    args.logger.info(f"Final F1 (mean+-std over folds): {final_f1_mean:.4f} +- {final_f1_std:.4f}")
    args.logger.info(f"Final Acc (mean+-std over folds): {final_acc_mean:.4f} +- {final_acc_std:.4f}")
    args.logger.info(f"Server active rules (mean+-std): {server_rules_mean:.2f} +- {server_rules_std:.2f}")
    args.logger.info(f"Client active rules (mean+-std): {client_rules_mean:.2f} +- {client_rules_std:.2f}")
    args.logger.info(f"Server used rules (mean+-std): {server_used_rules_mean:.2f} +- {server_used_rules_std:.2f}")
    args.logger.info(f"Client used rules (mean+-std): {client_used_rules_mean:.2f} +- {client_used_rules_std:.2f}")

    save_file_name = "fed" + str(args.dataset) + "-r" + str(args.n_rule) + "-c" + str(args.n_client) \
                     + "-p" + str(args.n_client_per_round) + '-' + str(args.partition_method) \
                     + str(args.partition_alpha) + "-nl" + str(args.nl) + "-" + args.criterion \
                     + "-lr" + str(args.lr) + ".mat"

    data_save_dir = "./results"
    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    data_save_file = f"{data_save_dir}/{save_file_name}"
    io.savemat(data_save_file, save_dict)

    csv_file_name = save_file_name.replace(".mat", "_client_fold.csv")
    csv_file_path = f"{data_save_dir}/{csv_file_name}"
    with open(csv_file_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "Dataset", "NClients", "Alpha", "Folds", "Fold", "Method", "Client",
            "ServerTotalRules", "ServerActiveRules", "ServerUsedRulesFinalRound",
            "ClientActiveRules", "ClientUsedRulesFinalRound",
            "ServerAntecedentsMean", "ClientAntecedentsMean",
            "F1", "Acc", "ClientF1Mean", "ClientAccMean",
        ])
        method = "fedfnn_client_analysis"
        for fold_idx in range(args.n_kfolds):
            for client_idx in range(args.n_client):
                writer.writerow([
                    args.dataset,
                    args.n_client,
                    args.partition_alpha,
                    args.n_kfolds,
                    fold_idx + 1,
                    method,
                    client_idx + 1,
                    server_total_rules,
                    int(server_rule_active_mask_tsr[:, fold_idx].sum().item()),
                    int(server_used_rules_per_fold[fold_idx].item()),
                    int(client_rule_active_mask_tsr[client_idx, :, fold_idx].sum().item()),
                    int(client_used_rules_per_fold[client_idx, fold_idx].item()),
                    server_antecedents_mean,
                    client_antecedents_mean,
                    float(local_test_f1_tsr[final_round, client_idx, fold_idx].item()),
                    float(local_test_acc_tsr[final_round, client_idx, fold_idx].item()),
                    float(client_f1_mean_each[client_idx].item()),
                    float(client_acc_mean_each[client_idx].item()),
                ])

    csv_mean_name = save_file_name.replace(".mat", "_client_mean.csv")
    csv_mean_path = f"{data_save_dir}/{csv_mean_name}"
    with open(csv_mean_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "Dataset", "Client",
            "ServerTotalRules",
            "ServerActiveRulesMean", "ServerActiveRulesStd",
            "ServerUsedRulesFinalRoundMean", "ServerUsedRulesFinalRoundStd",
            "ClientActiveRulesMean", "ClientActiveRulesStd",
            "ClientUsedRulesFinalRoundMean", "ClientUsedRulesFinalRoundStd",
            "AntecedentsMean", "AntecedentsStd",
            "F1Mean", "F1Std", "AccMean", "AccStd",
        ])
        for client_idx in range(args.n_client):
            writer.writerow([
                args.dataset,
                client_idx + 1,
                server_total_rules,
                server_rules_mean,
                server_rules_std,
                server_used_rules_mean,
                server_used_rules_std,
                float(client_rules_mean_each[client_idx].item()),
                float(client_rules_std_each[client_idx].item()),
                float(client_used_rules_mean_each[client_idx].item()),
                float(client_used_rules_std_each[client_idx].item()),
                float(client_antecedents_mean_each[client_idx].item()),
                float(client_antecedents_std_each[client_idx].item()),
                float(client_f1_mean_each[client_idx].item()),
                float(client_f1_std_each[client_idx].item()),
                float(client_acc_mean_each[client_idx].item()),
                float(client_acc_std_each[client_idx].item()),
            ])

    txt_file_name = save_file_name.replace(".mat", "_rule_status_final_round.txt")
    txt_file_path = f"{data_save_dir}/{txt_file_name}"
    with open(txt_file_path, "w") as txt_file:
        txt_file.write("Rule status report (final training round)\n")
        txt_file.write(f"Dataset: {args.dataset}\n")
        txt_file.write(f"Clients: {args.n_client}\n")
        txt_file.write(f"Rules: {args.n_rule}\n")
        txt_file.write(f"Folds: {args.n_kfolds}\n\n")

        for fold_idx in range(args.n_kfolds):
            txt_file.write(f"=== Fold {fold_idx + 1} ===\n")
            fold_rule_structure = rule_structure_per_fold[fold_idx]

            txt_file.write("Server rules:\n")
            for rule_idx in range(args.n_rule):
                status = "ACTIVE" if bool(server_rule_active_mask_tsr[rule_idx, fold_idx].item()) else "INACTIVE"
                txt_file.write(f"  Rule {rule_idx + 1}: {status}\n")
                txt_file.write("    Antecedents:\n")
                proto = fold_rule_structure[rule_idx]["proto"]
                var = fold_rule_structure[rule_idx]["var"]
                for fea_idx in range(len(proto)):
                    txt_file.write(
                        f"      x{fea_idx + 1}: mu={proto[fea_idx]:.6f}, sigma={var[fea_idx]:.6f}\n"
                    )
                txt_file.write("    Consequents:\n")
                consq_w = fold_rule_structure[rule_idx]["consq_weight"]
                consq_b = fold_rule_structure[rule_idx]["consq_bias"]
                for class_idx in range(consq_w.shape[0]):
                    weights_str = ", ".join([f"{w:.6f}" for w in consq_w[class_idx]])
                    txt_file.write(
                        f"      class{class_idx + 1}: weights=[{weights_str}], bias={consq_b[class_idx]:.6f}\n"
                    )

            txt_file.write("Client rules:\n")
            for client_idx in range(args.n_client):
                client_used_count = int(client_used_rules_per_fold[client_idx, fold_idx].item())
                client_used_mask = local_train_rule_count[final_round, client_idx, :, fold_idx] > 0
                client_used_rule_ids = (
                    client_used_mask.nonzero(as_tuple=False).view(-1).detach().cpu().numpy().tolist()
                )
                txt_file.write(f"  Client {client_idx + 1}:\n")
                txt_file.write(f"    ClientUsedRulesFinalRound: {client_used_count}\n")
                txt_file.write(f"    ClientUsedRuleIdsFinalRound: {[int(idx) + 1 for idx in client_used_rule_ids]}\n")
                for rule_idx in range(args.n_rule):
                    # For client readability, ACTIVE means used on final round (count > 0).
                    status = "ACTIVE" if bool(client_used_mask[rule_idx].item()) else "INACTIVE"
                    txt_file.write(f"    Rule {rule_idx + 1}: {status}\n")
                    txt_file.write("      Antecedents:\n")
                    proto = fold_rule_structure[rule_idx]["proto"]
                    var = fold_rule_structure[rule_idx]["var"]
                    for fea_idx in range(len(proto)):
                        txt_file.write(
                            f"        x{fea_idx + 1}: mu={proto[fea_idx]:.6f}, sigma={var[fea_idx]:.6f}\n"
                        )
                    txt_file.write("      Consequents:\n")
                    consq_w = fold_rule_structure[rule_idx]["consq_weight"]
                    consq_b = fold_rule_structure[rule_idx]["consq_bias"]
                    for class_idx in range(consq_w.shape[0]):
                        weights_str = ", ".join([f"{w:.6f}" for w in consq_w[class_idx]])
                        txt_file.write(
                            f"        class{class_idx + 1}: weights=[{weights_str}], bias={consq_b[class_idx]:.6f}\n"
                        )

            txt_file.write("\n")

    print(f"Wrote MAT: {data_save_file}")
    print(f"Wrote CSV: {csv_file_path}")
    print(f"Wrote CSV: {csv_mean_path}")
    print(f"Wrote TXT: {txt_file_path}")
