import time
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import copy
import logging
import argparse
from utils.data import FederatedDataset, generate_nico_dg_train_new, load_data_for_domain, generate_nico_unique_train_ours
from utils.domainnet_data import generate_domainnet_ours, load_data_for_domain_domainnet
from utils.models import BackboneModel
from utils.logger import setup_logger
from utils.openimage_data import generate_openimage_ours, load_federated_data_openimage
from ptflops import get_model_complexity_info



random.seed(23)
def train_epoch(model, loader, criterion, optimizer, device, global_params=None, prox_mu=0.0, ewc_params=None):
    model.train()
    total_loss = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        if global_params is not None and prox_mu > 0:
            prox_term = 0
            for name, p in model.named_parameters():
                gp = global_params[name]
                gp = gp.to(device)
                prox_term += (p-gp).norm()**2
            loss = loss + (prox_mu/2) * prox_term
        if ewc_params is not None:
            named_params = dict(model.named_parameters())
            for name, (prev_param, fisher) in ewc_params.items():
                p = named_params.get(name, None)
                if p is None:
                    logging.warning(f"EWC skip: parameter '{name}' not found in model.named_parameters()")
                    continue
                if p.shape != prev_param.shape:
                    logging.warning(f"EWC skip: shape mismatch for '{name}': model {p.shape} vs prev {prev_param.shape}")
                    continue
                # loss = loss + (fisher/2) * (p - prev_param).pow(2).sum()
                loss = loss + 0.5 * (fisher * (p - prev_param).pow(2)).sum()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loaders, criterion, device):
    model.eval()
    total_loss = total_correct = total_samples = 0
    with torch.no_grad():
        for loader in loaders.values():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                total_loss += criterion(output, target).item() * data.size(0)
                preds = output.argmax(dim=1)
                total_correct += preds.eq(target).sum().item()
                total_samples += data.size(0)
    return total_loss/total_samples, total_correct/total_samples


def compute_fisher(model, loader, criterion, device):
    model.eval()
    fisher = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
    total = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        model.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                fisher[name] += p.grad.data.pow(2) * data.size(0)
        total += data.size(0)
    for name in fisher:
        fisher[name] /= total
    return fisher


def fedavg(clients_loaders, test_loaders, config, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global_model = BackboneModel(backbone_model=args.backbone,
                                 num_classes=args.num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    client_ids = list(clients_loaders.keys())
    print("Domains sequence:", client_ids)

    for round_idx, cid in enumerate(client_ids):
        print(f"\n[Round {round_idx + 1}] New domain: {cid}")

        for global_epoch in range(config['num_epochs']):
            print(f"  Global Epoch {global_epoch + 1}/{config['num_epochs']}")
            loader = clients_loaders[cid]
            local = copy.deepcopy(global_model).to(device)
            optimizer = torch.optim.SGD(local.parameters(),
                                        lr=config['learning_rate'],
                                        momentum=config['momentum'])

            train_epoch(local, loader, criterion, optimizer, device)

            new_state = {}
            old_state = global_model.state_dict()
            local_state = local.state_dict()
            for k in old_state:
                new_state[k] = (old_state[k].float() + local_state[k].float()) / 2
            global_model.load_state_dict(new_state)

        seen_domains = client_ids[: round_idx + 1]
        subset_tests = {d: test_loaders[d] for d in seen_domains}
        loss, acc = evaluate(global_model, subset_tests, criterion, device)
        print(f"\n>>> Eval after domain {cid} on {seen_domains}: "
              f"Loss={loss:.4f}, Acc={acc:.4f}")
        acc_logger.info(f"\n>>> Eval after domain {cid} on {seen_domains}: "
              f"Loss={loss:.4f}, Acc={acc:.4f}")

    # final evaluation (on all domains)
    final_loss, final_acc = evaluate(global_model, test_loaders, criterion, device)
    print(f"\nFinal Evaluation on all domains -> "
          f"Loss: {final_loss:.4f}, Accuracy: {final_acc:.4f}")

    return {'global_loss': final_loss, 'accuracy': final_acc}




def fedprox(clients_loaders, test_loaders, config, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global_model = BackboneModel(backbone_model=args.backbone,
                                 num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    mu = config.get('mu', args.prox_mu)             # FedProx penalty coefficient
    lr = config['learning_rate']
    momentum = config['momentum']
    epochs = config['num_epochs']

    client_ids = list(clients_loaders.keys())
    print("Domain order:", client_ids)

    for t, cid in enumerate(client_ids, start=1):
        print(f"\n=== Round {t}: Adding domain '{cid}' ===")

        # --- Local Prox training on this domain ---
        for ep in range(epochs):
            local = copy.deepcopy(global_model).to(device)
            optimizer = torch.optim.SGD(local.parameters(), lr=lr, momentum=momentum)

            train_epoch(local,
                        clients_loaders[cid],
                        criterion,
                        optimizer,
                        device,
                        prox_mu=mu,
                        global_params=global_model.state_dict())

            new_state = {}
            old_state = global_model.state_dict()
            local_state = local.state_dict()
            for k in old_state:
                new_state[k] = 0.5 * (old_state[k].float() + local_state[k].float())
            global_model.load_state_dict(new_state)

            print(f"  Domain '{cid}' Epoch {ep+1}/{epochs} complete")

        seen = client_ids[:t]
        sub_tests = {d: test_loaders[d] for d in seen}
        loss, acc = evaluate(global_model, sub_tests, criterion, device)
        print(f"\n>>> Eval after adding '{cid}' on {seen}: "
              f"Loss={loss:.4f}, Acc={acc:.4f}")
        acc_logger.info(f"\n>>> Eval after adding '{cid}' on {seen}: "
              f"Loss={loss:.4f}, Acc={acc:.4f}")

    # final evaluation on all domains
    final_loss, final_acc = evaluate(global_model, test_loaders, criterion, device)
    print(f"\nFinal Evaluation on all domains -> "
          f"Loss: {final_loss:.4f}, Accuracy: {final_acc:.4f}")

    return {'global_loss': final_loss, 'accuracy': final_acc}


def fedavg_ewc(clients_loaders, test_loaders, config, args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global_model = BackboneModel(backbone_model=args.backbone,
                                 num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    mu = config.get('mu', args.prox_mu)             
    lr = config['learning_rate']
    momentum = config['momentum']
    epochs = config['num_epochs']
    lambda_ewc = args.lambda_ewc                  

    ewc = None
    client_ids = list(clients_loaders.keys())
    print("Domain order:", client_ids)

    for t, cid in enumerate(client_ids, start=1):
        print(f"\n=== Round {t}: Adding domain '{cid}' ===")

        for ep in range(epochs):
            local = copy.deepcopy(global_model).to(device)
            optimizer = torch.optim.SGD(local.parameters(), lr=lr, momentum=momentum)

            train_epoch(
                local,
                clients_loaders[cid],
                criterion,
                optimizer,
                device,
                prox_mu=mu,
                global_params=global_model.state_dict(),
                ewc_params=ewc  
            )

            new_state = {}
            old_state = global_model.state_dict()
            local_state = local.state_dict()
            for k in old_state:
                new_state[k] = 0.5 * (old_state[k].float() + local_state[k].float())
            global_model.load_state_dict(new_state)

            print(f"  Domain '{cid}' Epoch {ep+1}/{epochs} complete")

        fish = compute_fisher(global_model, clients_loaders[cid], criterion, device)
        if ewc is None:
            ewc = {
                name: (param.clone().detach(), fish[name] * lambda_ewc)
                for name, param in global_model.named_parameters()
            }
        else:
            new_ewc = {}
            for name, (mu_prev, fish_prev) in ewc.items():
                if name in fish and fish[name].shape == fish_prev.shape:
                    new_ewc[name] = (
                        global_model.state_dict()[name].clone().detach(),
                        fish_prev + fish[name] * lambda_ewc
                    )
                else:
                    new_ewc[name] = (mu_prev, fish_prev)
            ewc = new_ewc

        seen = client_ids[:t]
        sub_tests = {d: test_loaders[d] for d in seen}
        loss, acc = evaluate(global_model, sub_tests, criterion, device)
        print(f"\n>>> Eval after adding '{cid}' on {seen}: "
              f"Loss={loss:.4f}, Acc={acc:.4f}")
        acc_logger.info(f"\n>>> Eval after adding '{cid}' on {seen}: "
              f"Loss={loss:.4f}, Acc={acc:.4f}")

    final_loss, final_acc = evaluate(global_model, test_loaders, criterion, device)
    print(f"\nFinal Evaluation on all domains -> "
          f"Loss: {final_loss:.4f}, Accuracy: {final_acc:.4f}")

    return {'global_loss': final_loss, 'accuracy': final_acc}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Federated incremental learning with Backbone models and full datasets")
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18','resnet50','resnet101','vgg16','densenet121','vit_b_16','vit_b_32'],
                        help='Backbone architecture')
    parser.add_argument('--dataset', type=str, default='nico_dg',
                        choices=['nico_dg','nico_u','domainnet','openimage'], help='Dataset')
    parser.add_argument('--round', type=int, default=1, help='Round')
    parser.add_argument('--num_samples', type=int, default=50, choices=[10,20,30,40,50], help='Samples per category/domain')
    parser.add_argument('--num_classes', type=int, default=60, help='Number of classes')
    parser.add_argument('--iterations', type=int, default=20, help='Training iterations')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--label_skew', default=False, help='Use label skew division')
    parser.add_argument('--k', type=int, default=5, help='Replay buffer size for FedAvg replay')
    parser.add_argument('--methods', nargs='+',
                        choices=['fedavg','fedprox','fedavg_ewc',], default=['fedavg'],
                        help='Federated methods to run')
    parser.add_argument('--prox_mu', type=float, default=0.1, help='Proximal term mu for FedProx')
    parser.add_argument('--lambda_ewc', type=float, default=0.1, help='EWC penalty weight')
    args = parser.parse_args()

    base_dir = f"logs/{args.methods}/{args.dataset}_{args.backbone}_{args.round}"
    if args.label_skew:
        base_dir += "_label_skew"
    os.makedirs(base_dir, exist_ok=True)
    gloss_logger = setup_logger('global_loss', os.path.join(base_dir, 'global_loss.log'))
    acc_logger = setup_logger('accuracy', os.path.join(base_dir, 'accuracy.log'))
    perf_logger = setup_logger('performance', os.path.join(base_dir, 'performance.log'))

    config = {
        'learning_rate': 0.001,
        'momentum': 0.9,
        'batch_size': args.batch_size,
        'num_epochs': args.iterations,
        'patience': 5
    }
    print(f"Config: {config}")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    transforms_dict = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    if args.dataset == 'nico_dg':
        domains = list(generate_nico_dg_train_new().keys())
        if args.label_skew:
            print("Training noniid with Nico DG")
            all_train = []
            all_test  = []
            for d in domains:
                all_train.extend(load_data_for_domain(d, is_train=True,  num_samples=args.num_samples))
                all_test .extend(load_data_for_domain(d, is_train=False, num_samples=None))
            NUM_CLASSES        = 60
            CLASSES_PER_CLIENT = 10
            NUM_CLIENTS        = 6 

            all_classes = list(range(NUM_CLASSES))
            client_classes = {
                client_id: random.sample(all_classes, CLASSES_PER_CLIENT)
                for client_id in range(NUM_CLIENTS)
            }
            print(client_classes)
            clients_loaders = {}
            test_loaders    = {}

            for client_id, cls_list in client_classes.items():
                client_train = [ (x,y) for (x,y) in all_train if y in cls_list ]
                clients_loaders[client_id] = DataLoader(
                    FederatedDataset(client_train, transforms_dict['train']),
                    batch_size=args.batch_size, shuffle=True
                )
                
                client_test = [ (x,y) for (x,y) in all_test if y in cls_list ]
                test_loaders[client_id] = DataLoader(
                    FederatedDataset(client_test, transforms_dict['test']),
                    batch_size=args.batch_size, shuffle=False
                )
        else:
            clients_loaders = {d: DataLoader(FederatedDataset(load_data_for_domain(d, True, num_samples=args.num_samples), transforms_dict['train']),
                                            batch_size=args.batch_size, shuffle=True)
                            for d in domains}
            test_loaders = {d: DataLoader(FederatedDataset(load_data_for_domain(d, False), transforms_dict['test']),
                                        batch_size=args.batch_size)
                            for d in domains}
    elif args.dataset == 'nico_u':
        category_dict, train_data = generate_nico_unique_train_ours(
            is_train=True, method='federated', samples=args.num_samples)
        
        if args.label_skew:
            print("Training noniid with Nico Unique")
            all_train = []
            for c in train_data:
                all_train.extend(train_data[c])
            
            test_data = generate_nico_unique_train_ours(
                is_train=False, method='federated', samples=args.num_samples)
            all_test = []
            for c in test_data:
                all_test.extend(test_data[c])
            
            NUM_CLASSES        = 60
            CLASSES_PER_CLIENT = 10   # e.g. 10
            NUM_CLIENTS        = 6          # e.g. 6
            
            all_classes = list(range(NUM_CLASSES))
            client_classes = {
                client_id: random.sample(all_classes, CLASSES_PER_CLIENT)
                for client_id in range(NUM_CLIENTS)
            }
            print("Client→classes mapping:", client_classes)
            
            clients_loaders = {}
            test_loaders    = {}
            for client_id, cls_list in client_classes.items():
                # train split
                client_train = [(x,y) for (x,y) in all_train if y in cls_list]
                clients_loaders[client_id] = DataLoader(
                    FederatedDataset(client_train, transforms_dict['train']),
                    batch_size=args.batch_size, shuffle=True
                )
                # test split
                client_test = [(x,y) for (x,y) in all_test if y in cls_list]
                test_loaders[client_id] = DataLoader(
                    FederatedDataset(client_test, transforms_dict['test']),
                    batch_size=args.batch_size, shuffle=False
                )
        else:
            clients_loaders = {
                c: DataLoader(
                    FederatedDataset(train_data[c], transforms_dict['train']),
                    batch_size=args.batch_size, shuffle=True
                )
                for c in train_data
            }
            test_data = generate_nico_unique_train_ours(
                is_train=False, method='federated', samples=args.num_samples)
            test_loaders = {
                c: DataLoader(
                    FederatedDataset(test_data[c], transforms_dict['test']),
                    batch_size=args.batch_size, shuffle=False
                )
                for c in test_data
            }
    elif args.dataset == 'domainnet':
        test_domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        _, _ = generate_domainnet_ours(num_samples=args.num_samples)
        clients_data = {d: load_data_for_domain_domainnet(d, samples=args.num_samples) for d in test_domains}
        clients_loaders = {d: DataLoader(FederatedDataset(clients_data[d], transforms_dict['train']),
                                        batch_size=args.batch_size, shuffle=True)
                           for d in test_domains}
        test_data = {d: load_data_for_domain_domainnet(d, is_train=False) for d in test_domains}
        test_loaders = {d: DataLoader(FederatedDataset(test_data[d], transforms_dict['test']),
                                     batch_size=args.batch_size)
                        for d in test_domains}
    elif args.dataset == 'openimage':
    # get per‐category splits
        clients_data, test_data = load_federated_data_openimage(samples=args.num_samples)

        if args.label_skew:
            print("Training noniid with OpenImage")

            all_train = [sample for cat in clients_data.values() for sample in cat]
            all_test  = [sample for cat in test_data.values() for sample in cat]

            NUM_CLASSES = 20
            class_ids   = list(range(NUM_CLASSES))
            random.shuffle(class_ids)

            client_classes = {}
            idx = 0
            for client_id in range(6):
                n_cls = 3 if client_id < 4 else 4
                client_classes[client_id] = class_ids[idx:idx + n_cls]
                idx += n_cls

            print("Client → classes mapping:", client_classes)

            clients_loaders = {}
            test_loaders    = {}
            for client_id, cls_list in client_classes.items():
                # training split
                client_train = [(x, y) for (x, y) in all_train if y in cls_list]
                clients_loaders[client_id] = DataLoader(
                    FederatedDataset(client_train, transforms_dict['train']),
                    batch_size=args.batch_size, shuffle=True
                )
                # test split
                client_test = [(x, y) for (x, y) in all_test if y in cls_list]
                test_loaders[client_id] = DataLoader(
                    FederatedDataset(client_test, transforms_dict['test']),
                    batch_size=args.batch_size, shuffle=False
                )

        else:
            _, _ = generate_openimage_ours(num_samples=args.num_samples)
            clients_data, test_data = load_federated_data_openimage(samples=args.num_samples)
            clients_loaders = {
                d: DataLoader(
                    FederatedDataset(clients_data[d], transforms_dict['train']),
                    batch_size=args.batch_size, shuffle=True
                )
                for d in clients_data
            }
            test_loaders = {
                d: DataLoader(
                    FederatedDataset(test_data[d], transforms_dict['test']),
                    batch_size=args.batch_size, shuffle=False
                )
                for d in test_data
            }
    else:
        raise ValueError("Dataset Not Implemented")

    # Compute and log model complexity
    model = BackboneModel(backbone_model=args.backbone, num_classes=args.num_classes)
    flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=False, print_per_layer_stat=False)
    perf_logger.info(f"Model FLOPs: {flops:,}")
    perf_logger.info(f"Model Params: {params:,}")

    # Execute federated methods
    results = {}
    for m in args.methods:
        perf_logger.info(f"Starting {m}")
        start = time.time()
        if m == 'fedavg':
            res = fedavg(clients_loaders, test_loaders, config, args)
        elif m == 'fedprox':
            res = fedprox(clients_loaders, test_loaders, config, args)
        elif m == 'fedavg_ewc':
            res = fedavg_ewc(clients_loaders, test_loaders, config, args)
        res['walltime'] = time.time() - start
        results[m] = res
        perf_logger.info(f"{m} walltime: {res['walltime']:.2f}s")

    # Log final metrics
    for m, r in results.items():
        gloss_logger.info(f"{m} loss: {r['global_loss']:.4f}")
        acc_logger.info(f"{m} acc: {r['accuracy']:.4f}")
        perf_logger.info(f"{m} loss/acc: {r['global_loss']:.4f}/{r['accuracy']:.4f}")
    print("Done federated incremental learning.")