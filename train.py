import time
from collections import defaultdict
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from torchvision import transforms
from PIL import Image
import copy
import logging
import argparse
from utils.data import FederatedDataset, generate_nico_dg_train_new, generate_nico_unique_train_ours_single, load_data_for_domain, generate_nico_unique_train_ours
from utils.domainnet_data import generate_domainnet_class_based, load_data_for_domain_domainnet
from utils.models import BackboneModel
from utils.logger import setup_logger
from utils.openimage_data import generate_openimage_ours_new, load_federated_data_openimage
from ptflops import get_model_complexity_info
start_time = time.time()

parser = argparse.ArgumentParser(description="Train a model on the NICO++_DG dataset")
parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'resnet101', 'vgg16', 'densenet121', 'vit_b_16', 'vit_b_32'], help='Backbone model to use')
parser.add_argument('--dataset', type=str, default='nico_dg', choices=['nico_dg', 'nico_u', 'domainnet', 'openimage'], help='Dataset for Training')
parser.add_argument('--round', type=int, default=1, help='Round')
parser.add_argument('--num_samples', type=int, default=50, choices=[10,20,30,40,50], help='number of samples to be used for each category and domain')
parser.add_argument('--num_classes', type=int, default=60 , help='Number of classes')
parser.add_argument('--iterations', type=int, default=20 , help='Number of training iterations')
parser.add_argument('--batch_size', type=int, default=32 , help='Batch Size')
parser.add_argument('--label_skew', type=bool, default=False, help='Label Skew of Feature skew divison')
parser.add_argument('--model', type=str, default='naive', choices=['naive', 'data', 'model', 'merged', 'topk', 'incremental'])
parser.add_argument('--k', type=int, default=5 , help='Retention value k')


args = parser.parse_args()



logger_base_dir = f"logs/{args.model}/{args.dataset}_{args.backbone}_{args.round}_{args.k}"
if args.label_skew is True:
    logger_base_dir += "_label_skew"
os.makedirs(logger_base_dir, exist_ok=True)

config = {
    'learning_rate': 0.001,
    'momentum': 0.9,
    'batch_size': 32,
    'num_epochs': 25,
    'patience': 5
}

# Example usage
# Global loss logger, accuracy logger and client loss logger
gloss_logger = setup_logger('gloss_logger', os.path.join(logger_base_dir, 'global_loss.log'), level=logging.INFO)
# ploss_logger = setup_logger('pretrain_logger', os.path.join(logger_base_dir, 'pretrain_loss.log'), level=logging.INFO)
accuracy_logger = setup_logger('accuracy_logger', os.path.join(logger_base_dir, 'accuracy.log'), level=logging.INFO)
# caccuracy_logger = setup_logger('cat_accuracy_logger', os.path.join(logger_base_dir, 'cat_accuracy.log'), level=logging.INFO)
# closs_logger = setup_logger('closs_logger', os.path.join(logger_base_dir, 'client_loss.log'), level=logging.INFO)


# Define base paths
base_path = "/proj/wasp-nest-cr01/users/x_obaza/oneshot_diff"
if args.dataset == "nico_dg":
    test_base_path = os.path.join(base_path, "NICO_DG_official")
    test_domains = ["outdoor", "autumn", "dim", "grass", "rock", "water"]
    combined_test_file = os.path.join(base_path, "combined_test_files.txt")
    train_base_path = os.path.join(base_path, "NICO_DG")
elif args.dataset == "nico_u":
    file_base_path = os.path.join(base_path, "NICO_unique_official", "NICO_unique_official")
elif args.dataset == "domainnet" or args.dataset == "openimage":
    pass
else:
    raise ValueError("Dataset Not Implemented")

# Initialize category dictionary
category_dict = {}
# Read test files and create category dictionary

### -------- Training file text file generation ------###
# Define training base path

# Initialize training file list
print(f"Config: {config}")

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # Ensures reproducible results
    torch.backends.cudnn.benchmark = False  # Slows down training slightly but ensures consistency

set_seed(42)
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}




class CustomDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.samples = []
        self.transform = transform

        # Read the file and parse image paths and labels
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                # Split from the right side, only at the last space
                image_path, label = line.rsplit(' ', 1)
                # print(f"The file image_path is {image_path} and Label is {label}")
                self.samples.append((image_path, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class CustomDatasetUnique(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


print(f"Using Dataset {args.dataset}")
if args.dataset == "nico_dg":
    image_data_single = generate_nico_dg_train_new()
    for d in image_data_single.keys():
        print(len(image_data_single[d]))
    image_datasets_single = {
        domain: FederatedDataset(data, data_transforms['train']) for domain, data in image_data_single.items()
    }
    dataloaders_single = {
        domain: DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True) for domain, dataset in image_datasets_single.items()
    }
    c_train_data = {domain: load_data_for_domain(domain, is_train=True) for domain in test_domains}
    clients_train_data = {}
    clients_aux_data = []
    for domain in c_train_data.keys():
        clients_train_data[domain] = c_train_data[domain][:int(1 * len(c_train_data[domain]))]

    clients_train_datasets = {domain: FederatedDataset(data, data_transforms['train']) for domain, data in clients_train_data.items()}
    clients_train_dataloaders = {domain: DataLoader(dataset, batch_size=32, shuffle=True) for domain, dataset in clients_train_datasets.items()}
    clients_test_data = {domain: load_data_for_domain(domain, is_train=False) for domain in test_domains}
    clients_test_datasets = {domain: FederatedDataset(data, data_transforms['test']) for domain, data in clients_test_data.items()}
    clients_test_dataloaders = {domain: DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True) for domain, dataset in clients_test_datasets.items()}
elif args.dataset == "nico_u":

    category_dict, training_data = generate_nico_unique_train_ours_single(samples=args.num_samples)
    for d in training_data.keys():
        print(len(training_data[d]))
    image_datasets_single = {
        domain: FederatedDataset(data, data_transforms['train']) for domain, data in training_data.items()
    }
    dataloaders_single = {
        domain: DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True) for domain, dataset in image_datasets_single.items()
    }
    category_dict, clients_train_data = generate_nico_unique_train_ours(is_train=True, method="federated", samples=args.num_samples)
    for c_data in clients_train_data.keys():
        print(len(clients_train_data[c_data]))
    clients_train_datasets = {client: FederatedDataset(data, data_transforms['test']) for client, data in clients_train_data.items()}
    clients_train_dataloaders = {client: DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True) for client, dataset in clients_train_datasets.items()}
    clients_test_data = generate_nico_unique_train_ours(is_train=False, method="federated")
    clients_test_datasets = {client: FederatedDataset(data, data_transforms['test']) for client, data in clients_test_data.items()}
    clients_test_dataloaders = {client: DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True) for client, dataset in clients_test_datasets.items()}
elif args.dataset == "domainnet":
    test_domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    if args.label_skew == True:
        dataloaders_single = generate_domainnet_class_based(num_samples=args.num_samples)
        clients_test_data = {domain: load_data_for_domain_domainnet(domain, is_train=False) for domain in test_domains}
        clients_test_datasets = {domain: FederatedDataset(data, data_transforms['test']) for domain, data in clients_test_data.items()}
        clients_test_dataloaders = {domain: DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True) for domain, dataset in clients_test_datasets.items()}

    else:
        dataloaders_single, clients_test_dataloaders = generate_domainnet_class_based(num_train_samples=args.num_samples)
    clients_train_data = {domain: load_data_for_domain_domainnet(domain, samples=args.num_samples) for domain in test_domains}
    clients_train_datasets = {domain: FederatedDataset(data, data_transforms['train']) for domain, data in clients_train_data.items()}
    clients_train_dataloaders = {domain: DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True) for domain, dataset in clients_train_datasets.items()}

    
elif args.dataset == "openimage":
    dataloaders_single = generate_openimage_ours_new(num_samples=args.num_samples)
    _, clients_test_data = load_federated_data_openimage(samples=args.num_samples)
    clients_train_data, clients_test_data = load_federated_data_openimage(samples=args.num_samples)
    clients_train_datasets = {domain: FederatedDataset(data, data_transforms['train']) for domain, data in clients_train_data.items()}
    clients_train_dataloaders = {domain: DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True) for domain, dataset in clients_train_datasets.items()}
    clients_test_datasets = {domain: FederatedDataset(data, data_transforms['test']) for domain, data in clients_test_data.items()}
    clients_test_dataloaders = {domain: DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True) for domain, dataset in clients_test_datasets.items()}

else:
    raise ValueError("Dataset Not Implemented")



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if args.label_skew == True and args.dataset in ['nico_dg', 'nico_u']:
    if args.dataset == "nico_dg":
        image_data_single = generate_nico_dg_train_new()
        c_test_data = {
        domain: load_data_for_domain(domain, is_train=False)
        for domain in test_domains
        }
    if args.dataset == "nico_u":
        _, image_data_single = generate_nico_unique_train_ours_single(samples=args.num_samples)
        c_test_data = generate_nico_unique_train_ours(is_train=False, method="federated")
    all_data = []
    for domain, data in image_data_single.items():
        all_data.extend(data)
    print(f"Total samples combined from all domains: {len(all_data)}")
    print(f"An example of the sample from All_data is {all_data[0]}")

    num_clients = 6

    client_class_assignment = {
        client: list(range(client * 10, (client + 1) * 10))
        for client in range(num_clients)
    }
    for client, classes in client_class_assignment.items():
        print(f"Client {client}: Assigned classes {classes}")

    client_data = {}
    for client in range(num_clients):
        classes = client_class_assignment[client]
        # Filter samples whose label is in the client's assigned classes.
        filtered_samples = [sample for sample in all_data if sample[1] in classes]
        client_data[client] = filtered_samples
        print(f"Client {client}: {len(filtered_samples)} samples")

    datasets_single = {
        client: FederatedDataset(data, data_transforms['train'])
        for client, data in client_data.items()
    }

    dataloaders_single = {
        client: DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        for client, dataset in datasets_single.items()
    }
    all_test_data = [s for data in c_test_data.values() for s in data]
    print(f"Length of test data : {len(all_test_data)}")
    client_test_data = {
        client: [
            sample
            for sample in all_test_data
            if sample[1] in client_class_assignment[client]
        ]
        for client in range(num_clients)
    }
    print(f"Clients Test Data Keys {client_test_data.keys()}")

    test_datasets_single = {
        client: FederatedDataset(data, data_transforms['test'])
        for client, data in client_test_data.items()
    }
    clients_test_dataloaders = {
        client: DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,         
            num_workers=4,
            pin_memory=True
        )
        for client, dataset in test_datasets_single.items()
    }






def distillation_loss(outputs, teacher_outputs, temperature=2.0):
    T = temperature
    p = nn.functional.log_softmax(outputs / T, dim=1)
    q = nn.functional.softmax(teacher_outputs / T, dim=1)
    loss = nn.KLDivLoss(reduction='batchmean')(p, q) * (T * T)
    return loss

def update_memory(memory, current_dataset, model, memory_per_domain=50):
    new_exemplars = []
    indices = torch.randperm(len(current_dataset))[:memory_per_domain]
    for idx in indices:
        exemplar = current_dataset[idx]
        new_exemplars.append(exemplar)
    if memory is None:
        memory = new_exemplars
    else:
        memory += new_exemplars
    return memory
def print_model_complexity(model, input_shape=(3, 224, 224)):
    macs, params = get_model_complexity_info(model, input_shape, as_strings=True, print_per_layer_stat=False)
    print(f"Model Complexity: {macs} MACs, {params} parameters")



def train_incremental_model_naive(model, dataloaders_train,
                                  num_epochs=10, device='cuda', lr=0.001, batch_size=32):
    cumulative_datasets = []
    
    domains = list(dataloaders_train.keys())
    for t, domain in enumerate(domains, start=1):
        model = BackboneModel(backbone_model=args.backbone, num_classes=args.num_classes).to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        criterion = nn.CrossEntropyLoss()
        print(f"\n=== Naive Incremental Training: Adding domain {domain} ===")
        
        dataset_t = dataloaders_train[domain].dataset if hasattr(dataloaders_train[domain], 'dataset') else dataloaders_train[domain]
        cumulative_datasets.append(dataset_t)
        cumulative_dataset = torch.utils.data.ConcatDataset(cumulative_datasets)
        cumulative_loader = DataLoader(cumulative_dataset, batch_size=batch_size, shuffle=True)
        
        print_model_complexity(model)
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            total_samples = 0
            for inputs, labels in cumulative_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
            
            epoch_loss = running_loss / total_samples
            print(f"Increment {t} - Domain {domain} Epoch {epoch}/{num_epochs-1}: Loss: {epoch_loss:.4f}")
            gloss_logger.info(f"Increment {t} - Domain {domain} Epoch {epoch}/{num_epochs-1}: Loss: {epoch_loss:.4f}")
            scheduler.step(epoch_loss)
        
        # Test on all domains seen so far.
        print(f"Testing Accuracy after domain {domain}")
        accuracy_logger.info(f"Testing Accuracy after domain {domain}")
        test_model(model, {d: clients_test_dataloaders[d] for d in domains[:t]})
        
    return model



def train_incremental_model_regularized(model, dataloaders_train,
                                        num_epochs=10, device='cuda', alpha=0.5, lr=0.001, batch_size=32, reg_lambda=0.01):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    
    # At the start, there is no previous model
    previous_model = None
    
    domains = list(dataloaders_train.keys())
    for t, domain in enumerate(domains, start=1):
        print(f"\n=== Regularized Incremental Training: Training on domain {domain} ===")
        
        # If the dataloader does not wrap a dataset, try to extract it.
        current_dataset = dataloaders_train[domain].dataset if hasattr(dataloaders_train[domain], 'dataset') else dataloaders_train[domain]
        current_loader = DataLoader(current_dataset, batch_size=batch_size, shuffle=True)
        
        # Print current model complexity.
        print_model_complexity(model)
        
        # Save a snapshot of the model before training on the new domain.
        teacher_model = copy.deepcopy(model)
        teacher_model.eval()
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            total_samples = 0
            
            for inputs, labels in current_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss_cls = criterion(outputs, labels)
                loss = loss_cls
                
                # Option 1: Use Parameter Regularization if a previous model exists.
                if previous_model is not None:
                    reg_loss = 0.0
                    # Sum squared difference over all parameters.
                    for p_current, p_old in zip(model.parameters(), previous_model.parameters()):
                        reg_loss += torch.nn.functional.mse_loss(p_current, p_old)
                    loss += reg_lambda * reg_loss
                
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
            
            epoch_loss = running_loss / total_samples
            gloss_logger.info(f"Domain {domain} Epoch {epoch}/{num_epochs-1}: Loss: {epoch_loss:.4f}")
            print(f"Domain {domain} Epoch {epoch}/{num_epochs-1}: Loss: {epoch_loss:.4f}")
            scheduler.step(epoch_loss)
        
        # Update the reference model for regularization.
        previous_model = copy.deepcopy(model)
        previous_model.eval()
        
        # Evaluate on all domains seen so far.
        print(f"Testing after domain {domain}")
        accuracy_logger.info(f"Testing after domain {domain}")
        test_model(model, {d: clients_test_dataloaders[d] for d in domains[:t]})
        
    return model
# Domain-incremental training function
def train_incremental_model(model, dataloaders_train, num_epochs=10, device='cuda', alpha=0.5, lr=0.001):

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # Using a scheduler that reduces LR when the loss plateaus.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    criterion = nn.CrossEntropyLoss()
    memory = None  # buffer for exemplars from previous domains

    domains = list(dataloaders_train.keys())
    for d, domain in enumerate(domains):
        print(f"\n=== Training on domain: {domain} ===")
        current_loader = dataloaders_train[domain]
        
        if memory is not None:
            memory_loader = DataLoader(memory, batch_size=32, shuffle=True)
            memory_iter = iter(memory_loader)
        else:
            memory_loader = None
        
        teacher_model = copy.deepcopy(model)
        teacher_model.eval()

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            # Iterate over the current domain's data
            for inputs, labels in current_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss_cls = criterion(outputs, labels)
                loss = loss_cls

                if memory_loader is not None:
                    try:
                        mem_inputs, _ = next(memory_iter)
                    except StopIteration:
                        memory_iter = iter(memory_loader)
                        mem_inputs, _ = next(memory_iter)
                    mem_inputs = mem_inputs.to(device)
                    mem_outputs = model(mem_inputs)
                    with torch.no_grad():
                        teacher_outputs = teacher_model(mem_inputs)
                    loss_distill = distillation_loss(mem_outputs, teacher_outputs)
                    loss += alpha * loss_distill

                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples
            print(f"Domain {domain} - Epoch {epoch}/{num_epochs-1}: Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            gloss_logger.info(f"Domain {domain} - Epoch {epoch}/{num_epochs-1}: Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            scheduler.step(epoch_loss)
        
        memory = update_memory(memory, current_loader.dataset, model, memory_per_domain=50)
        test_model(model, test_dataloaders=clients_test_dataloaders)
    return model

def select_topk_per_class_indices(model, dataset, k, score_fn, device='cuda'):

    model.eval()
    scores_by_class = defaultdict(list)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            per_sample = score_fn(outputs, labels)  # shape [batch]
            start = batch_idx * loader.batch_size
            for i in range(inputs.size(0)):
                label = labels[i].item()
                global_idx = start + i
                score = per_sample[i].item()
                scores_by_class[label].append((score, global_idx))

    # pick top-k per class
    topk_indices = []
    for label, scored in scores_by_class.items():
        scored.sort(key=lambda x: x[0], reverse=True)
        topk_indices.extend(idx for _, idx in scored[:k])
    return topk_indices


def train_incremental_topk(model,
                           dataloaders_train,
                           test_dataloaders,
                           k,
                           num_epochs=10,
                           device='cuda'):

    model.to(device)
    memory = []  
    domains = list(dataloaders_train.keys())

    for t, domain in enumerate(domains, start=1):
        current_ds = getattr(dataloaders_train[domain], 'dataset', dataloaders_train[domain])

        train_datasets = [current_ds] + memory if memory else [current_ds]
        train_ds = ConcatDataset(train_datasets)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

        optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(train_ds)
            print(f"[Domain {domain}] Epoch {epoch}/{num_epochs-1} — Loss: {epoch_loss:.4f}")

        def loss_score_fn(outputs, labels):
            return nn.functional.cross_entropy(outputs, labels, reduction='none')

        important_idxs = select_topk_per_class_indices(
            model, current_ds, k=k, score_fn=loss_score_fn, device=device)
        exemplar_ds = Subset(current_ds, important_idxs)
        memory.append(exemplar_ds)

        # evaluation on seen domains
        seen = domains[:t]
        seen_loaders = {d: test_dataloaders[d] for d in seen}
        print(f"\n>>> Testing after domain {domain}")
        test_model(model, seen_loaders)
        print("-" * 40)

    return model



def test_model(model, test_dataloaders):
    model.eval()  # Set the model to evaluation mode
    device = next(model.parameters()).device  # Get the device model is on
    total_correct = 0
    total_samples = 0
    category_correct = defaultdict(int)  # Correct predictions per category
    category_samples = defaultdict(int)  # Total samples per category
    
    domain_accuracies = {}
    
    with torch.no_grad():
        for domain, dataloader in test_dataloaders.items():
            domain_correct = 0
            domain_samples = 0
            
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct = (predicted == labels).sum().item()
                # loss = nn.CrossEntropyLoss(outputs, labels)
                domain_correct += correct
                domain_samples += labels.size(0)
                for label, pred in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                    category_samples[label] += 1  # Increment total count for category
                    if label == pred:  # Check if prediction is correct
                        category_correct[label] += 1 
            
            domain_accuracy = 100 * domain_correct / domain_samples
            domain_accuracies[domain] = domain_accuracy
            total_correct += domain_correct
            total_samples += domain_samples
            
            print(f"Accuracy for domain {domain}: {domain_accuracy:.2f}%")
            accuracy_logger.info(f"Accuracy for domain {domain}: {domain_accuracy:.2f}%")
    
    overall_accuracy = 100 * total_correct / total_samples
    print(f"Overall accuracy across all domains: {overall_accuracy:.2f}%")
    accuracy_logger.info(f"Overall accuracy across all domains: {overall_accuracy:.2f}%")
    

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BackboneModel(backbone_model=args.backbone, num_classes=args.num_classes)
    model = model.to(device)

    start_time = time.time()
    if args.model == 'incremental':
        model = train_incremental_naive(model, dataloaders_single, num_epochs=config['num_epochs'], device=device)
    elif args.model == 'naive':
        print("Starting training with Naive mode")
        accuracy_logger.info("Starting training with Naive mode")
        model = train_incremental_model_naive(model, dataloaders_single, num_epochs=config['num_epochs'],
                                              device=device, lr=config['learning_rate'], batch_size=config['batch_size'])
    elif args.model == 'model':
        print("Starting training with Model mode")
        accuracy_logger.info("Starting training with Model mode")
        model = train_incremental_model_regularized(model, dataloaders_single, num_epochs=config['num_epochs'], device=device,
                                                    alpha=0.5, lr=config['learning_rate'], batch_size=config['batch_size'])
    elif args.model == 'topk':
        print("Starting training with Top K mode")
        accuracy_logger.info("Starting training with Top K mode")
        model = train_incremental_topk(model, dataloaders_single, clients_test_dataloaders, k=args.k,
                                       num_epochs=config['num_epochs'], device=device)
    end_time = time.time()
    print(f"Training runtime: {end_time - start_time:.2f} seconds")
    accuracy_logger.info(f"Training runtime: {end_time - start_time:.2f} seconds")
    
    