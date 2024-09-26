import os
import torch
import json

import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.u_net_model import UNet
from dataset_utils.ralis_collate_fn import RalisCollateFn
from dataset_utils.ralis_dataset import RalisDataset

def create_segmentation_model(segmentation_model_path:str, device_ids:list):

    device = torch.device(device_ids[0]) if torch.cuda.is_available() else torch.device("cpu")
    model = UNet(output_classes=len(RalisCollateFn().valid_classes), device=device)

    if os.path.exists(segmentation_model_path):
        print(f'Loading U-Net Model')
        model.load_state_dict(
            torch.load(segmentation_model_path)
        )

    model.to(device)
    return model

def create_dataset(partition_json_fn:str):
    state_dataset = RalisDataset(
        ANNOTATIONS_DIR,
        IMAGES_DIR,
        "state_dataset",
        partition_json_fn
    )

    state_dataloader = DataLoader(
        state_dataset, batch_size=1, collate_fn=RalisCollateFn()
    )

    return state_dataloader

def compute_label_distribution(prediction_crop:torch.tensor):

    num_classes = prediction_crop.shape[1]
    class_predictions = torch.argmax(prediction_crop, dim=1)  
    normalized_counts = torch.zeros(prediction_crop.shape[0], num_classes, device=prediction_crop.device)

    #per sample in batch/state dataset
    for idx, crop in enumerate(class_predictions):    
        counts = torch.bincount(crop.view(-1), minlength=num_classes)

        total_pixels = crop.shape[0] * crop.shape[1]
        normalized_counts[idx] = counts.float() / total_pixels

    return normalized_counts

def calculate_pixel_entropy(prob_tensor):
    """ Calculate the entropy of probability distributions along the last dimension of the tensor """
    # Clip probabilities to prevent log(0)
    eps = 1e-10
    prob_tensor = torch.clamp(prob_tensor, min=eps, max=1-eps)
    entropies = torch.zeros(prob_tensor.shape[0], prob_tensor.shape[-2], prob_tensor.shape[-1], device=prob_tensor.device)
    
    for idx, prob in enumerate(prob_tensor):
        entropy = -torch.sum(prob * torch.log(prob), dim=0)
        entropies[idx] += entropy

    return entropies

def normalize_entropy(entropies:torch.tensor):
    
    ks_x = entropies.shape[1]//8
    ks_y = entropies.shape[2]//8

     # Min pooling (using max pooling on the negative entropy)
    min_pooled = -F.max_pool2d(-entropies, kernel_size=(ks_x, ks_y))
    avg_pooled  = F.avg_pool2d(entropies, kernel_size=(ks_x, ks_y))
    max_pooled  = F.max_pool2d(entropies, kernel_size=(ks_x, ks_y))
    
    return min_pooled, avg_pooled, max_pooled

def compute_normalized_entropy(prediction_crop:torch.tensor):
    entropies = calculate_pixel_entropy(prediction_crop)
    min_pooled, avg_pooled, max_pooled = normalize_entropy(entropies)

    return min_pooled, avg_pooled, max_pooled

def compute_metrics_per_crop(crops:list, state_predictions:torch.tensor, state_labels:torch.tensor):

    batch_size = state_predictions.shape[0]
    num_classes = state_predictions.shape[1]
    num_regions = len(crops)

    ''' 
    Normalized Label: torch.Size([10, 128, 8]) -> (1280, 8)
    Normalized state min pooled: torch.Size([10, 128, 4, 4]) 
    Normalized state max pooled: torch.Size([10, 128, 4, 4])
    Normalized state avg pooled: torch.Size([10, 128, 4, 4])
    '''

    normalized_state_label_distribution = torch.zeros(batch_size, num_regions, num_classes)
    normalized_state_min_pooled = torch.zeros(batch_size, num_regions, 8, 8) #(avg, min and max pool has a kernel size of (16x16) = (4x4))
    normalized_state_avg_pooled = torch.zeros(batch_size, num_regions, 8, 8)
    normalized_state_max_pooled = torch.zeros(batch_size, num_regions, 8, 8)

    for idx, crop_region in enumerate(crops):
        start_x, start_y = crop_region[0]
        end_x, end_y = crop_region[1]

        prediction_crop = state_predictions[:, :, start_x:end_x, start_y:end_y]
        label_crop = state_labels[:, start_x:end_x, start_y:end_y]
        normalized_counts = compute_label_distribution(prediction_crop)
        min_pooled, avg_pooled, max_pooled = compute_normalized_entropy(prediction_crop)

        normalized_state_label_distribution[:, idx, :] = normalized_counts
        normalized_state_min_pooled[:, idx, :] = min_pooled
        normalized_state_avg_pooled[:, idx, :] = avg_pooled
        normalized_state_max_pooled[:, idx, :] = max_pooled

    normalized_state_label_distribution = normalized_state_label_distribution.view(-1, num_classes)

    normalized_state_avg_pooled = normalized_state_avg_pooled.view(-1, 64)
    normalized_state_min_pooled = normalized_state_min_pooled.view(-1, 64)
    normalized_state_max_pooled = normalized_state_max_pooled.view(-1, 64)

    return torch.cat(
        [normalized_state_label_distribution, normalized_state_min_pooled, normalized_state_avg_pooled, normalized_state_max_pooled],
        dim=-1
    )

def compute_current_state(model:UNet, state_dataloader:DataLoader):
    predicted_tensor_list, label_tensor_list = [], []

    model.eval()

    for idx, data_items in enumerate(state_dataloader):
        for k,v in data_items.items():
            if torch.is_tensor(v):
                data_items[k] = v.to(model.device)

        with torch.no_grad():
            _, prediction_map, label_map = model(**data_items)
        predicted_tensor_list.append(prediction_map)
        label_tensor_list.append(label_map)  

    predicted_tensor_list = torch.cat(predicted_tensor_list, dim=0) #(10, n_c, 512, 1024)
    label_tensor_list = torch.cat(label_tensor_list, dim=0) #(10, 512, 1024)

    state_features = compute_metrics_per_crop(
        state_dataloader.collate_fn.crops, predicted_tensor_list, label_tensor_list
    )

    del predicted_tensor_list, label_tensor_list

    return state_features

def calculate_reward_iou(prediction: torch.tensor, label: torch.tensor, classes: list):
    iou_per_class = {}

    for cls in classes:
        intersection = ((prediction.argmax(dim=1) == cls) & (label == cls)).sum().item()
        union = ((prediction.argmax(dim=1) == cls) | (label == cls)).sum().item()
        iou = intersection / union if union != 0 else 0  # Avoid division by zero

        iou_per_class[cls] = iou

    return torch.tensor(list(iou_per_class.values()))

def calculate_reward_accuracy(prediction:torch.tensor, label:torch.tensor, classes:list):

    accuracy_per_class = {}

    for cls in classes:
        correct = ((prediction.argmax(dim=1) == cls) & (label == cls)).sum().item()
        total = (label == cls).sum().item()
        accuracy = correct / total if total != 0 else 0  # Avoid division by zero        

        accuracy_per_class[cls] = accuracy

    return torch.tensor(list(accuracy_per_class.values())) 

def calculate_reward_accuracy(cm_tensor:torch.tensor, prediction:torch.tensor, label:torch.tensor, classes:list):
    
    prediction_flatten = prediction.argmax(dim=1).view(-1)
    label_flatten = label.view(-1)

    for i in classes:
        for j in classes:
            cm_tensor[i][j] = cm_tensor[i][j] + ((prediction_flatten == i) * (label_flatten == j)).sum().type(torch.IntTensor)

    return cm_tensor

def evaluate(cm:torch.tensor):
    # Compute metrics
    TP_perclass = cm.diagonal().type(torch.float32)
    jaccard_perclass = TP_perclass / (cm.sum(1) + cm.sum(0) - TP_perclass)
    jaccard = torch.mean(jaccard_perclass)
    accuracy = TP_perclass.sum() / cm.sum()

    return accuracy, jaccard, jaccard_perclass

def compute_reward(model:UNet, reward_dataloader:DataLoader):
    reward_acc, reward_iou = torch.zeros(8), torch.zeros(8)

    model.eval()
    classes = reward_dataloader.collate_fn.classes
    cm_tensor = torch.zeros(len(classes), len(classes)).type(torch.IntTensor).to(model.device)

    for idx, data_items in enumerate(reward_dataloader):
        for k,v in data_items.items():
            if torch.is_tensor(v):
                data_items[k] = v.to(model.device)

        with torch.no_grad():
            _, prediction_map, label_map = model(**data_items)

        cm_tensor = calculate_reward_accuracy(cm_tensor, prediction_map, label_map, classes)

        # num_classes = prediction_map.shape[1]

        # acc_per_class = calculate_reward_accuracy(prediction_map, label_map, list(range(num_classes)))
        # iou_per_class = calculate_reward_iou(prediction_map, label_map, list(range(num_classes)))        

        # reward_acc += acc_per_class
        # reward_iou += iou_per_class

    # reward_acc /= len(reward_dataloader)
    # reward_iou /= len(reward_dataloader)

    accuracy, mean_iou, iou_per_class = evaluate(cm_tensor)
    # return acc_per_class, iou_per_class
    return accuracy, mean_iou, iou_per_class
        

if __name__ == "__main__":

    DEVICE_IDS = [2]

    ANNOTATIONS_DIR = "../gtFine/train"
    IMAGES_DIR = "../leftImg8bit/train"

    model = create_segmentation_model(
        '../code/residual_attention_u_net_batch_size=6/model_checkpoints/best-model.pt', device_ids=DEVICE_IDS
    )

    state_dataloader = create_dataset("split_files.json")

    compute_current_state(
        model, state_dataloader
    )