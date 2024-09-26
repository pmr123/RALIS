import torch, cv2
from collections import defaultdict
from torch.optim import AdamW, SGD
import torch.utils
from model.u_net_model import UNet
from dataset_utils.ralis_collate_fn import RalisTrainCollateFn

from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class UNetTrainer:

    def __init__(self, model:UNet, ralis_train_collate_fn:RalisTrainCollateFn, batch_size:int=1):

        self.model = model
        self.ralis_train_collate_fn = ralis_train_collate_fn
        self.batch_size = batch_size

        # self._init_optimizer()

    def _init_optimizer(self):

        print(f'Initailizing Optimizer for {self.model.__class__.__name__}')
        param_dict = []

        param_dict.append({
            "params":self.model.decoder_module.parameters(), "lr":5e-5, "model_name":"U-Net" 
        })

        self.optimizer = SGD(
            params=param_dict, weight_decay=0.005, nesterov=True, momentum=0.9
        )

        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.7
        )

    def prepare_training_dataloader(self, selection_regions:list):

        images_info_dict = defaultdict(lambda:defaultdict(list))

        for region in selection_regions:
            city_name, region = region
            image_fn, region = region.split('-region_')
            region_info = [int(i) for i in region.split('_')]
            image_id = '_'.join(image_fn.split('_')[:-1])
            gtfine_image_file = f'{image_id}_gtFine_labelIds.png'

            file_path = f'{self.ralis_train_collate_fn.images_dir}/{city_name}/{image_fn}'
            images_info_dict[file_path]['regions'].append(region_info)

            images_info_dict[file_path]['target_path'] = f'{self.ralis_train_collate_fn.annotations_dir}/{city_name}/{gtfine_image_file}'

        training_corpus = {
            "image_tensors":[],
            "label_tensors":[],
            "region_masks":[]
        }
        
        for file_path in images_info_dict:
            gt_file = images_info_dict[file_path]['target_path']

            original_image = cv2.imread(file_path)
            gtfine_image = cv2.imread(gt_file, cv2.IMREAD_UNCHANGED)

            if self.ralis_train_collate_fn.crop:
                original_image = original_image[:800,:,:]
                gtfine_image = gtfine_image[:800, :]     

            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            for category_id, label_list in self.ralis_train_collate_fn.category_id_2_label.items():
                for label in label_list:                
                    id = label.id
                    gtfine_image[gtfine_image == id] = category_id  

            original_image = cv2.resize(original_image, (self.ralis_train_collate_fn.resizing_height, self.ralis_train_collate_fn.resizing_width), interpolation=cv2.INTER_LINEAR)
            gtfine_image = cv2.resize(gtfine_image, (self.ralis_train_collate_fn.resizing_height, self.ralis_train_collate_fn.resizing_width), interpolation=cv2.INTER_LINEAR)

            original_image = np.transpose(original_image, (2, 0, 1))
            original_image = torch.from_numpy(original_image).float()
            gtfine_image = torch.from_numpy(gtfine_image).long()

            selection_mask = torch.zeros(original_image.shape[-2], original_image.shape[-1])

            for region in images_info_dict[file_path]['regions']:
                sx, sy, ex, ey = region 
                selection_mask[sx:ex, sy:ey] = 1 

            training_corpus["image_tensors"].append(original_image.unsqueeze(0))
            training_corpus["label_tensors"].append(gtfine_image.unsqueeze(0))
            training_corpus["region_masks"].append(selection_mask.unsqueeze(0))

        for k, v in training_corpus.items():
            training_corpus[k] = torch.concat(v, dim=0).to(self.model.device)
            # print(f'{k}\t{training_corpus[k].size()}')

        dataset = TensorDataset(training_corpus["image_tensors"], training_corpus["label_tensors"], training_corpus["region_masks"])
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train(self, episode:int, step:int):
        self._init_optimizer()
        dataloader = self.prepare_training_dataloader(self.ralis_train_collate_fn.labelled_pool)

        self.model.train()
        
        total_loss = 0.0
        accuracies_per_class = torch.zeros(8)
        iou_per_class = torch.zeros(8)     

        print(f'Training Episode {episode} - Step {step} - Training Labelled Corpus {len(dataloader.dataset)}')   

        for idx, (batch_images, batch_labels, batch_masks) in enumerate(dataloader):
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                data_items = {
                    "image_tensors":batch_images.to(self.model.device),
                    "label_tensors":batch_labels.to(self.model.device),
                    "region_masks":batch_masks.float().to(self.model.device)
                }
                _, pred_map, _ = self.model(**data_items)

                target_masked = batch_masks * batch_labels
                prediction_masked = batch_masks.unsqueeze(1).expand_as(pred_map) * pred_map

                masked_loss = torch.nn.functional.cross_entropy(
                    prediction_masked, target_masked.long(), reduction='sum', ignore_index=0
                )/batch_masks.sum()

                masked_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=4
                )

                self.optimizer.step()
                self.lr_scheduler.step()

                acc, iou = self.compute_scores(
                    prediction_masked, batch_masks, target_masked
                )

                total_loss += masked_loss.item()
                accuracies_per_class += acc
                iou_per_class += iou 

        accuracies_per_class /= len(dataloader)
        iou_per_class /= len(dataloader)
        total_loss /= len(dataloader)

        del self.optimizer
        del self.lr_scheduler
        del dataloader
        del data_items

        return total_loss, accuracies_per_class, iou_per_class
        

    def compute_scores(self, prediction_masked:torch.tensor, batch_masks:torch.tensor, target_masked:torch.tensor, epsilon=1e-6):

        num_classes = prediction_masked.shape[1]
        predicted_classes = prediction_masked.argmax(dim=1)

        accuracies_per_class = torch.zeros(num_classes)
        ious_per_class = torch.zeros(num_classes)

        # Only consider masked regions for IOU and Accuracy computation
        for class_index in range(num_classes):
            true_class = target_masked == class_index
            pred_class = predicted_classes == class_index
            mask = batch_masks == 1  # Assuming mask is 1 for regions to consider
            
            # Compute True Positives (TP), False Positives (FP), and False Negatives (FN)
            TP = (pred_class & true_class & mask).sum(dim=[1, 2])
            FP = (pred_class & ~true_class & mask).sum(dim=[1, 2])
            FN = (~pred_class & true_class & mask).sum(dim=[1, 2])
            
            # Handle division by zero for accuracy
            valid = TP + FP + FN > 0
            accuracies_per_class[class_index] = torch.where(valid, TP / (TP + FP + FN + epsilon), torch.tensor(0.0)).mean()
            
            # Intersection and Union for current class
            intersection = TP
            union = TP + FP + FN
            
            # Handle division by zero for IOU
            valid_iou = union > 0
            ious_per_class[class_index] = torch.where(valid_iou, intersection / (union + epsilon), torch.tensor(0.0)).mean()

        return accuracies_per_class, ious_per_class

        