import os
import torch
import json

import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.u_net_model import UNet
from model.dqn import QueryNetworkDQN
from dataset_utils.ralis_collate_fn import RalisCollateFn, RalisTrainCollateFn
from dataset_utils.ralis_dataset import RalisDataset
from dataset_utils.utils import plot_loss_curve, plot_reward_curve

from compute_state import compute_current_state, compute_reward
from u_net_trainer import UNetTrainer
from dqn_trainer import DQNTrainer
from transition import ExperienceReplay
from logger import Logger

def compute_weighted_reward(cur_iou, next_iou, epsilon=1e-6, cap=10):
    # Ensure IOU values are non-zero to avoid division by zero
    safe_iou = cur_iou + epsilon
    
    # Calculate weights inversely proportional to the current IOU values
    weights = 1 / safe_iou
    
    # Normalize weights to sum to 1 and cap them to avoid excessively large values
    weights = torch.clamp(weights / weights.sum(), max=cap / weights.size(0))
    
    # Calculate the change in IOU
    delta_iou = next_iou - cur_iou
    
    # Calculate the weighted reward
    weighted_reward = (delta_iou * weights).sum(dim=-1)

    return weighted_reward

def create_segmentation_model(segmentation_model_path:str, device_ids:list):

    device = torch.device(device_ids[0]) if torch.cuda.is_available() else torch.device("cpu")
    model = UNet(output_classes=len(RalisCollateFn().valid_classes), device=device)
    # model = torch.nn.DataParallel(model, device_ids=device_ids[:2])

    # Load GPU model on a different GPU

    if os.path.exists(segmentation_model_path):
        print(f'Loading U-Net Model')
        # new_checkpoint = {'module.' + k: v for k, v in torch.load(segmentation_model_path).items()}
        # model.load_state_dict(
        #     new_checkpoint   
        # )

        model.load_state_dict(
            torch.load(segmentation_model_path)
        )

    model.to(device)
    # return model.module
    return model

def soft_update(target_dqn, policy_dqn, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    Args:
        target_dqn (PyTorch model): weights will be updated
        policy_dqn (PyTorch model): source model that provides the new weights
        tau (float): interpolation parameter 
    """
    for target_param, policy_param in zip(target_dqn.parameters(), policy_dqn.parameters()):
        target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

def create_dqn(state_dim:int, action_dim:int, device:int):
    device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
    policy_dqn = QueryNetworkDQN(state_size=state_dim, action_size=action_dim, device=device)
    target_dqn = QueryNetworkDQN(state_size=state_dim, action_size=action_dim, device=device)

    policy_dqn.to(device)
    target_dqn.to(device)

    return policy_dqn, target_dqn

def create_dataset(partition_json_fn:str, dataset_type:str, annotations_dir:str, images_dir:str, batch_size:int=2):
    dataset = RalisDataset(
        annotations_dir,
        images_dir,
        dataset_type,
        partition_json_fn
    )

    state_dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=RalisCollateFn()
    )

    return state_dataloader


def ralis_main(model:UNet, policy_dqn:QueryNetworkDQN, target_dqn:QueryNetworkDQN, state_dataloader:DataLoader, reward_dataloader:DataLoader, partition_json:dict, logger:Logger):

    memory = ExperienceReplay(capacity=10000)
    dqn_trainer = DQNTrainer(target_dqn, policy_dqn)

    policy_losses = []
    rewards = []
    steps_done = 0

    test_accuracy, test_mean_iou, cur_test_iou_per_class = compute_reward(model, testing_dataloader)    
    best_test_performance = test_mean_iou

    logger.log_message(f'Performance Before Ralis - {cur_test_iou_per_class} Average IOU - {best_test_performance}')
    logger.log_new_line()

    # print(f'Performance Before Ralis - {cur_test_iou_per_class} Average IOU - {best_test_performance}')

    BUDGET = 3854 #(256 * 14)

    for episode in range(EPISODES):
        # [1280, 200] - state features

        state_features = compute_current_state(model, state_dataloader)
        cur_reward_accuracy, cur_reward_mean_iou, cur_reward_iou_per_class = compute_reward(model, reward_dataloader)
        logger.log_message(f'Cur Reward IOU: {cur_reward_iou_per_class}')
        logger.log_new_line()

        ralis_train_collate_fn = RalisTrainCollateFn(
            TRAIN_IMAGES_DIR, TRAIN_ANNOTATIONS_DIR, partition_json["training_dataset"], pool_size=50
        )          

        u_net_trainer = UNetTrainer(model, ralis_train_collate_fn)
        current_step = 0

        budget_reached = False
        episode_reward = 0.0
        
        total_episode_loss = 0.0
        total_episode_u_net_training_loss = 0.0        

        train_predictions = ralis_train_collate_fn.compute_train_prediction_maps(model, batch_size=6)
        sampled_pool = ralis_train_collate_fn.sample_k_pool_unlabelled() 
        actions, selected_actions = ralis_train_collate_fn.compute_action_features(policy_dqn, sampled_pool, train_predictions, state_features, steps_done)
        current_state = (state_features.to(policy_dqn.device), actions.to(policy_dqn.device))

        while len(ralis_train_collate_fn.labelled_pool) < BUDGET and not budget_reached:
            ''' 
            state_features - (1280, 200); actions - (256, pool_size, 200), selected_actions - (256, 1) [indices per pool]
            --- selected_actions - is the indices from (0 to pool_size) for each 
            '''

            training_loss, _, _ = u_net_trainer.train(episode=episode, step=current_step)
            next_reward_accuracy, next_reward_mean_iou, next_reward_iou_per_class = compute_reward(model, reward_dataloader)

            reward = next_reward_mean_iou - cur_reward_mean_iou
            # reward = (next_reward_iou_per_class - cur_reward_iou_per_class).sum(dim=-1)
            # reward = compute_weighted_reward(cur_reward_iou_per_class, next_reward_iou_per_class)
            logger.log_line()
            logger.log_message(f'Next Reward IOU: {next_reward_iou_per_class} - Delta Rewards IOU: {next_reward_iou_per_class - cur_reward_iou_per_class}')
            logger.log_new_line()
            # print(f'Next Reward IOU: {next_reward_iou_per_class} - Delta Rewards IOU: {next_reward_iou_per_class - cur_reward_iou_per_class}')

            steps_done += 1

            if len(ralis_train_collate_fn.labelled_pool) < BUDGET and not budget_reached:
                next_state_features = compute_current_state(model, state_dataloader)
                train_predictions = ralis_train_collate_fn.compute_train_prediction_maps(model, batch_size=6)
                sampled_pool = ralis_train_collate_fn.sample_k_pool_unlabelled() 
                next_actions, next_selected_actions = ralis_train_collate_fn.compute_action_features(policy_dqn, sampled_pool, train_predictions, next_state_features, steps_done)

                next_state = (next_state_features.to(policy_dqn.device), next_actions.to(policy_dqn.device))
                memory.add_memory(
                    current_state=current_state, 
                    actions=selected_actions.to(policy_dqn.device),
                    next_state=next_state,
                    reward=reward.to(policy_dqn.device)
                )

                current_state = next_state
                # cur_reward_acc_per_class = next_reward_acc_per_class
                cur_reward_iou_per_class = next_reward_iou_per_class
                cur_reward_mean_iou = next_reward_mean_iou

                # state_features = next_state_features 
                selected_actions = next_selected_actions

            else:
                next_state = (None, None) 
                selected_actions=None
                memory.add_memory(
                    current_state=current_state, 
                    actions=selected_actions.to(policy_dqn.device),
                    reward=reward.to(policy_dqn.device), 
                    next_state=next_state
                )
                budget_reached = True
                selected_actions = next_selected_actions     

            loss = dqn_trainer.train_double_dqn(memory)
            episode_reward += reward

            total_episode_u_net_training_loss += training_loss
            total_episode_loss += loss 

            logger.log_message(f'Episode {episode} - step {current_step} - reward: {reward} - policy net loss {loss.item():.4f} - training loss {training_loss:.4f} - Selected Regions {len(ralis_train_collate_fn.labelled_pool)}')            

            # print(f'Episode {episode} - step {current_step} - reward: {reward} - policy net loss {loss.item():.4f} - training loss {training_loss:.4f} - Selected Regions {len(ralis_train_collate_fn.labelled_pool)}')
            current_step += 1

            rewards.append(reward)
            policy_losses.append(loss.item())

            # del next_reward_acc_per_class
            # del next_reward_iou_per_class
            # del next_state_features

            if current_step % TARGET_UPDATE == 0:
                # print('Updating Target Network')
                logger.log_new_line()
                logger.log_message('Updating Target Network')
                # target_dqn.load_state_dict(policy_dqn.state_dict())
                soft_update(target_dqn, policy_dqn, TAU)

            torch.cuda.empty_cache()

        # plot_loss_curve(policy_losses, f'{OUTPUT_DIR}/loss_curve.png')
        # plot_reward_curve(rewards, f'{OUTPUT_DIR}/rewards_curve.png')
        
        total_episode_loss = total_episode_loss/current_step
        total_training_loss = total_episode_u_net_training_loss/current_step

        cur_test_acc_per_class, cur_test_iou_per_class = compute_reward(model, testing_dataloader)

        # BUDGET = BUDGET + int(BUDGET//2) 

        logger.log_line()
        logger.log_message(f'Episode Complete ------ Loss {total_episode_loss:.4f} U-Net Training Loss {total_training_loss:.4f}')
        logger.log_new_line()
        logger.log_message(f'Testing Dataset Accuracy {cur_test_acc_per_class} IOU {cur_test_iou_per_class}')
        # print(f'Episode Complete ------ Loss {total_episode_loss:.4f} U-Net Training Loss {total_training_loss:.4f}')
        # print(f'Testing Dataset Accuracy {cur_test_acc_per_class} IOU {cur_test_iou_per_class}')

        if cur_test_iou_per_class.sum(dim=-1).item()/len(cur_test_iou_per_class) > best_test_performance:
            torch.save(
                policy_dqn.state_dict(), f'{OUTPUT_DIR}/policy_dqn.pt'
            )

            best_test_performance = cur_test_iou_per_class.sum(dim=-1).item()/len(cur_test_iou_per_class)

            # print(f'Saving Policy Net at U-Nets best Performance - {cur_test_iou_per_class} Average IOU - {best_test_performance}')
            logger.log_line()
            logger.log_message(f'Saving Policy Net at U-Nets best Performance - {cur_test_iou_per_class} Average IOU - {best_test_performance}')
            logger.log_new_line()

        # print(f'Reloading Model')
        logger.log_message(f'Reloading U-Net Model')
        model = create_segmentation_model(
            'model_checkpoints/best-model.pt', device_ids=DEVICE_IDS
        )

if __name__ == "__main__":

    DEVICE_IDS = [0] #[3, 8, 4] #first two for U-net while running on Weasly

    TRAIN_ANNOTATIONS_DIR = "gtFine_trainvaltest/gtFine/train" #"../gtFine/train"
    TRAIN_IMAGES_DIR = "leftImg8bit_beta_0.02/train" #"../leftImg8bit_beta_0.02/train"

    VAL_ANNOTATIONS_DIR = "gtFine_trainvaltest/gtFine/val" #"../gtFine/val"
    VAL_IMAGES_DIR = "leftImg8bit_beta_0.02/val" #"../leftImg8bit_beta_0.02/val"

    EPISODES = 4
    # BUDGET = 3854 #(256 * 14)

    TARGET_UPDATE = 5
    TAU = 0.005
    OUTPUT_DIR = "outputs_hazy_3"

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    logger = Logger(OUTPUT_DIR)

    model = create_segmentation_model(
        'model_checkpoints/best-model.pt', device_ids=DEVICE_IDS
    )

    '''
    action_dim=260 if hist features are used else 200
    '''
    policy_dqn, target_dqn = create_dqn(
        state_dim=200, action_dim=200, device=DEVICE_IDS[-1]
    )

    partition_json = json.load(open("hazy_split_files_new.json"))

    state_dataloader = create_dataset("hazy_split_files_new.json", "state_dataset", TRAIN_ANNOTATIONS_DIR, TRAIN_IMAGES_DIR, batch_size=1)
    reward_dataloader = create_dataset("hazy_split_files_new.json", "reward_dataset", TRAIN_ANNOTATIONS_DIR, TRAIN_IMAGES_DIR)
    # validation_dataloader = create_dataset("hazy_split_files_new.json", "validation_dataset", TRAIN_ANNOTATIONS_DIR, TRAIN_IMAGES_DIR)
    testing_dataloader = create_dataset("hazy_split_files_new.json", "testing_dataset", VAL_ANNOTATIONS_DIR, VAL_IMAGES_DIR)

    ralis_main(
        model, policy_dqn, target_dqn, state_dataloader, reward_dataloader, partition_json, logger
    )

    '''
    IOU tensor([0.0000, 0.9034, 0.3100, 0.1316, 0.3321, 0.5158, 0.2044, 0.3390])
    '''
