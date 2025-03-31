import torch
import wandb
from datetime import datetime
import yaml
import os
import subprocess
import numpy as np
import pandas as pd
import shutil
from data.dataloader import load_data
from model.network import create_model, cri_opt_sch
from model.utils import train, validate, test
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}\n')

def save_prediction_ratio(predictions, file_path):
    total_predictions = len(predictions)
    positive_predictions = np.sum(predictions)
    prediction_ratio = positive_predictions / total_predictions
    np.save(file_path, prediction_ratio)


def test(model, dataloader, device, save_ratio_path='prediction_ratio.npy'):
    """
    Test the model and calculate evaluation metrics.
    """
    model.eval()

    ground_truth = []
    predictions = []
    logits_all = []

    for batch in dataloader:
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels']

        with torch.inference_mode():
            logits = model(inputs, attention_mask).squeeze(1)

        preds = (logits > 0.6).long().cpu().numpy()
        logits_all.extend(logits.cpu().numpy().tolist())
        predictions.extend(preds)
        ground_truth.extend(labels.numpy().tolist())

    # Calculate metrics
    accuracy = accuracy_score(ground_truth, predictions)
    mcc = matthews_corrcoef(ground_truth, predictions)

    tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()
    sensitivity = tp / (tp + fn)  # True Positive Rate
    specificity = tn / (tn + fp)  # True Negative Rate
    fdr = fp / (fp + tp) if (fp + tp) > 0 else 0  # False Discovery Rate

    # Log and save prediction ratio
    save_prediction_ratio(predictions, save_ratio_path)

    # Return metrics
    metrics = {
        'accuracy': accuracy * 100,
        'mcc': mcc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'fdr': fdr,
    }

    return metrics


def train_model():
    print(f'{"="*30}{"TRAINING":^20}{"="*30}')

    best_acc = 0
    for epoch in range(config['epochs']):
        train_loss = train(model, train_data_loader, optimizer, criterion, scheduler, device)
        curr_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{config["epochs"]} - Train Loss: {train_loss}\tLR: {curr_lr}')
        val_loss, val_acc = validate(model, val_data_loader, criterion, device)
        print(f'Epoch {epoch+1}/{config["epochs"]} - Validation Loss: {val_loss}\tValidation Accuracy: {val_acc}\n')
        scheduler.step(val_acc)
        if not config['debug']:
            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'lr': curr_lr
            })

        if val_acc >= best_acc and not config['debug']:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'acc': val_acc,
                'lr': curr_lr
            }, f'{save_dir}/model.pt')
            print('Model Saved\n')
    wandb.finish()


# Load configurations and initialize
config = yaml.load(open('/Users/irene/PeptideBERT/config.yaml', 'r'), Loader=yaml.FullLoader)
config['device'] = device

train_data_loader, val_data_loader, test_data_loader = load_data(config)
config['sch']['steps'] = len(train_data_loader)

model = create_model(config)
criterion, optimizer, scheduler = cri_opt_sch(config, model)

if not config['debug']:
    run_name = f'{config["task"]}-{datetime.now().strftime("%m%d_%H%M")}'
    wandb.init(project='PeptideBERT', name=run_name)

    save_dir = f'./checkpoints/{run_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    shutil.copy('/Users/irene/PeptideBERT/config.yaml', f'{save_dir}/config.yaml')
    shutil.copy('/Users/irene/PeptideBERT/model/network.py', f'{save_dir}/network.py')

# Train the model
train_model()

# Load the best model and test
if not config['debug']:
    model.load_state_dict(torch.load(f'{save_dir}/model.pt')['model_state_dict'], strict=False)

metrics = test(model, test_data_loader, device, save_ratio_path=f'{save_dir}/prediction_ratio.npy')

# Print metrics
print(f"Test Accuracy: {metrics['accuracy']}%")
print(f"MCC: {metrics['mcc']}")
print(f"Sensitivity (TPR): {metrics['sensitivity']}")
print(f"Specificity (TNR): {metrics['specificity']}")
print(f"FDR: {metrics['fdr']}")
