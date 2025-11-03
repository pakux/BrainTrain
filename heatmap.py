import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import nibabel as nib
from dataloaders import dataloader
from architectures import sfcn_cls, sfcn_ssl2, head, lora_layers
from nilearn.plotting import plot_stat_map

# Import configuration
import config as cfg


def load_model(model_path, device):
    """Load trained model"""
    # Create model architecture based on training mode
    if cfg.TRAINING_MODE == 'sfcn':
        model = sfcn_cls.SFCN(output_dim=cfg.N_CLASSES).to(device)
    
    elif cfg.TRAINING_MODE in ['linear', 'ssl-finetuned', 'lora']:
        backbone = sfcn_ssl2.SFCN()
        
        # For LoRA, apply LoRA layers before loading weights
        if cfg.TRAINING_MODE == 'lora':
            backbone = lora_layers.apply_lora_to_model(
                backbone,
                rank=cfg.LORA_RANK,
                alpha=cfg.LORA_ALPHA,
                target_modules=cfg.LORA_TARGET_MODULES
            )
        
        model = head.ClassifierHeadMLP_(backbone, output_dim=cfg.N_CLASSES).to(device)
    
    else:
        raise ValueError(f"Invalid TRAINING_MODE: {cfg.TRAINING_MODE}")
    
    # Load checkpoint
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    print(f"Successfully loaded {cfg.TRAINING_MODE} model")
    model.eval()
    return model


def find_last_conv_layer(model):
    """Find the last convolutional layer in the model"""
    last_conv = None
    last_conv_name = None
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv3d):
            last_conv = module
            last_conv_name = name
    
    if last_conv is None:
        raise ValueError("No Conv3d layer found in model!")
    
    print(f"Using layer for GradCAM: {last_conv_name}")
    return last_conv


def compute_gradcam(model, image, target_layer, target='logit_diff', class_idx=None, mode='magnitude'):
    """
    Compute GradCAM attention map using intermediate layer activations.
    
    Args:
        model: PyTorch model
        image: Input image tensor [1,1,D,H,W]
        target_layer: Layer to compute gradients for
        target: 'logit_diff' (default), 'pred', or 'target_class'
        class_idx: Target class index (only used if target='target_class')
        mode: 'magnitude' (ReLU applied) or 'signed' (preserve sign)
    
    Returns:
        gradcam: Attention map [D,H,W]
        pred_class: Predicted class
        confidence: Prediction confidence
    """
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        activations.append(output.detach())
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())
    
    # Register hooks
    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)
    
    try:
        # Forward pass
        model.zero_grad(set_to_none=True)
        logits = model(image)
        probs = torch.softmax(logits, dim=1)
        
        # Determine target and get predictions
        if target == 'logit_diff' and logits.shape[1] == 2:
            score = (logits[0, 1] - logits[0, 0])
            pred_class = int(probs[0, 1] > probs[0, 0])
            conf = probs[0, pred_class].item()
        elif target == 'pred':
            pred_class = torch.argmax(probs, dim=1).item()
            score = logits[0, pred_class]
            conf = probs[0, pred_class].item()
        elif target == 'target_class':
            if class_idx is None:
                raise ValueError("Must specify class_idx when target='target_class'")
            pred_class = torch.argmax(probs, dim=1).item()
            score = logits[0, class_idx]
            conf = probs[0, class_idx].item()
        else:
            raise ValueError(f"Unknown target: {target}")
        
        # Backward pass
        score.backward()
        
        # Compute GradCAM
        act = activations[0]  # [1, C, D, H, W]
        grad = gradients[0]   # [1, C, D, H, W]
        
        # Global average pooling of gradients
        weights = grad.mean(dim=(2, 3, 4), keepdim=True)  # [1, C, 1, 1, 1]
        
        # Weighted combination of activation maps
        cam = (weights * act).sum(dim=1, keepdim=True)  # [1, 1, D, H, W]
        
        # Apply mode
        if mode == 'magnitude':
            cam = F.relu(cam)
        # else: keep signed values (mode == 'signed')
        
        cam = cam.squeeze().cpu().numpy()  # [D, H, W]
        
        # Normalize
        if mode == 'magnitude':
            # Normalize to [0, 1]
            if cam.max() > 0:
                cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:  # signed
            # Normalize to [-1, 1] while preserving sign
            max_abs = np.abs(cam).max()
            if max_abs > 0:
                cam = cam / max_abs
        
        return cam, pred_class, conf
    
    finally:
        handle_f.remove()
        handle_b.remove()


def compute_saliency(model, image, target='logit_diff', class_idx=None, mode='magnitude'):
    """
    Compute saliency map (gradient of output w.r.t. input).
    
    Args:
        model: PyTorch model
        image: Input image tensor [1,1,D,H,W]
        target: 'logit_diff' (default), 'pred', or 'target_class'
        class_idx: Target class index (only used if target='target_class')
        mode: 'magnitude' (absolute value) or 'signed' (preserve sign)
    
    Returns:
        saliency: Attention map [D,H,W]
        pred_class: Predicted class
        confidence: Prediction confidence
    """
    image.requires_grad = True
    model.zero_grad(set_to_none=True)
    
    logits = model(image)
    probs = torch.softmax(logits, dim=1)
    
    # Determine target
    if target == 'logit_diff' and logits.shape[1] == 2:
        score = (logits[0, 1] - logits[0, 0])
        pred_class = int(probs[0, 1] > probs[0, 0])
        conf = probs[0, pred_class].item()
    elif target == 'pred':
        pred_class = torch.argmax(probs, dim=1).item()
        score = logits[0, pred_class]
        conf = probs[0, pred_class].item()
    elif target == 'target_class':
        if class_idx is None:
            raise ValueError("Must specify class_idx when target='target_class'")
        pred_class = torch.argmax(probs, dim=1).item()
        score = logits[0, class_idx]
        conf = probs[0, class_idx].item()
    else:
        raise ValueError(f"Unknown target: {target}")
    
    # Backward
    score.backward()
    
    # Get saliency map
    if mode == 'magnitude':
        saliency = image.grad.data.abs().squeeze().cpu().numpy()  # [D, H, W]
        # Normalize to [0, 1]
        if saliency.max() > 0:
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    else:  # signed
        saliency = image.grad.data.squeeze().cpu().numpy()  # [D, H, W]
        # Normalize to [-1, 1] while preserving sign
        max_abs = np.abs(saliency).max()
        if max_abs > 0:
            saliency = saliency / max_abs
    
    image.requires_grad = False
    return saliency, pred_class, conf


def load_image(path, device):
    """Load and preprocess image"""
    img_data = np.load(path)
    
    # Display version (8-bit for overlays)
    p1, p99 = np.percentile(img_data, (1, 99))
    img_np = np.clip(img_data, p1, p99)
    img_np = ((img_np - img_np.min()) / max(img_np.max() - img_np.min(), 1e-8) * 255).astype(np.uint8)
    
    # Model input version
    img_t = torch.from_numpy(img_data).float().unsqueeze(0).unsqueeze(0).to(device)
    return img_t, img_np


def save_visualization(heatmap, image, name, output_dir, signed=False, affine=None):
    """Save heatmap visualization as overlays AND NIfTI files"""
    import nibabel as nib
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Default affine if none provided
    if affine is None:
        affine = np.eye(4)
    
    # Save brain image as NIfTI
    brain_nifti_path = os.path.join(output_dir, f"{name}_brain.nii.gz")
    brain_img = nib.Nifti1Image(image, affine)
    nib.save(brain_img, brain_nifti_path)
    print(f"Saved brain NIfTI: {brain_nifti_path}")
    
    # Save heatmap as NIfTI
    heatmap_nifti_path = os.path.join(output_dir, f"{name}_heatmap.nii.gz")
    heatmap_img = nib.Nifti1Image(heatmap, affine)
    nib.save(heatmap_img, heatmap_nifti_path)
    print(f"Saved heatmap NIfTI: {heatmap_nifti_path}")
    
    # Use niilearn to plot heatmaps

    cmap = "RdBlu_r" if signed else "hot"
    vmin = np.abs(heat_slice).max() if signed else 0.05 # vmin and vmax are used for the whole 3D image thus also those slices that
    vmax = np.abs(heat_slice).max() if signed else 0.5  # are not shown. This might lead to very pale heatmaps => reducing vmax to 0.5

    plot_stat_map(heatmap_img, 
            bg_img=brain_img,
            # cut_coords=[48,48,48], # If set to this the center coordinates are choosen, otherwise the region with highest weight
            radiological=True, # R/L are switched 
            cmap=cmap, 
            black_bg=True, 
            annotate=True, 
            draw_cross=False,
            # alpha=0.5, 
            # transparency_range=[0.5,1],
            transparency=0.8, # heatmap_img[0],
            title=f"{subtest.upper()}",
            threshold=0.05,
            vmin=0.05,
            vmax=0.5
            )
    # plt.show()
    save_path = os.path.join(output_dir, f"{name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved PNG: {save_path}")


 




def generate_heatmaps(heatmap_dir, attention_method='gradcam', 
                      attention_mode='magnitude', mode='top_individual', top_n=5, 
                      attention_target='logit_diff', attention_class_idx=None):
    """
    Main function to generate heatmaps
    
    Args:
        heatmap_dir: Directory to save heatmap visualizations in png and quantitative regional analysis in csv 
        attention_method: 'gradcam' or 'saliency'
        attention_mode: 'magnitude' or 'signed'
        mode: 'single', 'average', or 'top_individual'
        top_n: Number of top samples to visualize
        attention_target: 'logit_diff', 'pred', or 'target_class'
        attention_class_idx: Target class index (only for 'target_class')
    """
    device = cfg.DEVICE
    
    print("\n" + "="*70)
    print("HEATMAP GENERATION")
    print("="*70)
    print(f"Test cohort: {cfg.TEST_COHORT}")
    print(f"Attention method: {attention_method}")
    print(f"Attention mode: {attention_mode}")
    print(f"Visualization mode: {mode}")
    print(f"Top N: {top_n}")
    
    # Create output directories
    os.makedirs(heatmap_dir, exist_ok=True)
    model_path=f'{cfg.MODEL_DIR}/{cfg.TRAINING_MODE}/{cfg.EXPERIMENT_NAME}.pth'

    # Load model
    model = load_model(model_path, device)
    
    # Get target layer for GradCAM
    if attention_method == 'gradcam':
        target_layer = find_last_conv_layer(model)
    # Load CSV data
    df = pd.read_csv(cfg.CSV_TEST)
    print(f"Test dataset size: {len(df)}")
    # Create test dataset
    test_dataset = dataloader.BrainDataset(
        csv_file=cfg.CSV_TEST,
        root_dir=cfg.TENSOR_DIR_TEST,
        column_name=cfg.COLUMN_NAME,
        num_rows=None,
        num_classes=cfg.N_CLASSES,
        task='classification'
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Process images
    results = []
    signed = (attention_mode == 'signed')
    
    print("\nGenerating heatmaps...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        eid = str(row['eid'])
        label = int(row[cfg.COLUMN_NAME])
        
        # Construct image path
        image_path = os.path.join(cfg.TENSOR_DIR_TEST, f"{eid}.npy")
        
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Load image
        image_t, image_np = load_image(image_path, device)
        
        # Compute attention
        if attention_method == 'gradcam':
            att_map, pred_class, confidence = compute_gradcam(
                model, image_t, target_layer,
                target=attention_target,
                class_idx=attention_class_idx,
                mode=attention_mode
            )
        else:  # saliency
            att_map, pred_class, confidence = compute_saliency(
                model, image_t,
                target=attention_target,
                class_idx=attention_class_idx,
                mode=attention_mode
            )
        
        results.append({
            'heatmap': att_map,
            'image': image_np,
            'confidence': confidence,
            'pred_class': pred_class,
            'true_class': label,
            'eid': eid
        })
        
        if mode == 'single':
            break
    
    # Generate visualizations
    print("\nCreating visualizations...")
    
    if mode == 'single':
        result = results[0]
        save_visualization(
            result['heatmap'], result['image'],
            f"single_{result['eid']}_pred{result['pred_class']}_conf{result['confidence']:.3f}",
            heatmap_dir,
            signed=signed
        )
    
    elif mode == 'average':
        top_results = sorted(results, key=lambda x: x['confidence'], reverse=True)[:top_n]
        avg_heatmap = np.mean([r['heatmap'] for r in top_results], axis=0)
        save_visualization(
            avg_heatmap, results[0]['image'], 
            f"average_top{top_n}",
            heatmap_dir, 
            signed=signed
        )
    
    elif mode == 'top_individual':
        # Get top confident predictions for class 1
        positive_results = [r for r in results if r['pred_class'] == 1]
        top_positive = sorted(positive_results, key=lambda x: x['confidence'], reverse=True)[:top_n]
        
        for i, result in enumerate(top_positive):
            save_visualization(
                result['heatmap'], result['image'],
                f"top{i+1}_{result['eid']}_conf{result['confidence']:.3f}",
                heatmap_dir,
                signed=signed
            )
    
    # Save summary
    summary_df = pd.DataFrame([{
        'eid': r['eid'],
        'pred_class': r['pred_class'],
        'true_class': r['true_class'],
        'confidence': r['confidence']
    } for r in results])
    
    summary_path = os.path.join(heatmap_dir, 'heatmap_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")
    print(f"Heatmaps saved to: {heatmap_dir}")
    print("\nHeatmap generation completed!")
    
    return results


def main():
    """Main function"""
    # Setup
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.DEVICE)
    torch.manual_seed(42)
    
    print("\n" + "="*70)
    print("HEATMAP CONFIGURATION")
    print("="*70)
    print(f"Training mode: {cfg.TRAINING_MODE}")
    print(f"Model: {cfg.MODEL_DIR}")
    print(f"Test cohort: {cfg.TEST_COHORT}")
    print(f"Test CSV: {cfg.CSV_TEST}")
    print("="*70)
    
    # Set heatmap parameters (modify these as needed)
    attention_method = cfg.ATTENTION_METHOD      # Options: 'gradcam', 'saliency'
    attention_mode = cfg.ATTENTION_MODE      # Options: 'magnitude', 'signed'
    mode = cfg.HEATMAP_MODE       # Options: 'single', 'average', 'top_individual'
    top_n = cfg.HEATMAP_TOP_N
    explainability_path = f'{cfg.COLUMN_NAME}/{cfg.TEST_COHORT}/{cfg.TRAINING_MODE}/{cfg.ATTENTION_METHOD}/{cfg.ATTENTION_MODE}/{cfg.EXPERIMENT_NAME}'
    
    # Create output directories
    heatmap_dir = os.path.join(cfg.EXPLAINABILITY_DIR, explainability_path)
    
    # Generate heatmaps
    generate_heatmaps(
        heatmap_dir=heatmap_dir,
        attention_method=attention_method,
        attention_mode=attention_mode,
        mode=mode,
        top_n=top_n,
        attention_target='logit_diff',
        attention_class_idx=None
    )
    
    print("\n✓ All done!")


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

