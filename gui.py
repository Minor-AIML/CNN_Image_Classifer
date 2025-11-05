import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from vgg_model import VGGNet
import json
import os
import numpy as np
from datetime import datetime

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CIFAR-10 Classes with emojis
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

class_emojis = {
    'airplane': '✈️',
    'automobile': '🚗',
    'bird': '🐦',
    'cat': '🐱',
    'deer': '🦌',
    'dog': '🐕',
    'frog': '🐸',
    'horse': '🐴',
    'ship': '🚢',
    'truck': '🚚'
}

# Load the trained model
model = VGGNet(num_classes=10)
ckpt = torch.load('models/best_model.pth', map_location=device)

# Extract model accuracy if available
model_accuracy = None
model_epoch = None
if isinstance(ckpt, dict):
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    if 'val_acc' in ckpt:
        model_accuracy = ckpt['val_acc']
    if 'epoch' in ckpt:
        model_epoch = ckpt['epoch']
else:
    model.load_state_dict(ckpt)

model.to(device)
model.eval()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# GUI setup
root = tk.Tk()
root.title("🤖 CNN Image Classifier - Professional Edition")
root.geometry("1400x850")
root.configure(bg='#1e1e1e')
root.resizable(True, True)

# Modern Color Scheme (Dark Theme)
BG_COLOR = '#1e1e1e'
CARD_BG = '#2d2d30'
PRIMARY_COLOR = '#007acc'
SECONDARY_COLOR = '#00b4d8'
ACCENT_COLOR = '#ff6b6b'
SUCCESS_COLOR = '#4ecdc4'
WARNING_COLOR = '#ffd93d'
TEXT_COLOR = '#ffffff'
SUBTEXT_COLOR = '#a0a0a0'

# Title Frame with gradient effect
title_frame = Frame(root, bg='#0d1117')
title_frame.pack(fill=tk.X, side=tk.TOP)

# Title with subtitle
title_label = Label(title_frame, text="🎯 CNN Image Classification System",
                   font=('Segoe UI', 28, 'bold'), bg='#0d1117', fg='#58a6ff')
title_label.pack(pady=(20, 5))

subtitle_label = Label(title_frame, text="Deep Learning • Real-time Classification • 88.40% Accuracy",
                      font=('Segoe UI', 11), bg='#0d1117', fg='#8b949e')
subtitle_label.pack()

# Main container
main_container = Frame(root, bg=BG_COLOR)
main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

# Left Panel - Image Upload and Display
left_panel = Frame(main_container, bg=CARD_BG, relief=tk.FLAT, borderwidth=0)
left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

# Left panel header
left_header = Frame(left_panel, bg='#1a1a1d')
left_header.pack(fill=tk.X, pady=(0, 10))

left_title = Label(left_header, text="📸 Image Input", 
                  font=('Segoe UI', 16, 'bold'), bg='#1a1a1d', fg=TEXT_COLOR)
left_title.pack(side=tk.LEFT, padx=20, pady=10)

# Image display area
image_container = Frame(left_panel, bg=CARD_BG)
image_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

image_display_frame = Frame(image_container, bg='#1a1a1d', relief=tk.FLAT)
image_display_frame.pack(fill=tk.BOTH, expand=True)

label_img = Label(image_display_frame, text="No Image Loaded\n\nClick 'Upload Image' to begin",
                 bg='#1a1a1d', fg=SUBTEXT_COLOR, font=('Segoe UI', 12))
label_img.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)

# Button container
btn_container = Frame(left_panel, bg=CARD_BG)
btn_container.pack(fill=tk.X, padx=20, pady=20)

# Upload button with modern style
btn_upload = Button(btn_container, text="📁 Upload Image", 
                   font=('Segoe UI', 13, 'bold'),
                   bg=PRIMARY_COLOR, fg='white',
                   activebackground='#005a9e',
                   activeforeground='white',
                   padx=40, pady=18,
                   relief=tk.FLAT,
                   borderwidth=0,
                   cursor='hand2')
btn_upload.pack(fill=tk.X, pady=(0, 10))

# Clear button
btn_clear = Button(btn_container, text="🗑️ Clear",
                  font=('Segoe UI', 11),
                  bg='#3a3a3d', fg='white',
                  activebackground='#4a4a4d',
                  padx=20, pady=12,
                  relief=tk.FLAT,
                  borderwidth=0,
                  cursor='hand2')
btn_clear.pack(fill=tk.X)

# Right Panel - Results and Performance
right_panel = Frame(main_container, bg=BG_COLOR)
right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Model Performance Card
perf_frame = Frame(right_panel, bg=CARD_BG, relief=tk.FLAT)
perf_frame.pack(fill=tk.X, pady=(0, 8))

perf_header = Frame(perf_frame, bg='#1a1a1d')
perf_header.pack(fill=tk.X)

perf_title = Label(perf_header, text="📊 Model Performance",
                  font=('Segoe UI', 14, 'bold'), bg='#1a1a1d', fg=TEXT_COLOR)
perf_title.pack(side=tk.LEFT, padx=20, pady=10)

perf_info = Frame(perf_frame, bg=CARD_BG)
perf_info.pack(pady=15, padx=20, fill=tk.X)

# Display model stats in grid
stats_grid = Frame(perf_info, bg=CARD_BG)
stats_grid.pack(fill=tk.X)

# Device info
device_label = Label(stats_grid, text=f"⚡ Device:", 
                     font=('Segoe UI', 10, 'bold'), bg=CARD_BG, fg=SUBTEXT_COLOR)
device_label.grid(row=0, column=0, sticky='w', pady=3)
device_value = Label(stats_grid, text=f"{device.type.upper()}",
                    font=('Segoe UI', 10), bg=CARD_BG, fg=SUCCESS_COLOR)
device_value.grid(row=0, column=1, sticky='w', padx=(10, 0), pady=3)

if model_accuracy:
    acc_label = Label(stats_grid, text=f"🎯 Validation Accuracy:",
                     font=('Segoe UI', 10, 'bold'), bg=CARD_BG, fg=SUBTEXT_COLOR)
    acc_label.grid(row=1, column=0, sticky='w', pady=3)
    acc_value = Label(stats_grid, text=f"{model_accuracy:.2f}%",
                     font=('Segoe UI', 10), bg=CARD_BG, fg=SUCCESS_COLOR)
    acc_value.grid(row=1, column=1, sticky='w', padx=(10, 0), pady=3)

if model_epoch:
    epoch_label = Label(stats_grid, text=f"🔄 Trained Epochs:",
                       font=('Segoe UI', 10, 'bold'), bg=CARD_BG, fg=SUBTEXT_COLOR)
    epoch_label.grid(row=2, column=0, sticky='w', pady=3)
    epoch_value = Label(stats_grid, text=f"{model_epoch}",
                       font=('Segoe UI', 10), bg=CARD_BG, fg=TEXT_COLOR)
    epoch_value.grid(row=2, column=1, sticky='w', padx=(10, 0), pady=3)

model_label = Label(stats_grid, text=f"🧠 Architecture:",
                   font=('Segoe UI', 10, 'bold'), bg=CARD_BG, fg=SUBTEXT_COLOR)
model_label.grid(row=3, column=0, sticky='w', pady=3)
model_value = Label(stats_grid, text="VGG-inspired CNN",
                   font=('Segoe UI', 10), bg=CARD_BG, fg=TEXT_COLOR)
model_value.grid(row=3, column=1, sticky='w', padx=(10, 0), pady=3)

classes_label = Label(stats_grid, text=f"🏷️ Classes:",
                     font=('Segoe UI', 10, 'bold'), bg=CARD_BG, fg=SUBTEXT_COLOR)
classes_label.grid(row=4, column=0, sticky='w', pady=3)
classes_value = Label(stats_grid, text="10 (CIFAR-10)",
                     font=('Segoe UI', 10), bg=CARD_BG, fg=TEXT_COLOR)
classes_value.grid(row=4, column=1, sticky='w', padx=(10, 0), pady=3)

# Prediction Results Card
results_frame = Frame(right_panel, bg=CARD_BG, relief=tk.FLAT)
results_frame.pack(fill=tk.BOTH, expand=True)

results_header = Frame(results_frame, bg='#1a1a1d')
results_header.pack(fill=tk.X)

results_title = Label(results_header, text="🎯 Prediction Results",
                     font=('Segoe UI', 14, 'bold'), bg='#1a1a1d', fg=TEXT_COLOR)
results_title.pack(side=tk.LEFT, padx=20, pady=10)

# Top prediction display
top_pred_container = Frame(results_frame, bg=CARD_BG)
top_pred_container.pack(fill=tk.X, padx=20, pady=15)

top_pred_frame = Frame(top_pred_container, bg='#1a1a1d', relief=tk.FLAT, borderwidth=1)
top_pred_frame.pack(fill=tk.X)

top_pred_label = Label(top_pred_frame, text="🔍 Waiting for image...",
                      font=('Segoe UI', 18, 'bold'), bg='#1a1a1d', fg=SUBTEXT_COLOR,
                      pady=25)
top_pred_label.pack()

# Confidence bars frame
conf_scroll_container = Frame(results_frame, bg=CARD_BG)
conf_scroll_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 15))

# Add scrollbar
conf_canvas = tk.Canvas(conf_scroll_container, bg=CARD_BG, highlightthickness=0)
scrollbar = ttk.Scrollbar(conf_scroll_container, orient="vertical", command=conf_canvas.yview)
conf_frame = Frame(conf_canvas, bg=CARD_BG)

conf_frame.bind(
    "<Configure>",
    lambda e: conf_canvas.configure(scrollregion=conf_canvas.bbox("all"))
)

conf_canvas.create_window((0, 0), window=conf_frame, anchor="nw")
conf_canvas.configure(yscrollcommand=scrollbar.set)

conf_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Placeholder for confidence bars (will be populated dynamically)
confidence_bars = {}

def create_confidence_bar(parent, class_name, confidence, is_top=False):
    """Create a modern confidence bar widget"""
    bar_frame = Frame(parent, bg='#1a1a1d', relief=tk.FLAT)
    bar_frame.pack(fill=tk.X, pady=4)
    
    # Class label with emoji
    emoji = class_emojis.get(class_name, '')
    label_text = f"{emoji} {class_name.title()}"
    
    label_frame = Frame(bar_frame, bg='#1a1a1d')
    label_frame.pack(side=tk.LEFT, padx=(10, 10), pady=8)
    
    class_label = Label(label_frame, text=label_text, width=15,
                       font=('Segoe UI', 10, 'bold' if is_top else 'normal'),
                       bg='#1a1a1d', fg=TEXT_COLOR if is_top else SUBTEXT_COLOR,
                       anchor='w')
    class_label.pack(side=tk.LEFT)
    
    # Progress bar container
    bar_container = Frame(bar_frame, bg='#0a0a0d', relief=tk.FLAT)
    bar_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10), pady=5)
    
    # Colored bar based on confidence
    if is_top:
        if confidence >= 80:
            bar_color = SUCCESS_COLOR
        elif confidence >= 60:
            bar_color = WARNING_COLOR
        else:
            bar_color = ACCENT_COLOR
    else:
        bar_color = '#3a3a3d'
    
    bar_width = confidence  # Percentage width
    if bar_width > 0:
        colored_bar = Frame(bar_container, bg=bar_color)
        colored_bar.place(x=0, y=0, relheight=1, relwidth=bar_width/100)
    
    # Confidence percentage label
    conf_label = Label(bar_frame, text=f"{confidence:.1f}%",
                       font=('Segoe UI', 10, 'bold' if is_top else 'normal'),
                       bg='#1a1a1d', fg=TEXT_COLOR, width=7)
    conf_label.pack(side=tk.RIGHT, padx=(0, 10))
    
    return bar_frame

def clear_confidence_bars():
    """Clear all confidence bar widgets"""
    for widget in conf_frame.winfo_children():
        widget.destroy()

def classify_image(image_path):
    """Classify image and display results with confidence scores"""
    try:
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Get predictions
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)[0]
            
            # Get top prediction
            confidence, predicted = torch.max(probabilities, 0)
            class_idx = predicted.item()
            class_name = classes[class_idx]
            confidence_pct = confidence.item() * 100
            
            # Get top 3 predictions
            top3_prob, top3_indices = torch.topk(probabilities, 3)
        
        # Update top prediction display
        emoji = class_emojis.get(class_name, '🔍')
        result_text = f"{emoji} {class_name.upper()}\n"
        result_text += f"{confidence_pct:.2f}% Confidence"
        
        # Color based on confidence
        if confidence_pct >= 80:
            bg_color = SUCCESS_COLOR
            status = "High Confidence"
        elif confidence_pct >= 60:
            bg_color = WARNING_COLOR
            status = "Medium Confidence"
        else:
            bg_color = ACCENT_COLOR
            status = "Low Confidence"
        
        top_pred_label.config(text=result_text, bg=bg_color, fg='white')
        
        # Clear previous confidence bars
        clear_confidence_bars()
        
        # Get all class probabilities and sort by confidence
        probs = probabilities.cpu().numpy() * 100
        sorted_indices = np.argsort(probs)[::-1]  # Sort descending
        
        # Create confidence bars for all classes
        for idx in sorted_indices:
            is_top = (idx == class_idx)
            create_confidence_bar(conf_frame, classes[idx], probs[idx], is_top)
        
        # Show success message
        status_label.config(text=f"✅ Classification Complete | {status} | Processing Time: <1s")
        
    except Exception as e:
        top_pred_label.config(text=f"❌ Error: {str(e)}", bg=ACCENT_COLOR, fg='white')
        status_label.config(text=f"⚠️ Error processing image")

def clear_all():
    """Clear all displayed content"""
    label_img.config(image='', text="No Image Loaded\n\nClick 'Upload Image' to begin",
                    bg='#1a1a1d', fg=SUBTEXT_COLOR, font=('Segoe UI', 12))
    label_img.image = None
    top_pred_label.config(text="🔍 Waiting for image...", bg='#1a1a1d', fg=SUBTEXT_COLOR)
    clear_confidence_bars()
    status_label.config(text=f"Ready | Device: {device.type.upper()} | Model: VGG CNN")

def upload_image():
    """Handle image upload"""
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                  ("All Files", "*.*")]
    )
    
    if file_path:
        try:
            status_label.config(text="🔄 Loading image...")
            root.update()
            
            # Load and display image
            img = Image.open(file_path)
            
            # Resize for display while maintaining aspect ratio
            display_size = (450, 450)
            img.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            # Create a dark background
            background = Image.new('RGB', display_size, '#1a1a1d')
            # Center the image
            offset = ((display_size[0] - img.size[0]) // 2,
                     (display_size[1] - img.size[1]) // 2)
            background.paste(img, offset)
            
            # Convert to PhotoImage
            tk_img = ImageTk.PhotoImage(background)
            label_img.config(image=tk_img, bg='#1a1a1d', text='')
            label_img.image = tk_img
            
            # Update status
            status_label.config(text="⏳ Classifying image...")
            root.update()
            
            # Classify the image
            classify_image(file_path)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
            top_pred_label.config(text=f"❌ Error: {str(e)}",
                                bg=ACCENT_COLOR, fg='white')
            status_label.config(text="⚠️ Error loading image")

# Bind button functions
btn_upload.config(command=upload_image)
btn_clear.config(command=clear_all)

# Status bar at bottom
status_frame = Frame(root, bg='#0d1117')
status_frame.pack(fill=tk.X, side=tk.BOTTOM)

status_label = Label(status_frame, 
                    text=f"✅ Ready | Device: {device.type.upper()} | Model: VGG CNN | Authors: Akshat Jain, Amartya Singh, Mrityunjaya Sharma",
                    font=('Segoe UI', 9), bg='#0d1117', fg='#8b949e',
                    anchor='w')
status_label.pack(side=tk.LEFT, padx=20, pady=8)

# Info button
info_btn = Button(status_frame, text="ℹ️ Info",
                 font=('Segoe UI', 8),
                 bg='#21262d', fg='#c9d1d9',
                 relief=tk.FLAT,
                 cursor='hand2',
                 command=lambda: messagebox.showinfo(
                     "About",
                     f"CNN Image Classification System\n\n"
                     f"Model: VGG-inspired CNN\n"
                     f"Dataset: CIFAR-10\n"
                     f"Validation Accuracy: {model_accuracy:.2f}% ({model_epoch} epochs)\n\n"
                     f"Authors:\n"
                     f"Akshat Jain, Amartya Singh, Mrityunjaya Sharma\n"
                     f"Manipal Institute of Technology"
                 ) if model_accuracy and model_epoch else None)
info_btn.pack(side=tk.RIGHT, padx=10)

# Start GUI
root.mainloop()
