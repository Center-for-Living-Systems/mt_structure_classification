import matplotlib.pyplot as plt
import os
from PIL import Image
from matplotlib.cm import get_cmap
from matplotlib.patches import Patch
import numpy as np
import tifffile
import pandas as pd

def plot_training_images_by_label(dataframe, image_dir, output_dir, num_rows=4, num_cols=8):
    os.makedirs(output_dir, exist_ok=True)
    labels = dataframe["label"].unique()
    print(labels)
    for label in labels:
        label_df = dataframe[dataframe["label"] == label]
        image_files = [os.path.join(image_dir, f) for f in label_df["filename"][:num_rows * num_cols]]
        
        # if len(image_files) < num_rows * num_cols:
        #     continue  # Skip labels with insufficient images
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 8))
        fig.suptitle(f"Training Label: {label}", fontsize=14)
        
        for i, ax in enumerate(axes.flat):
            if i < len(image_files):
                img = Image.open(image_files[i])
                ax.imshow(img,cmap='gray',vmax=0.8*255,vmin=0)
            ax.axis("off")
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(os.path.join(output_dir, f"{label}_panel.png"))
        plt.close()


# Function to plot images in a 4x8 panel per predicted label
def plot_images_by_label(image_dir, output_dir, num_rows=4, num_cols=8):
    os.makedirs(output_dir, exist_ok=True)
    predicted_labels = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
    
    for label in predicted_labels:
        label_dir = os.path.join(image_dir, label)
        image_files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".png")]
        
        if len(image_files) < num_rows * num_cols:
            continue  # Skip labels with insufficient images
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 8))
        fig.suptitle(f"Predicted Label: {label}", fontsize=14)
        
        for i, ax in enumerate(axes.flat):
            img = Image.open(image_files[i])
            ax.imshow(img)
            ax.axis("off")
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(os.path.join(output_dir, f"{label}_panel.png"))
        plt.close()
    
    # Plot panels of images per predicted label
    plot_images_by_label(output_dir, os.path.join(output_dir, "classified_panels"))

# Function to plot images in a 4x8 panel per predicted label
def plot_images_by_label(image_dir, output_dir, num_rows=5, num_cols=10):
    os.makedirs(output_dir, exist_ok=True)
    predicted_labels = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
    
    for label in predicted_labels:
        label_dir = os.path.join(image_dir, label)
        image_files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".png")]
        
        if len(image_files) < num_rows * num_cols:
            continue  # Skip labels with insufficient images
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 8))
        fig.suptitle(f"Predicted Label: {label}", fontsize=14)
        
        for i, ax in enumerate(axes.flat):
            img = Image.open(image_files[i])
            ax.imshow(img,cmap='gray',vmax=0.8*255,vmin=0)
            ax.axis("off")
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(os.path.join(output_dir, f"{label}_panel.png"))
        plt.close()


# Function to plot images in a panel per class
def plot_training_images_by_label(dataframe, image_dir, output_dir, num_rows=4, num_cols=8):
    os.makedirs(output_dir, exist_ok=True)
    labels = dataframe["label"].unique()
    print(labels)
    for label in labels:
        label_df = dataframe[dataframe["label"] == label]
        image_files = [os.path.join(image_dir, f) for f in label_df["filename"][:num_rows * num_cols]]
        
        # if len(image_files) < num_rows * num_cols:
        #     continue  # Skip labels with insufficient images
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 8))
        fig.suptitle(f"Training Label: {label}", fontsize=14)
        
        for i, ax in enumerate(axes.flat):
            if i < len(image_files):
                img = Image.open(image_files[i])
                ax.imshow(img,cmap='gray',vmax=0.8*255,vmin=0)
            ax.axis("off")
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(os.path.join(output_dir, f"{label}_panel.png"))
        plt.close()



def df_add_group_guv_size(label_results_df,obj_df, test_image_dir):

    label_results_df["basename"] = label_results_df["filename"].str.replace(r"\.png$", "", regex=True)
    obj_df["basename"] = obj_df["obj_filename"].str.replace(r"\.tif$", "", regex=True)
    df = pd.merge(obj_df,label_results_df, on="basename", how="inner")

    guv_sizes = []  
    for _, row in df.iterrows():
        pred_label = str(row["label"])
        filename = row["filename"]    

        image_path = os.path.join(test_image_dir, filename)

        if not os.path.exists(image_path):
            guv_sizes.append(None)
            continue

        # Load and convert image to grayscale
        img = tifffile.imread(image_path)
        img_np = np.array(img)

        # Compute content size: number of pixels > 0
        content_size = np.sum(img_np/255 > 0.0773)  # You can adjust the threshold
        diameter = 2*np.sqrt(content_size/np.pi)
        guv_sizes.append(diameter)

    # Add to DataFrame and save
    df["GUV_size"] = guv_sizes

    df["GUV_dimater_micron"] = (df["GUV_size"]*0.2355)
    df["GUV_group"] = np.ceil(df["GUV_dimater_micron"]/7).clip(upper=4)
    group_labels = {1: "00~07", 2: "07~14", 3: "14~21",4:">21"}
    df["GUV_diameter"] = df["GUV_group"].replace(group_labels)
    return df

def stack_percentile(df,group_con = 'a_actinin',group_title='Tau with A_actinin'):
    # Filter out unwanted label
    df_filtered = df[df['label'] != 'ZZZ']
    colors = get_cmap('tab10')
    desired_order = ['Patch', 'Network', 'Filaments', 'Cluster']    
    color_map = {label: colors(9 - int(i * 3)) for i, label in enumerate(desired_order)}


    if group_con:
        # Parameters
        concentrations = df[group_con].unique()
        n_subplots = len(concentrations)
    else:
        concentrations = ['']
        n_subplots = 1        
    
        
    # Create figure
    fig, ax = plt.subplots(1, n_subplots, figsize=(2 * n_subplots, 4), sharey=True)
    if n_subplots == 1:
        ax = [ax]  # make iterable

    for i, conc in enumerate(concentrations):
        if group_con:
            sub_df = df_filtered[df_filtered[group_con] == conc]
        else:
            sub_df = df_filtered
            
        # Count and normalize
        count_df = sub_df.groupby(['GUV_diameter', 'label']).size().unstack(fill_value=0)
        percentage_df = count_df.div(count_df.sum(axis=1), axis=0) * 100
        
        plot_order = desired_order[::-1]   # put 'Patch' at the TOP visually

        percentage_df = percentage_df.reindex(columns=plot_order, fill_value=0)

        percentage_df.plot(
            kind='bar',
            stacked=True,
            color=[color_map[l] for l in plot_order],  # <-- same order as columns
            edgecolor='black',
            ax=ax[i],
            legend=False
        )


        ax[i].set_ylim(0, 100)
        ax[i].set_xlabel(f'GUV diameter\n{group_con} {conc}')

    # Common settings
    ax[0].set_ylabel('Percentage (%)')
    fig.suptitle( 'Stacked Bar Chart of GUV Diameter and '+ group_title)


    # Manual legend in your preferred label order (not reversed)
    legend_elements = [Patch(facecolor=color_map[l], edgecolor='black', label=l)
                    for l in desired_order]
    fig.legend(
        handles=legend_elements,      
        title="Type",
        bbox_to_anchor=(1.05, 0.8),
        loc='upper left'
    )
    plt.tight_layout()
    plt.show()
    return fig

