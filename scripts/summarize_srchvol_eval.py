"""Rough script to plot and compute derrivative searchvolume metrics like AUC.

No argparse or configs are loaded here. Set your config options at the top of 
the script.
"""
import pandas
import numpy as np
import matplotlib.pyplot as plt
import os
import pdb
from collections import defaultdict
from scipy.interpolate import make_interp_spline

# Looping over the directories <root_path>/<exp_name>/<dataset>/<scene_name>
root_path = "srchvol_out"
exp_names = ["rayfronts_0", "rayfronts_10", "rayfronts_20",
             "unidir_semfronts_10", "unidir_semfronts_20", 
             "spherical_semfronts_10", "spherical_semfronts_20", 
             "sempose_0"]

datasets = ["TartanAirDataset"]
scene_names = ["Downtown" , "Factory", "ConstructionSiteOvercast", "AbandonedCableDay"]

include_end_value = False
# How much of the class has entered the observed area threshold. After which the
# search terminates for a given class
det_thresh = 0.5
# Store frequencey weighted metrics.
freq_weighted = False

# Plot visualizations. You may want to limit the classes and metrics to plot.
vis=False

out = f"{root_path}/srchvol_summary.csv"

extra_metrics = {
  # "metric_name": lambda m: m["scene_tp"]*2
}

# Metrics to compute AUC for.
metrics =["obsrv_occ_iou", "unobsrv_srchcutvol",
          "unobsrv_recall", "unobsrv_srchcutvol_recall"]

# Used for plotting.
pretty_labels = {
  "obsrv_occ_iou": "IoU",
  "unobsrv_srchcutvol": "SCV",
  "unobsrv_recall": "Recall",
  "unobsrv_srchcutvol_recall": "SCVR",
  "building": "Text Query: Building",
  "chimney": "Text Query: Chimney",
}
new_color_cycle = ['#d90368', '#00cc66', '#3c153b', '#3f37c9', '#1be7ff', '#3a0ca3']

# Set the default color cycle in Matplotlib
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=new_color_cycle)

def prettify(k):
  if k in pretty_labels:
    return pretty_labels[k]
  else:
    return k

smoothing = 5
csv_header = "exp_name,dataset,scene_name,det_thresh,"
if include_end_value:
  csv_header += ",".join([f"AUC_{x},END_{x}" for x in metrics])
else:
  csv_header += ",".join([f"AUC_{x}" for x in metrics])

csv_lines = [csv_header]

for dataset in datasets:
  for exp_name in exp_names:
    exp_scene_results = defaultdict(list)
    for scene_name in scene_names:
      print(f"{exp_name}/{dataset}/{scene_name}")
      path = f"{root_path}/{exp_name}/{dataset}/{scene_name}"


      classwise_path = os.path.join(path, "srchvol_online_classwise_results")

      filtered_results = defaultdict(list)
      cls_rs = list()

      total_freq = 0
      for fn in os.listdir(classwise_path):
        cls_id, cls_name = fn.split(".")[0].split("_")

        cls_r = pandas.read_csv(os.path.join(classwise_path, fn))
        for k, f in extra_metrics.items():
          cls_r[k] = f(cls_r)
        
        freq = cls_r.iloc[-1]["scene_tp"] + cls_r.iloc[-1]["scene_fn"]
        total_freq += freq
        exists_in_scene = freq > 0
        if not exists_in_scene:
          print(f"{fn} does not exist in scene")
          continue
        cls_rs.append((fn, cls_r, freq))

      ncols = 1
      nrows = int(np.ceil(len(cls_rs) / ncols))
      plt.figure(figsize=(5, 8))
      for i, (fn, cls_r, freq) in enumerate(cls_rs):
        if vis:
          plt.subplot(nrows, ncols, i+1)

        esm = ~(np.cumsum(cls_r["scene_gt_occ_vs_unobs_iou"] > det_thresh) > 0)
        step_sizes = np.diff(cls_r["step"], prepend=0)
        for c in metrics:
          na_mask = cls_r[c][esm].notna()
          auc = np.sum(cls_r[c][esm][na_mask]*step_sizes[esm][na_mask])
          auc = auc / np.sum(step_sizes[esm][na_mask])
          if vis:
            x = cls_r["step"][esm]
            y = cls_r[c][esm]
            window = np.ones(smoothing) / smoothing
            if np.all(np.isfinite(y)):
              pad_width = smoothing // 2
              y_padded = np.pad(y, pad_width, mode='edge')

              # Apply the convolution
              y_smooth = np.convolve(y_padded, window, mode='valid')
              x_smooth = x
              
            else:
              x_smooth = x
              y_smooth = y

            ls ='--' if c in ["unobsrv_srchcutvol", "unobsrv_recall"] else "-"
            plt.plot(x_smooth, y_smooth, label=f"{prettify(c)} AUC={auc:.2f}", linestyle=ls)
          
          if include_end_value:
            end = cls_r[c][esm].iloc[-1]
            filtered_results[c].append((auc, end, freq))
          else:
            filtered_results[c].append((auc, freq))

        if vis:
          plt.legend()
          plt.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
          plt.xlabel("Time Step", fontweight='bold')
          plt.ylabel("Value of Metric", fontweight='bold')
          plt.title(prettify(fn.split(".")[0].split("_")[-1]), fontweight='bold')
      if vis:
        plt.suptitle(prettify(exp_name), fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{root_path}/{exp_name}_{scene_name}.pdf")

        plt.show()

      csv_line = [exp_name,dataset,scene_name, str(det_thresh)]
      for k,v in filtered_results.items():
        if include_end_value:
          auc, end, freq = np.array(v).T
          if freq_weighted:
            auc = np.nanmean(auc*freq/total_freq)
            end = np.nanmean(end*freq/total_freq)
          else:
            auc = np.nanmean(auc)
            end = np.nanmean(end)
          csv_line.extend([str(auc), str(end)])
          print(f"mAUC-{k} = {auc:.4f}, end-{k} = {end:.4f}")
          exp_scene_results[k].append((auc, end))
        else:
          auc, freq = np.array(v).T
          if freq_weighted:
            auc = np.nanmean(auc*freq/total_freq)
          else:
            auc = np.nanmean(auc)
          csv_line.extend([str(auc)])
          print(f"mAUC-{k} = {auc:.4f}")
          exp_scene_results[k].append(auc)
      
      csv_lines.append(",".join(csv_line))
    
    csv_line = [exp_name, dataset,"avg", str(det_thresh)]
    print(f"{exp_name}/{dataset}/Avg")
    for k, v in exp_scene_results.items():
      if include_end_value:
        auc, end = np.array(v).T
        auc = np.mean(auc)
        end = np.mean(end)
        csv_line.extend([str(auc), str(end)])
        print(f"mAUC-{k} = {auc:.4f}, end-{k} = {end:.4f}")
      else:
        auc = np.mean(v)
        csv_line.extend([str(auc)])
        print(f"mAUC-{k} = {auc:.4f}")

    csv_lines.append(",".join(csv_line))

with open(out, "w") as f:
  f.writelines([f"{x}\n" for x in csv_lines])