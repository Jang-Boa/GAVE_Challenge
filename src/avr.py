import os
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import concurrent.futures
from collections import defaultdict
import traceback
import math
from tqdm import tqdm

def get_result(in_dir, out_dir, thresh=0.5):
    '''
        Obtain the partitioned graph from the probability graph, using only the R and B channels, and then place the intersection part in G
    '''
    
    out_Path = Path(out_dir)
    out_Path.mkdir(parents=True, exist_ok=True)
    
    for prob in Path(in_dir).glob("*.png"):
        out_filepath = out_Path / prob.name
        
        try:
            with Image.open(prob) as img:
                img_array = np.array(img)
                
            if img_array.dtype == np.uint8:
                img_array = img_array.astype(np.float32) / 255.0
            elif img_array.dtype == np.uint16:
                img_array = img_array.astype(np.float32) / 65535.0
                
            r = img_array[:, :, 0]
            g = img_array[:, :, 1]
            b = img_array[:, :, 2]
            
            r_c = (r > thresh).astype(np.float32)
            b_c = (b > thresh).astype(np.float32)
            g_c = (r_c>0) & (b_c>0)
            
            seg = np.zeros_like(img_array)
            
            seg[:, :, 0] = r_c
            seg[:, :, 2] = b_c
            
            seg[g_c, 0] = 0
            seg[g_c, 1] = 1
            seg[g_c, 2] = 0
            
            seg = (seg*255).astype(np.uint8)
            Image.fromarray(seg).save(out_filepath)
            
            # print(f"Save to {out_filepath}")            
                    
        except Exception as e:
            print(f"Reading img {prob} error{e}")


def reconstruct_disc_circle(disc_mask):
    """
        Input the segmented image and output the fitted circular mask image (of uint8 type, with values of 0 or 255)
    """
    if disc_mask.max() <= 1:
        disc_mask = (disc_mask * 255).astype(np.uint8)
    else:
        disc_mask = disc_mask.astype(np.uint8)

    contours, _ = cv2.findContours(disc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise ValueError("The outline was not found. Skip this picture")

    largest_contour = max(contours, key=cv2.contourArea)

    # Fit the minimum circumscribed circle
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    center = (int(x), int(y))
    radius = int(radius)

    circle_mask = np.zeros_like(disc_mask, dtype=np.uint8)
    cv2.circle(circle_mask, center, radius, 255, 1)

    return circle_mask

def process_all_images(in_dir, out_dir, suffixes=('.png', '.jpg', '.bmp')):
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(in_dir) if f.lower().endswith(suffixes)]

    for fname in tqdm(files, desc="Processing masks"):
        in_path = os.path.join(in_dir, fname)
        out_path = os.path.join(out_dir, fname)

        try:
            mask = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Can't read the file : {in_path}, jump it")
                continue
            circle_mask = reconstruct_disc_circle(mask)
            cv2.imwrite(out_path, circle_mask)
        except Exception as e:
            print(f"[Error] When processing {fname} encounter: {e}")


def process_image(av_img_path, disc_img_path):
    try:
        av_img = cv2.imread(av_img_path, cv2.IMREAD_COLOR)
        disc_img = cv2.imread(disc_img_path, cv2.IMREAD_GRAYSCALE)
        
        if av_img is None:
            print(f"Error: Unable to read A/V images: {av_img_path}")
            return None, None, None, None
        if disc_img is None:
            print(f"Error: Disc images cannot be read: {disc_img_path}")
            return None, None, None, None
        
        if av_img.shape[:2] != disc_img.shape:
            disc_img = cv2.resize(disc_img, (av_img.shape[1], av_img.shape[0]))
        
        _, disc_mask = cv2.threshold(disc_img, 200, 255, cv2.THRESH_BINARY)
        
        artery_img = np.zeros_like(av_img)
        vein_img = np.zeros_like(av_img)

        artery_mask = np.logical_or(
            np.all(av_img == [0, 0, 255], axis=-1),  
            np.all(av_img == [0, 255, 0], axis=-1)   
        )
        vein_mask = np.logical_or(
            np.all(av_img == [255, 0, 0], axis=-1),  
            np.all(av_img == [0, 255, 0], axis=-1)   
        )
        

        artery_img[artery_mask] = [0, 0, 255] 
        vein_img[vein_mask] = [255, 0, 0] 

        final_artery_img = cv2.bitwise_and(artery_img, artery_img, mask=disc_mask)
        final_vein_img = cv2.bitwise_and(vein_img, vein_img, mask=disc_mask)
        
        artery_gray = cv2.cvtColor(final_artery_img, cv2.COLOR_BGR2GRAY)
        _, artery_bin = cv2.threshold(artery_gray, 1, 255, cv2.THRESH_BINARY)
        
        num_labels_red, _, stats_red, _ = cv2.connectedComponentsWithStats(
            artery_bin, 8, cv2.CV_32S
        )
        red_arcs = []
        red_components = []
        for i in range(1, num_labels_red):  
            area = stats_red[i, cv2.CC_STAT_AREA]
            red_arcs.append(int(area))
            red_components.append(stats_red[i])
        
        vein_gray = cv2.cvtColor(final_vein_img, cv2.COLOR_BGR2GRAY)
        _, vein_bin = cv2.threshold(vein_gray, 1, 255, cv2.THRESH_BINARY)
        
        num_labels_blue, _, stats_blue, _ = cv2.connectedComponentsWithStats(
            vein_bin, 8, cv2.CV_32S
        )
        blue_arcs = []
        blue_components = []
        for i in range(1, num_labels_blue):  
            area = stats_blue[i, cv2.CC_STAT_AREA]
            blue_arcs.append(int(area))
            blue_components.append(stats_blue[i])
        
        red_arcs_sorted = sorted(red_arcs, reverse=True)[:4]
        blue_arcs_sorted = sorted(blue_arcs, reverse=True)[:4]
        
        red_top4_avg = np.mean(red_arcs_sorted) if red_arcs_sorted else 0
        blue_top4_avg = np.mean(blue_arcs_sorted) if blue_arcs_sorted else 0
        
        ratio = red_top4_avg / blue_top4_avg if blue_top4_avg > 0 else float('inf')
        
        visualization_a = np.zeros_like(av_img)
        visualization_v = np.zeros_like(av_img)
        
        visualization_a[np.where(final_artery_img[:, :, 2] > 0)] = [0, 0, 255]
        visualization_v[np.where(final_vein_img[:, :, 0] > 0)] = [255, 0, 0]
        
        for stat in red_components:
            x, y, w, h, area = stat
            cv2.rectangle(visualization_a, (x-5, y-5), (x + w + 5, y + h + 5), (0, 255, 255), 1)
            cv2.putText(visualization_a, f"A:{int(area)}", (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        for stat in blue_components:
            x, y, w, h, area = stat
            cv2.rectangle(visualization_v, (x-5, y-5), (x + w + 5, y + h + 5), (255, 255, 0), 1)
            cv2.putText(visualization_v, f"V:{int(area)}", (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.putText(visualization_a, f"Red Top4: {red_top4_avg:.1f} ({ratio:.2f})", 
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(visualization_v, f"Blue Top4: {blue_top4_avg:.1f}", 
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        disc_colored = cv2.cvtColor(disc_mask, cv2.COLOR_GRAY2BGR)
        av_img_resized = cv2.resize(av_img, (disc_colored.shape[1], disc_colored.shape[0]))
        top_row = np.hstack((av_img_resized, disc_colored))
        bottom_row = np.hstack((visualization_a, visualization_v))
        comparison = np.vstack((top_row, bottom_row))
        
        return red_arcs, blue_arcs, comparison, os.path.basename(av_img_path), red_top4_avg, blue_top4_avg, ratio
    
    except Exception as e:
        print(f"An error occurred when processing the picture: {av_img_path}")
        print(traceback.format_exc())
        return None, None, None, None, None, None, None

def process_images(av_dir, disc_dir, comparison_dir=None, log_file=None):
    supported_formats = (".png")
    av_files = [f for f in os.listdir(av_dir) if f.lower().endswith(supported_formats)]

    if comparison_dir and not os.path.exists(comparison_dir):
        os.makedirs(comparison_dir)
    
    all_results = []
    logs = []  
    ratios = [] 
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for filename in av_files:
            av_path = os.path.join(av_dir, filename)
            disc_path = os.path.join(disc_dir, filename)
            
            if os.path.exists(disc_path):
                futures.append(executor.submit(process_image, av_path, disc_path))
            else:
                log = f"Warning: {filename}  does not exist in the Disc directory"
                print(log)
                logs.append(log)
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is None:
                continue
                
            red_arcs, blue_arcs, comparison_img, filename, red_top4_avg, blue_top4_avg, ratio = result
            
            if red_arcs is not None and comparison_img is not None:
                if comparison_dir:
                    comp_path = os.path.join(comparison_dir, f"comparison_{filename}")
                    cv2.imwrite(comp_path, comparison_img)
                
                red_count = len(red_arcs)
                blue_count = len(blue_arcs)
                red_avg = sum(red_arcs) / red_count if red_count > 0 else 0
                blue_avg = sum(blue_arcs) / blue_count if blue_count > 0 else 0
                
                all_results.append({
                    "filename": filename,
                    "red_arcs": red_arcs,
                    "blue_arcs": blue_arcs,
                    "red_arc_count": red_count,
                    "blue_arc_count": blue_count,
                    "avg_red_arc": red_avg,
                    "avg_blue_arc": blue_avg,
                    "red_top4_avg": red_top4_avg,
                    "blue_top4_avg": blue_top4_avg,
                    "ratio": ratio
                })
                
                if math.isfinite(ratio) and ratio > 0:
                    ratios.append(ratio)
                
                log = [
                    f"\nProcessing completed: {filename}",
                    f"Arterial arc: {red_count} segment, length: {red_arcs}",
                    f"Average of the Top4 arteries: {red_top4_avg:.1f} pixles",
                    f"Venous arc: {blue_count} segment, length: {blue_arcs}",
                    f"Average of the Top4 veins: {blue_top4_avg:.1f} pixels",
                    f"AVR: {ratio:.4f}"
                ]
                
                # for line in log:
                #     print(line)
                logs.extend(log)
    
    return all_results, logs, ratios

def generate_report(results, ratios, report_path):
    all_red_arcs = [arc for res in results for arc in res["red_arcs"]]
    all_blue_arcs = [arc for res in results for arc in res["blue_arcs"]]
    
    ratio_avg = np.mean(ratios) if ratios else 0
    ratio_std = np.std(ratios) if ratios else 0
    
    report = "\n" + "=" * 60 + "\n"
    report += "Summary (Based on Top-4 Average Length)\n"
    report += "=" * 60 + "\n"
    report += "Filename".ljust(10) + "AVR\n"
    report += "-" * 60 + "\n"
    
    for res in results:
        filename = res["filename"]
        red_top4 = res["red_top4_avg"]
        blue_top4 = res["blue_top4_avg"]
        ratio = res["ratio"]
        
        if math.isfinite(ratio):
            report += f"{filename.ljust(10)}"
            report += f"{ratio:.4f}\n"
        else:
            report += f"{filename.ljust(65)}{red_top4:.1f}".ljust(15)
            report += f"{blue_top4:.1f}".ljust(15)
            report += "N/A (no venous arcs)\n"
    
    report += "\nOverall ratio statistics:\n"
    report += f"Average AVR: {ratio_avg:.4f}\n"
    report += f"Standard deviation of AVR: {ratio_std:.4f}\n"
    report += "=" * 60
    
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write(report)
    
    # print(report)
    
    return report


def AVR(av_dir, disc_dir, comparison_dir, report_path):
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("AVR Analysis Report (Based on Top-4 Average Length)\n")
        f.write("=" * 60 + "\n\n")
    
    results, logs, ratios = process_images(av_dir, disc_dir, comparison_dir, report_path)

    with open(report_path, 'a', encoding='utf-8') as f:
        for log in logs:
            f.write(log + "\n")

    if results:
        generate_report(results, ratios, report_path)

        all_red_top4 = [res["red_top4_avg"] for res in results]
        all_blue_top4 = [res["blue_top4_avg"] for res in results]

        # print("\nFinal Summary:")
        # print(f"Overall average of artery Top-4: {np.mean(all_red_top4):.1f}")
        # print(f"Overall average of vein Top-4: {np.mean(all_blue_top4):.1f}")
        # print(f"Overall AVR: {np.mean(ratios):.4f}")
        
        # Write final summary to file
        with open(report_path, 'a', encoding='utf-8') as f:
            f.write("\n\nFinal Summary:\n")
            f.write(f"Overall average of artery (red) Top-4: {np.mean(all_red_top4):.1f}\n")
            f.write(f"Overall average of vein Top-4: {np.mean(all_blue_top4):.1f}\n")
            f.write(f"Overall AVR: {np.mean(ratios):.4f}\n")
    else:
        print("No valid images found for processing.")
        with open(report_path, 'a', encoding='utf-8') as f:
            f.write("\nNo valid images found for processing.\n")
    
    # print(f"Report saved to: {report_path}")

def load_values(file_path):
    """
    retrun dict: {filename: value}
    """
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip() == '':
                continue
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            fname, value = parts
            try:
                data[fname] = float(value)
            except ValueError:
                print(f"[Warning] Skip invalid lines: {line.strip()}")
    return data

def eval_AVR(gt_dict, pred_dict):
    common_keys = set(gt_dict.keys()) & set(pred_dict.keys())
    if len(common_keys) == 0:
        raise ValueError("There are no common file names available for comparison!")

    abs_errors = []
    smape_values = []
    print("Img_ID, GT, Pred")
    for k in common_keys:
        
        y_true = gt_dict[k]
        y_pred = pred_dict[k]
        print(k, y_true, y_pred)
        
        abs_errors.append(abs(y_true - y_pred))
        denominator = (abs(y_true) + abs(y_pred)) / 2
        if denominator == 0:
            smape_values.append(0.0)
        else:
            smape_values.append(abs(y_true - y_pred) / denominator)

    mae = sum(abs_errors) / len(abs_errors)
    smape = 100 * sum(smape_values) / len(smape_values)

    return mae, smape



