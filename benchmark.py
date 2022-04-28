import os, sys
from bisect import bisect_left

ROOT_DIR = os.path.dirname(__file__)
sys.path.append(ROOT_DIR)

from similarity import CalcEMD, sift_sim, pixel_sim, structural_sim


DATA_FOLDER = os.path.join(ROOT_DIR, "Data", "dataset")
GND_FOLDER = os.path.join(ROOT_DIR, "Data", "ground_truth_data")
QUERY_PATH = os.path.join(ROOT_DIR, "Data", "29020.jpg")
OUTPUT_FOLDER = os.path.join(ROOT_DIR, "BenchmarkResult")

metric_names = ["Earth Mover's Distance", "SIFT Similarity", "Pixel Similarity", "Structural Similarity"]
metric_funcs = [CalcEMD, sift_sim, pixel_sim, structural_sim]

def benchmarkMetric(metric_func, metric_name):
    # Construct image name list
    file_names = []
    img_names = os.listdir(DATA_FOLDER)
    for img_name in img_names:
        file_names.append(os.path.join(DATA_FOLDER, img_name))

    # Construct ground name list
    gnd_names = os.listdir(GND_FOLDER)

    sim_list = []
    name_list = []
    for i, file_name in enumerate(file_names):
        # Calculate the similarity for the current image
        sim = metric_func(file_name, QUERY_PATH)
        
        if len(sim_list) == 0:
            sim_list.append(sim)
            name_list.append(img_names[i])
        else:
            insert_i = bisect_left(sim_list, sim)
            sim_list.insert(insert_i, sim)
            name_list.insert(insert_i, img_names[i])

    # Reverse the list if needed
    if metric_name != "Earth Mover's Distance":
        name_list.reverse()
        sim_list.reverse()

    pos = 0
    for i in range(0, len(gnd_names)):
        if name_list[i] in gnd_names:
            pos += 1

    acc = pos/len(gnd_names)
    return pos, acc

import time

if __name__ == "__main__":
    output_file_name = "{}.txt".format(round(time.time(), 4))
    output_f = open(os.path.join(OUTPUT_FOLDER, output_file_name), "w+")

    for i, metric_func in enumerate(metric_funcs):
        time_start = time.time()
        pos, acc = benchmarkMetric(metric_func, metric_names[i])
        time_end = time.time()
        time_elasped = round(time_end-time_start, 2)
        output_msg = "Benchmark Metric: {}\n\tNumber of positive images in top 10 images: {}\n\tAccuracy: {} \n\tTime elasped: {} seconds\n".format(metric_names[i], pos, acc, time_elasped)
        print(output_msg)
        output_f.write(output_msg)

    output_f.close()
