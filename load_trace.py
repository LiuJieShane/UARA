import os
import numpy as np

COOKED_TRACE_FOLDER = './user_traces/'


def load_trace(cooked_trace_folder=COOKED_TRACE_FOLDER):
    cooked_files = os.listdir(cooked_trace_folder)
    all_user_pos = []
    all_file_names = []
    for cooked_file in cooked_files:
        file_path = cooked_trace_folder + cooked_file
        user_pos = np.loadtxt(file_path)
        all_user_pos.append(user_pos)
        all_file_names.append(cooked_file)

    return all_user_pos, all_file_names