import numpy as np 

def check_total_dict_items(dict): 
    total_item = 0
    for v in dict.values(): 
        total_item += len(v)
    
    return total_item    

def find_indices_in_list(lst, start_v, end_v):
    found_lst_start_idx = None 
    found_lst_end_idx = None 
    for j, span in enumerate(lst):
        if span[0] == start_v: 
            found_lst_start_idx = j 
        if span[1] == end_v: 
            found_lst_end_idx = j 
    return (found_lst_start_idx, found_lst_end_idx)

def f1_metrics(tp, fp, fn):
    # Calculate precision, recall, and F1 score
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if (precision + recall) != 0:
        f1_score = 2 * precision * recall / (precision + recall)
    else: 
        f1_score = -np.inf

    return f1_score

def create_bool_array(tuple_list):
    # Calculate the maximum end index
    max_end_idx = max(end_idx for start_idx, end_idx in tuple_list)

    # Create a boolean array of the appropriate size
    bool_array = [False] * (max_end_idx + 1)

    # Set the values in the boolean array to True for the specified indices
    for start_idx, end_idx in tuple_list:
        bool_array[start_idx] = True
        bool_array[end_idx] = True 
    return bool_array 

