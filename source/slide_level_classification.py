import pandas as pd
import numpy as np 
from sklearn import metrics
from sklearn.metrics import classification_report, average_precision_score, confusion_matrix
import argparse

# Make ground truth data frame (per slide (patient))
# Output : slide_df (ground_truth_df), ebv_pos_id_list, ebv_neg_id_list
# slide_df : patient_id, is_EBV_pos 
def make_slide_df(patch_result):
    slide_df = patch_result.drop_duplicates("patient_id", keep="first")
    slide_df = slide_df[["patient_id", "is_EBV_pos"]]
    sample_id_list = slide_df["patient_id"].tolist()
    return slide_df, sample_id_list


def ebv_pos_inference(slide_df, sample_id, tumor_threshold, ebv_threshold):
    
    # Filter patch from patient with sample_id
    sample_df = slide_df[slide_df["patient_id"] == sample_id] 

    # Filter patches predicted as tumor (predicted probability > tumor_threshold)
    tumor = sample_df[sample_df["tumor_prediction"] >= tumor_threshold]
    num_tumor = len(tumor) # Number of patches predicted as tumor 

    # Among tumor patches, filter pacthes predicted as ebv_positive (predicted probability > ebv_threshold)
    ebv_pos = tumor[tumor["ebv_prediction"] >= ebv_threshold]
    num_ebv_pos = len(ebv_pos) # Number of patches predicted as ebv_positive among patches predicted as tumor 

    ebv_pos_ratio = num_ebv_pos / num_tumor # Ratio of ebv_positive patches / tumor patches 

    return num_tumor, num_ebv_pos, ebv_pos_ratio 


def slide_level_inference(patch_result, tumor_threshold, ebv_threshold):

    slide_df, sample_id_list = make_slide_df(patch_result) 

    num_tumor_list = [] # List of tumor patches per patient
    num_ebv_pos_list = [] # List of EBV positive patches per patient 
    ebv_pos_ratio_list = [] # List of (EBV Positive patch / Tumor patch) ratio per patient

    for id_ in sample_id_list:

        num_tumor, num_ebv_pos, ebv_pos_ratio = ebv_pos_inference(patch_result, id_, tumor_threshold, ebv_threshold)
        num_tumor_list.append(num_tumor)
        num_ebv_pos_list.append(num_ebv_pos)
        ebv_pos_ratio_list.append(ebv_pos_ratio)

    # Add column to gt (ground truth dataframe)
    slide_df["num_tumor"] = num_tumor_list
    slide_df["num_ebv_pos"] = num_ebv_pos_list
    slide_df["ebv_pos_ratio"] = ebv_pos_ratio_list

    return slide_df 


def evaluate(patch_result, tumor_threshold, ebv_threshold, ebv_pos_ratio_threshold):

    gt_pred_df = slide_level_inference(patch_result, tumor_threshold, ebv_threshold)
    gt_pred_df["ebv_pos_pred"] = gt_pred_df["ebv_pos_ratio"] > ebv_pos_ratio_threshold # Patient predicted as ebv positive

    print("============== Evaluation Metrics ==============")
    print("Tumor classifier threshold : {} / EBV classifier threshold : {} / EBV_pos_ratio_threshold : {}".format(tumor_threshold, ebv_threshold, ebv_pos_ratio_threshold))

    # Classification report
    print("< Classification Report >")
    print(classification_report(gt_pred_df["is_EBV_pos"], gt_pred_df["ebv_pos_pred"]))

    # Specificity, NPV (negative predictive value)
    cm = confusion_matrix(gt_pred_df["is_EBV_pos"], gt_pred_df["ebv_pos_pred"])
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    specificity = tn / (tn + fp)
    npv = tn / (tn + fn)
    print("Specificity : {}".format(np.round(specificity, 3)))
    print("NPV : {}".format(np.round(npv, 3)))

    # AUROC 
    auroc = np.round(metrics.roc_auc_score(gt_pred_df["is_EBV_pos"], gt_pred_df["ebv_pos_ratio"]), 2)
    print("AUROC : {}".format(auroc))

    # AUPRC
    auprc = np.round(average_precision_score(gt_pred_df["is_EBV_pos"], gt_pred_df["ebv_pos_ratio"]),2)
    print("AUPRC : {}".format(auprc))


def main():
    patch_result = pd.read_csv(PATCH_RESULT, engine="python")
    evaluate(patch_result, tumor_threshold=TUMOR_THRESHOLD, ebv_threshold=EBV_THRESHOLD, ebv_pos_ratio_threshold=EBV_POS_RATIO_THRESHOLD)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="EBVNet Slide-level Inference")
    parser.add_argument('--patch-result', type=str)
    parser.add_argument('--tumor-threshold', type=float)
    parser.add_argument('--ebv-threshold', type=float)
    parser.add_argument('--ebv-pos-ratio-threshold', type=float)

    args = parser.parse_args()

    PATCH_RESULT = args.patch_result
    TUMOR_THRESHOLD = args.tumor_threshold
    EBV_THRESHOLD = args.ebv_threshold
    EBV_POS_RATIO_THRESHOLD = args.ebv_pos_ratio_threshold

    main()