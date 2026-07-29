import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from matplotlib.lines import Line2D
list_mae, list_mse, list_rmse, list_r2 = [], [], [], []
list_acc, list_prec, list_rec, list_f1 = [], [], [], []
prefix="original"
prop="bg"


#db="hetdb_stack"
db="samba_full_stack_layer"
#db="hetdb_samba_stack"
#db="samba_full_layer"
#db="samba_mono_layer"
#db="hetmono"
# 29 1/d + 1/d**6 # 26 1/d + 1/d #30 old SE 1/d+ 1/d**6 # 31 1/d only
#db="bidb_stack"
#db="hetdb_stack"
#db="hetdb"
# 28 1/d**6 # 26-27 1/d #30 old SE 1/d+ 1/d**6 # 31 1/d only
#db="bidb"

for fold in range(0, 4):
    pretrain="pre-trained" #outp
    prop="bg"
    #df = pd.read_csv(f"/home/anv/AIRS/OpenMat/PotNet/pre-trained/potnet_target_prediction_results_test_set_bidb_potentials_vdw_cl_lj_cutoff2_{fold}.csv",header=0)
    #df = pd.read_csv(f"/home/anv/AIRS/OpenMat/PotNet/{pretrain}/potnet_target_prediction_results_test_set_bidb_old_potentials_cl_cl_distance_new82_{fold}.csv",header=0)
    #df = pd.read_csv(f"/home/anv/AIRS/OpenMat/PotNet/{pretrain}/potnet_target_prediction_results_test_set_hetdb_old_potentials_cl_cl_distance_new106_{fold}.csv",header=0)
    #df = pd.read_csv(f"/home/anv/AIRS/OpenMat/PotNet/{pretrain}/potnet_target_prediction_results_test_set_{db}_gt_{fold}_73.csv",header=0)
    import pandas as pd

    
    df = pd.read_csv(f"/home/anv/BDIPNet/results/{db}_{fold}_test_0.csv",header=0)
    #df= pd.read_csv(f"/home/anv/AIRS/OpenMat/PotNet/results/{db}_{fold}_test_1.csv",header=0)
    df["mae"] = (df["target"] - df["prediction"]).abs()
    pair_part = df["id"].astype(str).str.strip().str.split("_").str[0]
    mono1 = pair_part.str.split("+").str[0]
    mono2 = pair_part.str.split("+").str[1]

    #df = df[mono1 != mono2].copy()
    y_true = df["target"].values  
    y_pred = df["prediction"].values 
    df = df.sort_values("id", ascending=True).copy()
    #print(df)
    print(len(df))
   
    
    if len(df) == 0:
        print(f"❌ Fold {fold}: empty after dropna → skip")
        continue
    if prop=="bg":
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        print(mae, mse, rmse, r2)
        list_mae.append(mae)
        list_mse.append(mse)
        list_rmse.append(rmse)
        list_r2.append(r2)

    else:
        y_true = df[0].values 
        logits = df.iloc[:, 1:].values  

        y_pred = np.argmax(logits, axis=1)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        list_acc.append(acc)
        list_prec.append(prec)
        list_rec.append(rec)
        list_f1.append(f1)

if prop=="bg":
    print(f"MAE  : {np.mean(list_mae):.4f} ± {np.std(list_mae):.4f}")
    print(f"MSE  : {np.mean(list_mse):.4f} ± {np.std(list_mse):.4f}")
    print(f"RMSE : {np.mean(list_rmse):.4f} ± {np.std(list_rmse):.4f}")
    print(f"R2   : {np.mean(list_r2):.4f} ± {np.std(list_r2):.4f}")
else:
    print(f"Accuracy : {np.mean(list_acc):.4f} ± {np.std(list_acc):.4f}")
    print(f"Precision: {np.mean(list_prec):.4f} ± {np.std(list_prec):.4f}")
    print(f"Recall   : {np.mean(list_rec):.4f} ± {np.std(list_rec):.4f}")
    print(f"F1-score : {np.mean(list_f1):.4f} ± {np.std(list_f1):.4f}")

