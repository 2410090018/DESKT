import os
import argparse
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import KFold
from datetime import datetime
from data_loaders import MostRecentQuestionSkillDataset, CounterDatasetWrapper
from model import DESKT
from train import model_train


def parse_args():
    parser = argparse.ArgumentParser()
    # 默认路径
    parser.add_argument("--data_dir", type=str, default="dataset/assistments17", help="folder that contains preprocessed_df.csv")
    parser.add_argument("--csv_name", type=str, default="preprocessed_df.csv")
    parser.add_argument("--model_out", type=str, default="saved_model")
    
    # 训练超参数
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--early_stop", type=int, default=10) 
    
    # 模型架构参数
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--embedding_size", type=int, default=64)
    parser.add_argument("--memory_size", type=int, default=64, help="Size of the memory matrix (Mk/Mv)")
    parser.add_argument("--dropout", type=float, default=0.05)
    
    # 系统参数
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_folds", type=int, default=5)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置随机种子，保证可重复性
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    csv_path = os.path.join(args.data_dir, args.csv_name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, sep="\t")
    print(f"Loaded CSV from {csv_path}. Shape: {df.shape}")


    required = ["user_id", "item_id", "timestamp", "correct", "skill_id",
                "Average_confidence(CONFUSED)",
                "Average_confidence(CONCENTRATING)", "Average_confidence(BORED)"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Required column missing: {c}. Please run preprocess.py first.")

    # 2. ID 处理 (Padding 0 预留)
    df = df.copy()
    df["skill_id"] = df["skill_id"].astype(int) + 1
    df["item_id"] = df["item_id"].astype(int) + 1
    
    num_skills = int(df["skill_id"].max()) + 1

    users = df["user_id"].unique()
    np.random.shuffle(users)

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    all_aucs, all_accs, all_rmses = [], [], []

    # [新增] 记录 5 折中表现最好的全局 AUC，以便保存最优秀的权重用于画图
    best_overall_auc = 0.0

    fold_idx = 0
    for train_idx, test_idx in kf.split(users):
        fold_idx += 1
        print(f"\n{'='*20} Fold {fold_idx} / {args.n_folds} {'='*20}")
        
        train_users = users[train_idx]
        test_users = users[test_idx]

        np.random.shuffle(train_users)
        cutoff = int(0.9 * len(train_users))
        train_u = train_users[:cutoff]
        valid_u = train_users[cutoff:]

        train_df = df[df["user_id"].isin(train_u)]
        valid_df = df[df["user_id"].isin(valid_u)]
        test_df = df[df["user_id"].isin(test_users)]

        print(f"Train users: {len(train_u)}, Valid users: {len(valid_u)}, Test users: {len(test_users)}")

        # Dataset & DataLoader
        train_ds = MostRecentQuestionSkillDataset(train_df, args.seq_len)
        valid_ds = MostRecentQuestionSkillDataset(valid_df, args.seq_len)
        test_ds = MostRecentQuestionSkillDataset(test_df, args.seq_len)

        train_loader = torch.utils.data.DataLoader(
            CounterDatasetWrapper(train_ds, args.seq_len),
            batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0
        )
        valid_loader = torch.utils.data.DataLoader(
            CounterDatasetWrapper(valid_ds, args.seq_len),
            batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0
        )
        test_loader = torch.utils.data.DataLoader(
            CounterDatasetWrapper(test_ds, args.seq_len),
            batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0
        )

        device = torch.device(args.device)

  
        model = DESKT(
            num_skills=num_skills,
            dim_s=args.embedding_size,
            size_m=args.memory_size,
            dropout=args.dropout,
            num_emotions=3
        ).to(device)
        
        if fold_idx == 1:
            print(f"DESKT model initialized with Difficulty-Enhanced Emotion Slip Logic.")

        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

        # 训练 (返回的是该 Fold 在测试集上的表现)
        auc, acc, rmse = model_train(
            fold_idx,
            model,
            opt,
            train_loader,
            valid_loader,
            test_loader,
            device,
            num_epochs=args.num_epochs,
            model_dir=args.model_out,
            early_stop_patience=args.early_stop
        )

        # =========================================================
        # [修改处] 保存最佳模型权重用于后续可视化分析
        # =========================================================
    print(f"\n>>> 检查 Fold {fold_idx} 模型表现 <<<")
    if auc > best_overall_auc:
            best_overall_auc = auc
            # 当遇到更高的 AUC 时，保存模型权重
            torch.save(model.state_dict(), "best_emotion_model.pth")
            print(f"🌟 发现更好的模型！全局最优 AUC 刷新为 {best_overall_auc:.4f}")
            print("💾 模型权重已安全保存到 best_emotion_model.pth (供 visualize.py 使用)")
    else:
            print(f"当前 Fold AUC ({auc:.4f}) 未超过历史最佳 ({best_overall_auc:.4f})，跳过保存。")
            
    print("=========================================================\n")
    all_aucs.append(auc)
    all_accs.append(acc)
    all_rmses.append(rmse)

    print("\n=== 5-fold Summary ===")
    print(f"AUC: {np.mean(all_aucs):.5f} ± {np.std(all_aucs):.5f}")
    print(f"ACC: {np.mean(all_accs):.5f} ± {np.std(all_accs):.5f}")
    print(f"RMSE: {np.mean(all_rmses):.5f} ± {np.std(all_rmses):.5f}")


if __name__ == "__main__":
    main()
