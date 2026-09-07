
import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error

def model_train(
    fold_idx,
    model,
    optimizer,
    train_loader,
    valid_loader,
    test_loader,
    device,
    num_epochs=50,
    model_dir="saved_model",
    early_stop_patience=10,
):
    os.makedirs(model_dir, exist_ok=True)
    best_valid_auc = -1.0
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        # ======================================================
        # 1. Training
        # ======================================================
        model.train()
        epoch_losses = []
        train_preds_list = []
        train_trues_list = []

        for batch in tqdm(train_loader, desc=f"Fold {fold_idx} Epoch {epoch}"):
            skills = batch["skills"].to(device).long()
            responses = batch["responses"].to(device).float()
            emotions = batch["emotions"].to(device).float() # 这里现在是 (B, L, 3)
            difficulty = batch["difficulty"].to(device).float()
            
      

            counter_mask, mask = batch["attention_mask"]
            counter_mask = counter_mask.to(device).long()
            mask = mask.to(device).long()

            feed = {
                "skills": skills,
                "responses": responses,
                "emotions": emotions,
                "difficulty": difficulty,
                "attention_mask": (counter_mask, mask)
            }

            out = model(feed)
     
            loss, _, _ = model.loss(feed, out)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_losses.append(loss.item())

            preds = out["pred"].detach().cpu()
            trues = out["true"].detach().cpu()
            m = trues > -1

            if m.sum() > 0:
                train_preds_list.append(preds[m])
                train_trues_list.append(trues[m])

        # 计算 Train AUC
        if len(train_preds_list) > 0:
            train_preds_np = torch.cat(train_preds_list).numpy().ravel()
            train_trues_np = torch.cat(train_trues_list).numpy().ravel()
            try:
                train_auc = roc_auc_score(train_trues_np, train_preds_np)
            except:
                train_auc = 0.0
        else:
            train_auc = 0.0
        train_loss = np.mean(epoch_losses)

        # ======================================================
        # 2. Validation
        # ======================================================
        model.eval()
        valid_preds = []
        valid_trues = []

        with torch.no_grad():
            for batch in valid_loader:
                skills = batch["skills"].to(device).long()
                responses = batch["responses"].to(device).float()
                emotions = batch["emotions"].to(device).float()
                difficulty = batch["difficulty"].to(device).float()

                feed = {
                    "skills": skills,
                    "responses": responses,
                    "emotions": emotions,
                    "difficulty": difficulty
                }

                out = model(feed)
                preds = out["pred"].detach().cpu()
                trues = out["true"].detach().cpu()
                m = trues > -1

                if m.sum() > 0:
                    valid_preds.append(preds[m])
                    valid_trues.append(trues[m])

        if len(valid_preds) > 0:
            valid_preds_np = torch.cat(valid_preds).numpy().ravel()
            valid_trues_np = torch.cat(valid_trues).numpy().ravel()
            try:
                valid_auc = roc_auc_score(valid_trues_np, valid_preds_np)
            except:
                valid_auc = 0.0
        else:
            valid_auc = 0.0

        print(f"Fold {fold_idx} Epoch {epoch} | Loss {train_loss:.6f} | Train AUC {train_auc:.4f} | Valid AUC {valid_auc:.4f}")

        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_epoch = epoch
            torch.save({"epoch": epoch, "state_dict": model.state_dict()},
                        os.path.join(model_dir, f"best_model_fold{fold_idx}.pt"))

        if epoch - best_epoch >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}. Best Valid AUC: {best_valid_auc:.4f}")
            break

    # ======================================================
    # 3. Test
    # ======================================================
    ckpt_path = os.path.join(model_dir, f"best_model_fold{fold_idx}.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])

    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for batch in test_loader:
         
            skills = batch["skills"].to(device).long()
            responses = batch["responses"].to(device).float()
            emotions = batch["emotions"].to(device).float()
            difficulty = batch["difficulty"].to(device).float()

            feed = {"skills": skills, "responses": responses, "emotions": emotions, "difficulty": difficulty}
            out = model(feed)
            preds = out["pred"].cpu()
            trues = out["true"].cpu()
            m = trues > -1
            if m.sum() > 0:
                all_preds.append(preds[m])
                all_trues.append(trues[m])

    if len(all_preds) > 0:
        all_preds = torch.cat(all_preds).numpy()
        all_trues = torch.cat(all_trues).numpy()
        test_auc = roc_auc_score(all_trues, all_preds)
        test_acc = accuracy_score(all_trues >= 0.5, all_preds >= 0.5)
        test_rmse = np.sqrt(mean_squared_error(all_trues, all_preds))
    else:
        test_auc = test_acc = test_rmse = 0.0

    print(f"Fold {fold_idx} Final Test -> AUC: {test_auc:.4f}, ACC: {test_acc:.4f}, RMSE: {test_rmse:.4f}")
    return test_auc, test_acc, test_rmse