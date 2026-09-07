
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ==========================================
# Helper Padding Functions
# ==========================================
def pad_sequence(seq, max_len, pad_value=0):
    seq = list(seq)
    L = len(seq)
    if L >= max_len:
        return torch.tensor(seq[-max_len:], dtype=torch.long)
    padded = [pad_value] * (max_len - L) + seq
    return torch.tensor(padded, dtype=torch.long)

def pad_sequence_left(seq, max_len, pad_value=0):
    seq = list(seq)
    L = len(seq)
    if L >= max_len:
        return torch.tensor(seq[:max_len], dtype=torch.long)
    padded = seq + [pad_value] * (max_len - L)
    return torch.tensor(padded, dtype=torch.long)

def pad_float_sequence(seq, max_len, pad_value=0.0):
    seq = list(seq)
    L = len(seq)
    if L >= max_len:
        return torch.tensor(seq[-max_len:], dtype=torch.float)
    padded = [pad_value] * (max_len - L) + seq
    return torch.tensor(padded, dtype=torch.float)

def pad_matrix(seq_mat, max_len, feat_dim, pad_value=0.0):
    seq_mat = list(seq_mat)
    L = len(seq_mat)
    if L >= max_len:
        return torch.tensor(seq_mat[-max_len:], dtype=torch.float)
    pad_rows = [[pad_value] * feat_dim for _ in range(max_len - L)]
    padded = pad_rows + seq_mat
    return torch.tensor(padded, dtype=torch.float)

def pad_matrix_left(seq_mat, max_len, feat_dim, pad_value=0.0):
    seq_mat = list(seq_mat)
    L = len(seq_mat)
    if L >= max_len:
        return torch.tensor(seq_mat[:max_len], dtype=torch.float)
    pad_rows = [[pad_value] * feat_dim for _ in range(max_len - L)]
    padded = seq_mat + pad_rows
    return torch.tensor(padded, dtype=torch.float)

# ==========================================
# Dataset Class 1: Most Recent (Right Aligned)
# ==========================================
class MostRecentQuestionSkillDataset(Dataset):
    """
    Right-aligned (most recent N interactions). 
    3 Emotions: CONFUSED, CONCENTRATING, BORED
    """
    def __init__(self, df: pd.DataFrame, seq_len: int, num_skills=None, num_questions=None):
        super().__init__()
        self.seq_len = seq_len

        # 1. Check required columns
        if "user_id" not in df.columns:
            raise KeyError("df must contain 'user_id' column")
        required = ["item_id", "correct", "skill_id"]
        for c in required:
            if c not in df.columns:
                raise KeyError(f"df must contain '{c}' column")

        if "problem_type_id" in df.columns:
            self.has_problem_type = True
        else:
            self.has_problem_type = False
 
        emo_cols = [
            "Average_confidence(CONFUSED)",
            "Average_confidence(CONCENTRATING)",
            "Average_confidence(BORED)",
        ]
        self.has_emotions = all(c in df.columns for c in emo_cols)

        # Difficulty check
        if "difficulty" in df.columns:
            self.has_difficulty = True
            item_difficulty = df[["item_id", "difficulty"]].drop_duplicates().set_index("item_id")["difficulty"].to_dict()
        else:
            self.has_difficulty = False
            acc = df.groupby("item_id")["correct"].mean()
            difficulty_series = (1.0 - acc).fillna(0.5)
            item_difficulty = difficulty_series.to_dict()

        # 2. Group by User
        self.questions = []
        self.skills = []
        self.responses = []
        self.emotions = []
        self.difficulties = []
        self.problem_types = []

        for uid, user_df in df.groupby("user_id"):
            q = user_df["item_id"].tolist()
            s = user_df["skill_id"].tolist()
            r = user_df["correct"].tolist()
             
            if self.has_emotions:
                co = user_df[emo_cols[0]].tolist()
                ct = user_df[emo_cols[1]].tolist()
                bo = user_df[emo_cols[2]].tolist()
                emo_mat = np.stack([co, ct, bo], axis=1) # (L, 3)
            else:
                emo_mat = np.zeros((len(q), 3), dtype=float)

            diff = [float(item_difficulty.get(int(it), 0.5)) for it in q]

            if self.has_problem_type:
                pt = user_df["problem_type_id"].tolist()
            else:
                pt = [0] * len(q)

            self.questions.append(q)
            self.skills.append(s)
            self.responses.append(r)
            self.emotions.append(emo_mat)
            self.difficulties.append(diff)
            self.problem_types.append(pt)

        # 3. Pad Sequences
        N = len(self.questions)
        self.padded_skills = torch.zeros((N, seq_len), dtype=torch.long)
        self.padded_responses = torch.full((N, seq_len), -1.0, dtype=torch.float)
 
        self.padded_emotions = torch.zeros((N, seq_len, 3), dtype=torch.float)
        self.padded_difficulty = torch.zeros((N, seq_len, 1), dtype=torch.float)
        self.padded_problem_types = torch.zeros((N, seq_len), dtype=torch.long)

        for i in range(N):
            s = self.skills[i]
            r = self.responses[i]
            e = self.emotions[i]
            d = self.difficulties[i]
            pt = self.problem_types[i]
            
            L = len(s)
            
            self.padded_skills[i] = pad_sequence(s, seq_len)
            
            rr = [-1.0] * (seq_len - L) + r[-seq_len:]
            self.padded_responses[i] = torch.tensor(rr, dtype=torch.float)
            
       
            self.padded_emotions[i] = pad_matrix(e, seq_len, 3)
            
            dd = [-1.0] * (seq_len - L) + d[-seq_len:]
            self.padded_difficulty[i, :, 0] = torch.tensor(dd, dtype=torch.float)
            
            self.padded_problem_types[i] = pad_sequence(pt, seq_len, pad_value=0)

    def __len__(self):
        return self.padded_skills.size(0)

    def __getitem__(self, idx):
        skills = self.padded_skills[idx]
        responses = self.padded_responses[idx]
        emotions = self.padded_emotions[idx]
        difficulty = self.padded_difficulty[idx]
        p_types = self.padded_problem_types[idx]

        mask = (skills != 0).long()
        counter_mask = mask.clone()

        return {
            "skills": skills,
            "responses": responses,
            "emotions": emotions,
            "difficulty": difficulty.squeeze(-1),
            "problem_type_id": p_types,
            "attention_mask": (counter_mask, mask),
        }

# ==========================================
# Dataset Class 2: Most Early (Left Aligned)
# ==========================================
class MostEarlyQuestionSkillDataset(Dataset):
    """
    Left-aligned (earliest N interactions).
    3 Emotions: CONFUSED, CONCENTRATING, BORED
    """
    def __init__(self, df: pd.DataFrame, seq_len: int, num_skills=None, num_questions=None):
        super().__init__()
        self.seq_len = seq_len

    
        emo_cols = [
            "Average_confidence(CONFUSED)",
            "Average_confidence(CONCENTRATING)",
            "Average_confidence(BORED)",
        ]
        self.has_emotions = all(c in df.columns for c in emo_cols)

        if "problem_type_id" in df.columns:
            self.has_problem_type = True
        else:
            self.has_problem_type = False

        if "difficulty" in df.columns:
            self.has_difficulty = True
            item_difficulty = df[["item_id", "difficulty"]].drop_duplicates().set_index("item_id")["difficulty"].to_dict()
        else:
            self.has_difficulty = False
            acc = df.groupby("item_id")["correct"].mean()
            difficulty_series = (1.0 - acc).fillna(0.5)
            item_difficulty = difficulty_series.to_dict()

        self.questions = []
        self.skills = []
        self.responses = []
        self.emotions = []
        self.difficulties = []
        self.problem_types = []

        for uid, user_df in df.groupby("user_id"):
            q = user_df["item_id"].tolist()
            s = user_df["skill_id"].tolist()
            r = user_df["correct"].tolist()
          
            if self.has_emotions:
                co = user_df[emo_cols[0]].tolist()
                ct = user_df[emo_cols[1]].tolist()
                bo = user_df[emo_cols[2]].tolist()
                emo_mat = np.stack([co, ct, bo], axis=1)
            else:
                emo_mat = np.zeros((len(q), 3), dtype=float)

            diff = [float(item_difficulty.get(int(it), 0.5)) for it in q]
            
            if self.has_problem_type:
                pt = user_df["problem_type_id"].tolist()
            else:
                pt = [0] * len(q)

            self.questions.append(q)
            self.skills.append(s)
            self.responses.append(r)
            self.emotions.append(emo_mat)
            self.difficulties.append(diff)
            self.problem_types.append(pt)

        N = len(self.questions)
        self.padded_skills = torch.zeros((N, seq_len), dtype=torch.long)
        self.padded_responses = torch.full((N, seq_len), -1.0, dtype=torch.float)
  
        self.padded_emotions = torch.zeros((N, seq_len, 3), dtype=torch.float)
        self.padded_difficulty = torch.zeros((N, seq_len, 1), dtype=torch.float)
        self.padded_problem_types = torch.zeros((N, seq_len), dtype=torch.long)

        for i in range(N):
            q = self.questions[i]
            s = self.skills[i]
            r = self.responses[i]
            e = self.emotions[i]
            d = self.difficulties[i]
            pt = self.problem_types[i]
            
            L = min(len(q), seq_len)

            self.padded_skills[i, :L] = torch.tensor(s[:L], dtype=torch.long)
            
            rr = r[:L] + [-1.0] * (seq_len - L)
            self.padded_responses[i] = torch.tensor(rr, dtype=torch.float)
       
            self.padded_emotions[i] = pad_matrix_left(e, seq_len, 3)
            
            dd = d[:L] + [-1.0] * (seq_len - L)
            self.padded_difficulty[i, :, 0] = torch.tensor(dd, dtype=torch.float)

            self.padded_problem_types[i, :L] = torch.tensor(pt[:L], dtype=torch.long)

    def __len__(self):
        return self.padded_skills.size(0)

    def __getitem__(self, idx):
        skills = self.padded_skills[idx]
        responses = self.padded_responses[idx]
        emotions = self.padded_emotions[idx]
        difficulty = self.padded_difficulty[idx]
        p_types = self.padded_problem_types[idx]

        mask = (skills != 0).long()
        counter_mask = mask.clone()

        return {
            "skills": skills,
            "responses": responses,
            "emotions": emotions,
            "difficulty": difficulty.squeeze(-1),
            "problem_type_id": p_types,
            "attention_mask": (counter_mask, mask),
        }

# ==========================================
# Wrapper (Passthrough)
# ==========================================
class CounterDatasetWrapper(Dataset):
    def __init__(self, base_dataset, seq_len):
        self.base = base_dataset
        self.seq_len = seq_len

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return self.base[idx]