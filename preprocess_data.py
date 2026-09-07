from argparse import ArgumentParser
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from scipy import sparse

import os
import pickle
import glob

from sklearn.preprocessing import MinMaxScaler


# ============================================================
# 基础配置
# ============================================================

BASE_PATH = "./dataset"

np.random.seed(12405)


# ============================================================
# 情绪列统一定义
# ============================================================

EMOTION_COLUMNS = [
    "Average_confidence(FRUSTRATED)",
    "Average_confidence(CONFUSED)",
    "Average_confidence(CONCENTRATING)",
    "Average_confidence(BORED)"
]


# ============================================================
# 1. ASSISTments17 文件合并
# ============================================================

def merge_assistments17_files(data_path):
    """
    ASSISTments2017:
    如果 assistments17.csv 不存在，
    自动合并 student_log_*.csv
    """

    merged_csv_path = os.path.join(
        data_path,
        "assistments17.csv"
    )

    # --------------------------------------------------------
    # 如果已经存在合并后的文件
    # --------------------------------------------------------

    if os.path.exists(merged_csv_path):

        print("=" * 60)
        print("ASSISTments17 merged file already exists.")
        print(f"Using: {merged_csv_path}")
        print("=" * 60)

        return merged_csv_path

    # --------------------------------------------------------
    # 查找 student_log 文件
    # --------------------------------------------------------

    print("=" * 60)
    print("Merged CSV not found.")
    print("Starting ASSISTments17 file merge process...")
    print("=" * 60)

    csv_files = glob.glob(
        os.path.join(
            data_path,
            "student_log_*.csv"
        )
    )

    if not csv_files:

        raise FileNotFoundError(
            f"\nNo 'student_log_*.csv' files found in:\n"
            f"{data_path}\n"
        )

    print(f"\nFound {len(csv_files)} student log files.")

    df_list = []

    # --------------------------------------------------------
    # 逐个读取
    # --------------------------------------------------------

    for file in sorted(csv_files):

        print(f"Reading: {file}")

        df_part = pd.read_csv(
            file,
            encoding="ISO-8859-1",
            low_memory=False
        )

        df_list.append(df_part)

    # --------------------------------------------------------
    # 合并
    # --------------------------------------------------------

    print("\nConcatenating all DataFrames...")

    df = pd.concat(
        df_list,
        ignore_index=True
    )

    print(
        f"Merged Data Shape: {df.shape}"
    )

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    print(
        f"\nSaving merged data to:\n"
        f"{merged_csv_path}"
    )

    df.to_csv(
        merged_csv_path,
        index=False
    )

    print("\nASSISTments17 merge complete.")

    return merged_csv_path


# ============================================================
# 2. 读取 ASSISTments12
# ============================================================

def load_assistments12(data_path):

    file_path = os.path.join(
        data_path,
        "assistments12.csv"
    )

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"\nASSISTments12 file not found:\n"
            f"{file_path}\n"
        )

    print("=" * 60)
    print("Loading ASSISTments12")
    print(f"File: {file_path}")
    print("=" * 60)

    df = pd.read_csv(
        file_path,
        encoding="ISO-8859-1",
        low_memory=False
    )

    return df


# ============================================================
# 3. 读取 ASSISTments17
# ============================================================

def load_assistments17(data_path):

    merged_csv_path = merge_assistments17_files(
        data_path
    )

    print("=" * 60)
    print("Loading ASSISTments17")
    print(f"File: {merged_csv_path}")
    print("=" * 60)

    df = pd.read_csv(
        merged_csv_path,
        encoding="ISO-8859-1",
        low_memory=False
    )

    return df


# ============================================================
# 4. ASSISTments12 Problem Type 分类
# ============================================================

def group_problem_type_12(ptype):

    s = str(ptype).strip().lower()

    choose_group = [
        "choose_1",
        "choose_n"
    ]

    if s in choose_group:

        return 1

    else:

        return 0


# ============================================================
# 5. ASSISTments17 Problem Type 分类
# ============================================================

def group_problem_type_17(ptype):

    s = str(ptype).strip().lower()

    # ASSISTments2017 选择题
    if s == "radioquestion":

        return 1

    else:

        return 0


# ============================================================
# 6. 数据集列名统一
# ============================================================

def normalize_dataset_columns(
    df,
    dataset_type
):
    """
    将 ASSISTments12 和 ASSISTments17
    统一为内部标准字段：

    user_id
    item_id
    skill_id
    attempt_count
    problem_type
    start_time
    end_time
    correct
    """

    # ========================================================
    # ASSISTments12
    # ========================================================

    if dataset_type == "assistments12":

        print("\nNormalizing ASSISTments12 columns...")

        rename_map = {

            "problem_id": "item_id"

        }

        df = df.rename(
            columns=rename_map
        )

        # ----------------------------------------------------
        # 时间字段
        # ----------------------------------------------------

        if "start_time" not in df.columns:

            raise KeyError(
                "ASSISTments12 missing column: start_time"
            )

        if "end_time" not in df.columns:

            print(
                "Warning: end_time not found."
            )

            df["end_time"] = df["start_time"]

        # ----------------------------------------------------
        # attempt_count
        # ----------------------------------------------------

        if "attempt_count" not in df.columns:

            print(
                "Warning: attempt_count not found."
            )

            df["attempt_count"] = 1

        return df

    # ========================================================
    # ASSISTments17
    # ========================================================

    elif dataset_type == "assistments17":

        print("\nNormalizing ASSISTments17 columns...")

        rename_map = {

            "ITEST_id": "user_id",

            "problemId": "item_id",

            "skill": "skill_id",

            "attemptCount": "attempt_count",

            "problemType": "problem_type",

            "startTime": "start_time",

            "endTime": "end_time",

            "confidence(FRUSTRATED)":
                "Average_confidence(FRUSTRATED)",

            "confidence(CONFUSED)":
                "Average_confidence(CONFUSED)",

            "confidence(CONCENTRATING)":
                "Average_confidence(CONCENTRATING)",

            "confidence(BORED)":
                "Average_confidence(BORED)"
        }

        df = df.rename(
            columns=rename_map
        )

        # ----------------------------------------------------
        # 防止 attempt_count 缺失
        # ----------------------------------------------------

        if "attempt_count" not in df.columns:

            print(
                "Warning: attempt_count not found."
            )

            df["attempt_count"] = 1

        # ----------------------------------------------------
        # end_time 缺失
        # ----------------------------------------------------

        if "end_time" not in df.columns:

            print(
                "Warning: end_time not found."
            )

            df["end_time"] = df["start_time"]

        return df

    else:

        raise ValueError(
            f"Unsupported dataset: {dataset_type}"
        )


# ============================================================
# 7. Problem Type 统一处理
# ============================================================

def process_problem_type(
    df,
    dataset_type,
    data_path
):

    # --------------------------------------------------------
    # problem_type 不存在
    # --------------------------------------------------------

    if "problem_type" not in df.columns:

        print(
            "Warning: problem_type column not found."
        )

        df["problem_type"] = "unknown"

    # --------------------------------------------------------
    # 缺失值
    # --------------------------------------------------------

    df["problem_type"] = (
        df["problem_type"]
        .fillna("unknown")
    )

    # --------------------------------------------------------
    # 根据数据集进行映射
    # --------------------------------------------------------

    if dataset_type == "assistments12":

        df["problem_type_id"] = (
            df["problem_type"]
            .apply(group_problem_type_12)
        )

        mapping_info = {

            "0":
                "Others (algebra, fill_in, open_response, rank, etc.) -> Guess Limit ~0.01",

            "1":
                "Choose (choose_1, choose_n) -> Guess Limit ~0.30"
        }

    elif dataset_type == "assistments17":

        df["problem_type_id"] = (
            df["problem_type"]
            .apply(group_problem_type_17)
        )

        mapping_info = {

            "0":
                "Others (textfield, noprobtype, etc.) -> Guess Limit ~0.01",

            "1":
                "Choose (radioquestion) -> Guess Limit ~0.30"
        }

    # --------------------------------------------------------
    # 统计
    # --------------------------------------------------------

    count_1 = (
        df["problem_type_id"] == 1
    ).sum()

    count_0 = (
        df["problem_type_id"] == 0
    ).sum()

    print("\n" + "=" * 60)

    print("Problem Type Statistics")

    print(
        f"ID 1 (Choose Group): {count_1}"
    )

    print(
        f"ID 0 (Others Group): {count_0}"
    )

    print("=" * 60 + "\n")

    # --------------------------------------------------------
    # 保存映射信息
    # --------------------------------------------------------

    mapping_path = os.path.join(
        data_path,
        "type_mapping_info.pkl"
    )

    with open(
        mapping_path,
        "wb"
    ) as f:

        pickle.dump(
            mapping_info,
            f
        )

    return df


# ============================================================
# 8. 时间处理
# ============================================================

def process_timestamp(
    df,
    dataset_type
):

    print("\nProcessing timestamps...")

    # ========================================================
    # ASSISTments12
    # ========================================================

    if dataset_type == "assistments12":

        df["start_time"] = pd.to_datetime(
            df["start_time"]
        )

        df["end_time"] = pd.to_datetime(
            df["end_time"]
        )

    # ========================================================
    # ASSISTments17
    # ========================================================

    elif dataset_type == "assistments17":

        try:

            df["start_time"] = pd.to_datetime(
                df["start_time"],
                unit="s"
            )

            df["end_time"] = pd.to_datetime(
                df["end_time"],
                unit="s"
            )

        except Exception:

            print(
                "Unix timestamp conversion failed."
            )

            print(
                "Trying normal datetime conversion..."
            )

            df["start_time"] = pd.to_datetime(
                df["start_time"]
            )

            df["end_time"] = pd.to_datetime(
                df["end_time"]
            )

    # ========================================================
    # 统一 timestamp
    # ========================================================

    df["timestamp"] = (

        df["start_time"]

        -

        df["start_time"].min()

    )

    df["timestamp"] = (

        df["timestamp"]

        .dt.total_seconds()

        .fillna(0)

        .astype(np.int64)

    )

    return df


# ============================================================
# 9. 情绪特征处理
# ============================================================

def process_emotions(df):

    print("\nProcessing emotion features...")

    # --------------------------------------------------------
    # 不存在则补 0
    # --------------------------------------------------------

    for col in EMOTION_COLUMNS:

        if col not in df.columns:

            print(
                f"Warning: {col} not found. Filling with 0."
            )

            df[col] = 0.0

    # --------------------------------------------------------
    # 转数值
    # --------------------------------------------------------

    for col in EMOTION_COLUMNS:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # 缺失值
    # --------------------------------------------------------

    df[EMOTION_COLUMNS] = (

        df[EMOTION_COLUMNS]

        .fillna(0.0)

    )

    # --------------------------------------------------------
    # MinMax Normalization
    # --------------------------------------------------------

    scaler = MinMaxScaler()

    df[EMOTION_COLUMNS] = (

        scaler.fit_transform(
            df[EMOTION_COLUMNS]
        )

    )

    return df


# ============================================================
# 10. 数据清洗
# ============================================================

def clean_data(
    df,
    min_user_inter_num,
    remove_nan_skills
):

    print("\nCleaning data...")

    # --------------------------------------------------------
    # correct
    # --------------------------------------------------------

    if "correct" not in df.columns:

        raise KeyError(
            "Missing required column: correct"
        )

    df["correct"] = pd.to_numeric(
        df["correct"],
        errors="coerce"
    )

    df = df[
        df["correct"].isin([0, 1])
    ]

    df["correct"] = (

        df["correct"]

        .astype(np.int32)

    )

    # --------------------------------------------------------
    # 必须存在的列
    # --------------------------------------------------------

    required_columns = [

        "user_id",

        "item_id",

        "skill_id",

        "start_time"
    ]

    for col in required_columns:

        if col not in df.columns:

            raise KeyError(

                f"Missing required column: {col}"

            )

    # --------------------------------------------------------
    # 处理 Skill
    # --------------------------------------------------------

    if remove_nan_skills:

        df = df.dropna(
            subset=[
                "skill_id"
            ]
        )

        df = df[

            df["skill_id"]

            .astype(str)

            .str.lower()

            .isin(["nan", "none"])

            == False

        ]

    else:

        df.loc[

            df["skill_id"].isnull(),

            "skill_id"

        ] = -1

    # --------------------------------------------------------
    # 删除 User / Item 缺失
    # --------------------------------------------------------

    df = df.dropna(

        subset=[

            "user_id",

            "item_id"

        ]

    )

    # --------------------------------------------------------
    # 用户最小交互次数
    # --------------------------------------------------------

    print(

        f"Filtering users with less than "

        f"{min_user_inter_num} interactions..."

    )

    df = (

        df.groupby("user_id")

        .filter(

            lambda x:

            len(x)

            >=

            min_user_inter_num

        )

    )

    return df


# ============================================================
# 11. ID 编码
# ============================================================

def encode_ids(df):

    print("\nEncoding IDs...")

    # --------------------------------------------------------
    # User ID
    # --------------------------------------------------------

    df["user_id"] = (

        np.unique(

            df["user_id"],

            return_inverse=True

        )[1]

    )

    # --------------------------------------------------------
    # Item ID
    # --------------------------------------------------------

    df["item_id"] = (

        np.unique(

            df["item_id"],

            return_inverse=True

        )[1]

    )

    # --------------------------------------------------------
    # Skill ID
    # --------------------------------------------------------

    df["skill_id"] = (

        np.unique(

            df["skill_id"]

            .astype(str),

            return_inverse=True

        )[1]

    )

    return df


# ============================================================
# 12. 构建 Q 矩阵
# ============================================================

def build_q_matrix(df):

    print("\nBuilding Q-Matrix...")

    num_items = (

        df["item_id"].max()

        +

        1

    )

    num_skills = (

        df["skill_id"].max()

        +

        1

    )

    # --------------------------------------------------------
    # 去重
    # --------------------------------------------------------

    unique_qs = (

        df[

            [

                "item_id",

                "skill_id"

            ]

        ]

        .drop_duplicates()

    )

    rows = (

        unique_qs["item_id"]

        .values

    )

    cols = (

        unique_qs["skill_id"]

        .values

    )

    data = np.ones(

        len(rows),

        dtype=np.float32

    )

    # --------------------------------------------------------
    # Sparse Q Matrix
    # --------------------------------------------------------

    Q_mat_sparse = csr_matrix(

        (

            data,

            (

                rows,

                cols

            )

        ),

        shape=(

            num_items,

            num_skills

        )

    )

    print(

        f"Q-Matrix Shape: "

        f"{Q_mat_sparse.shape}"

    )

    return Q_mat_sparse


# ============================================================
# 13. 删除重复交互
# ============================================================

def remove_duplicate_interactions(df):

    print(

        "\nRemoving duplicate interactions..."

    )

    # --------------------------------------------------------
    # 使用 user + timestamp
    # --------------------------------------------------------

    df = df.drop_duplicates(

        subset=[

            "user_id",

            "timestamp"

        ],

        keep="first"

    )

    return df


# ============================================================
# 14. Skill Combination 重映射
# ============================================================

def remap_skill_combinations(
    df,
    Q_mat_sparse
):

    print(

        "\nRemapping Skill Combinations..."

    )

    try:

        # ----------------------------------------------------
        # 如果矩阵过大，转 Dense 会占用大量内存
        # ----------------------------------------------------

        Q_mat_dense = (

            Q_mat_sparse

            .toarray()

        )

        # ----------------------------------------------------
        # 每个 Item 的技能组合
        # ----------------------------------------------------

        _, unique_skill_inv = np.unique(

            Q_mat_dense,

            axis=0,

            return_inverse=True

        )

        df["skill_id"] = (

            unique_skill_inv[

                df["item_id"]

            ]

        )

        print(

            "# Preprocessed Unique Skills "

            f"(Skill Combinations): "

            f"{df['skill_id'].nunique()}"

        )

    except MemoryError:

        print(

            "\nWARNING: MemoryError!"

        )

        print(

            "Keeping original skill IDs."

        )

    return df


# ============================================================
# 15. 保存预处理数据
# ============================================================

def save_preprocessed_data(
    df,
    data_path,
    Q_mat_sparse
):

    print("\n" + "=" * 60)

    print("Dataset Statistics")

    print(

        "# Users:",

        df["user_id"].nunique()

    )

    print(

        "# Skills:",

        df["skill_id"].nunique()

    )

    print(

        "# Items:",

        df["item_id"].nunique()

    )

    print(

        "# Interactions:",

        len(df)

    )

    print("=" * 60)

    # --------------------------------------------------------
    # 排序
    # --------------------------------------------------------

    df = df.sort_values(

        by=[

            "user_id",

            "timestamp"

        ]

    )

    # --------------------------------------------------------
    # 最终保存列
    # --------------------------------------------------------

    final_columns = [

        "user_id",

        "item_id",

        "timestamp",

        "correct",

        "attempt_count",

        "skill_id",

        "problem_type_id",

        "start_time",

        "end_time"

    ]

    final_columns += EMOTION_COLUMNS

    # --------------------------------------------------------
    # 只保存存在列
    # --------------------------------------------------------

    cols_to_save = [

        col

        for col in final_columns

        if col in df.columns

    ]

    df = df[

        cols_to_save

    ]

    df = (

        df

        .reset_index(

            drop=True

        )

    )

    # ========================================================
    # 保存 preprocessed_df.csv
    # ========================================================

    output_csv = os.path.join(

        data_path,

        "preprocessed_df.csv"

    )

    print(

        f"\nSaving preprocessed data to:\n"

        f"{output_csv}"

    )

    df.to_csv(

        output_csv,

        sep="\t",

        index=False

    )

    # ========================================================
    # 保存 question_skill_rel.pkl
    # ========================================================

    q_pkl_path = os.path.join(

        data_path,

        "question_skill_rel.pkl"

    )

    with open(

        q_pkl_path,

        "wb"

    ) as f:

        pickle.dump(

            Q_mat_sparse,

            f

        )

    # ========================================================
    # 保存 q_mat.npz
    # ========================================================

    q_npz_path = os.path.join(

        data_path,

        "q_mat.npz"

    )

    sparse.save_npz(

        q_npz_path,

        Q_mat_sparse

    )

    print("\n" + "=" * 60)

    print("Preprocessing Complete!")

    print("=" * 60)

    print(

        f"CSV:\n"

        f"  {output_csv}"

    )

    print(

        f"Q Matrix PKL:\n"

        f"  {q_pkl_path}"

    )

    print(

        f"Q Matrix NPZ:\n"

        f"  {q_npz_path}"

    )

    print("=" * 60 + "\n")


# ============================================================
# 16. 主预处理函数
# ============================================================

def prepare_assistments(

    data_name,

    min_user_inter_num,

    remove_nan_skills

):

    # --------------------------------------------------------
    # Dataset Type
    # --------------------------------------------------------

    dataset_type = (

        data_name

        .strip()

        .lower()

    )

    if dataset_type not in [

        "assistments12",

        "assistments17"

    ]:

        raise ValueError(

            "\nUnsupported dataset!\n"

            "Currently supported:\n"

            "  assistments12\n"

            "  assistments17\n"

        )

    # --------------------------------------------------------
    # 数据目录
    # --------------------------------------------------------

    data_path = os.path.join(

        BASE_PATH,

        dataset_type

    )

    # --------------------------------------------------------
    # 创建目录
    # --------------------------------------------------------

    os.makedirs(

        data_path,

        exist_ok=True

    )

    print("\n")

    print("=" * 70)

    print(

        f"START PREPROCESSING: "

        f"{dataset_type}"

    )

    print("=" * 70)

    print(

        f"Dataset Path: "

        f"{data_path}"

    )

    # ========================================================
    # Step 1
    # 读取数据
    # ========================================================

    if dataset_type == "assistments12":

        df = load_assistments12(

            data_path

        )

    elif dataset_type == "assistments17":

        df = load_assistments17(

            data_path

        )

    print(

        f"\nOriginal Data Shape: "

        f"{df.shape}"

    )

    # ========================================================
    # Step 2
    # 统一列名
    # ========================================================

    df = normalize_dataset_columns(

        df,

        dataset_type

    )

    # ========================================================
    # Step 3
    # Problem Type
    # ========================================================

    df = process_problem_type(

        df,

        dataset_type,

        data_path

    )

    # ========================================================
    # Step 4
    # 时间处理
    # ========================================================

    df = process_timestamp(

        df,

        dataset_type

    )

    # ========================================================
    # Step 5
    # 情绪特征
    # ========================================================

    df = process_emotions(

        df

    )

    # ========================================================
    # Step 6
    # 数据清洗
    # ========================================================

    df = clean_data(

        df,

        min_user_inter_num,

        remove_nan_skills

    )

    # ========================================================
    # Step 7
    # ID Encoding
    # ========================================================

    df = encode_ids(

        df

    )

    # ========================================================
    # Step 8
    # Q Matrix
    # ========================================================

    Q_mat_sparse = build_q_matrix(

        df

    )

    # ========================================================
    # Step 9
    # 删除重复
    # ========================================================

    df = remove_duplicate_interactions(

        df

    )

    # ========================================================
    # Step 10
    # Skill Combination
    # ========================================================

    df = remap_skill_combinations(

        df,

        Q_mat_sparse

    )

    # ========================================================
    # Step 11
    # 保存
    # ========================================================

    save_preprocessed_data(

        df,

        data_path,

        Q_mat_sparse

    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    parser = ArgumentParser(

        description=(

            "Unified ASSISTments12 / "

            "ASSISTments17 Preprocessing"

        )

    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    parser.add_argument(

        "--data_name",

        type=str,

        default="assistments17",

        choices=[

            "assistments12",

            "assistments17"

        ],

        help=(

            "Dataset name"

        )

    )

    # --------------------------------------------------------
    # Minimum interaction
    # --------------------------------------------------------

    parser.add_argument(

        "--min_user_inter_num",

        type=int,

        default=5,

        help=(

            "Minimum number of interactions "

            "for each student"

        )

    )

    # --------------------------------------------------------
    # Remove NaN Skills
    # --------------------------------------------------------

    parser.add_argument(

        "--remove_nan_skills",

        action="store_true",

        help=(

            "Remove interactions "

            "with missing skills"

        )

    )

    args = parser.parse_args()

    # ========================================================
    # Start
    # ========================================================

    prepare_assistments(

        data_name=

        args.data_name,

        min_user_inter_num=

        args.min_user_inter_num,

        remove_nan_skills=

        args.remove_nan_skills

    )