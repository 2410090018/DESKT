
import torch
from torch import nn
from torch.nn import (
    Module,
    Parameter,
    Embedding,
    Linear,
    Dropout
)
from torch.nn.init import (
    kaiming_normal_,
    xavier_uniform_
)
import torch.nn.functional as F


class DESKT(Module):

    def __init__(
        self,
        num_skills,
        dim_s=64,
        size_m=64,
        dropout=0.05,
        num_emotions=3
    ):

        super().__init__()

        self.model_name = "emotion_slip_layer"

        self.num_skills = num_skills
        self.dim_s = dim_s
        self.size_m = size_m

        # ==================================================
        # DKVMN
        # ==================================================

        self.k_emb_layer = Embedding(
            self.num_skills + 1,
            self.dim_s,
            padding_idx=0
        )

        self.Mk = Parameter(
            torch.Tensor(self.size_m, self.dim_s)
        )

        self.Mv0 = Parameter(
            torch.Tensor(self.size_m, self.dim_s)
        )

        kaiming_normal_(self.Mk)
        kaiming_normal_(self.Mv0)

        self.v_emb_layer = Embedding(
            (self.num_skills + 1) * 2,
            self.dim_s,
            padding_idx=0
        )

        self.e_layer = Linear(
            self.dim_s,
            self.dim_s
        )

        self.a_layer = Linear(
            self.dim_s,
            self.dim_s
        )

        self.f_layer = Linear(
            self.dim_s * 2,
            self.dim_s
        )

        self.dropout_layer = Dropout(dropout)

        mid = max(8, self.dim_s // 4)

        self.mastery_head = nn.Sequential(
            Linear(self.dim_s, mid),
            nn.ReLU(),
            Linear(mid, 1)
        )

        # ==================================================
        # IRT difficulty adjustment
        # ==================================================

        self.irt_gamma = nn.Parameter(
            torch.tensor(4.0)
        )

        # ==================================================
        # GRU-SlipNet
        # ==================================================

        self.slip_hidden_dim = 32

        self.emotion_gru = nn.GRU(
            input_size=num_emotions,
            hidden_size=self.slip_hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.slip_predictor = nn.Sequential(
            Linear(self.slip_hidden_dim, 16),
            nn.ReLU(),
            Dropout(dropout),
            Linear(16, 1)
        )

        # slip upper bound
        self.max_slip_rate = 0.8

        try:
            xavier_uniform_(self.v_emb_layer.weight)
        except:
            pass

        self.loss_fn = nn.BCELoss(
            reduction="none"
        )

    def forward(self, feed_dict):

        # ==================================================
        # Input
        # ==================================================

        skills = feed_dict["skills"]
        responses = feed_dict["responses"]

        emotions = feed_dict.get("emotions", None)
        difficulty = feed_dict.get("difficulty", None)

        batch_size, seq_len = skills.size()

        device = skills.device

        # ==================================================
        # Default emotion
        # ==================================================

        if emotions is None:

            emotions = torch.zeros(
                batch_size,
                seq_len,
                4,
                device=device
            )

        # ==================================================
        # Default difficulty
        # ==================================================

        if difficulty is None:

            difficulty = torch.full(
                (batch_size, seq_len),
                0.5,
                device=device
            )

        # ==================================================
        # Embedding
        # ==================================================

        v_idx = skills + (
            self.num_skills + 1
        ) * responses.clamp(0).long()

        k = self.k_emb_layer(skills)

        v = self.v_emb_layer(v_idx)

        # ==================================================
        # Attention
        # ==================================================

        w_logits = torch.matmul(
            k,
            self.Mk.T
        )

        w = torch.softmax(
            w_logits,
            dim=-1
        )

        # ==================================================
        # Memory erase/add
        # ==================================================

        e_all = torch.sigmoid(
            self.e_layer(v)
        )

        a_all = torch.tanh(
            self.a_layer(v)
        )

        Mcur = self.Mv0.unsqueeze(0).repeat(
            batch_size,
            1,
            1
        )

        Mv = [Mcur]

        for t in range(seq_len):

            wt = w[:, t, :].unsqueeze(-1)

            et = e_all[:, t, :].unsqueeze(1)

            at = a_all[:, t, :].unsqueeze(1)

            # erase
            Mcur = Mcur * (1 - wt * et)

            # add
            Mcur = Mcur + (wt * at)

            Mv.append(Mcur)

        Mv_all = torch.stack(
            Mv,
            dim=1
        )

        # ==================================================
        # Read
        # ==================================================

        read_content = (
            w.unsqueeze(-1)
            * Mv_all[:, :-1]
        ).sum(dim=2)

        mastery_input = torch.cat(
            [read_content, k],
            dim=-1
        )

        f = torch.tanh(
            self.f_layer(mastery_input)
        )

        # ==================================================
        # Mastery prediction
        # ==================================================

        latent_mastery = torch.sigmoid(
            self.mastery_head(
                self.dropout_layer(f)
            )
        ).squeeze(-1)

        item_mastery = torch.sigmoid(
            self.irt_gamma
            * (latent_mastery - difficulty)
        )

        # ==================================================
        # GRU-based dynamic slip
        # ==================================================

        # emotions:
        # [batch, seq_len, num_emotions]

        gru_out, _ = self.emotion_gru(
            emotions
        )

        # gru_out:
        # [batch, seq_len, hidden_dim]

        raw_slip = torch.sigmoid(
            self.slip_predictor(gru_out)
        ).squeeze(-1)

        # physical constraint
        slip_prob = (
            raw_slip
            * self.max_slip_rate
        )

        # ==================================================
        # Final prediction
        # ==================================================

        pred_final = item_mastery * (
            1 - slip_prob
        )

        out = {

            "pred":
                pred_final[:, 1:],

            "true":
                responses[:, 1:].float(),

            "emotions":
                emotions[:, 1:, :],

            "difficulty":
                difficulty[:, 1:],

            "mastery":
                item_mastery[:, 1:],

            "slip":
                slip_prob[:, 1:],

            "pred_pure":
                item_mastery[:, 1:],

            "w":
                w[:, 1:, :],

            "memory_state":
                Mv_all[:, 1:, :, :],

            "gru_hidden":
                gru_out[:, 1:, :]
        }

        return out

    def loss(
        self,
        feed_dict,
        out_dict
    ):

        pred_pure = out_dict[
            "pred_pure"
        ].flatten()

        pred_final = out_dict[
            "pred"
        ].flatten()

        true = out_dict[
            "true"
        ].flatten()

        slip_prob = out_dict[
            "slip"
        ].flatten()

        mask = true > -1

        valid_pure = pred_pure[mask]

        valid_final = pred_final[mask]

        valid_true = true[mask]

        valid_slip = slip_prob[mask]

        # ==================================================
        # 1. mastery loss
        # ==================================================

        loss_pure = self.loss_fn(
            valid_pure,
            valid_true
        ).mean()

        # ==================================================
        # 2. final prediction loss
        # ==================================================

        loss_final = self.loss_fn(
            valid_final,
            valid_true
        ).mean()

        # ==================================================
        # 3. weakly-supervised slip loss
        # ==================================================

        detached_pure = valid_pure.detach()

        ideal_slip = torch.where(

            valid_true == 1.0,

            torch.zeros_like(
                detached_pure
            ),

            detached_pure
            * self.max_slip_rate
        )

        loss_slip_guided = F.mse_loss(
            valid_slip,
            ideal_slip
        )

        # ==================================================
        # total loss
        # ==================================================

        total_loss = (
            loss_pure
            + loss_final
            + 2 * loss_slip_guided
        )

        return (
            total_loss,
            int(mask.sum().item()),
            float(valid_true.sum().item())
        )

