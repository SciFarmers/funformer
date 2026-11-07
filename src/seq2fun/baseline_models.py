import math
import os
import json
import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from typing import Optional, Dict


class Seq2FunMixin:
    """Mixin that adds save_pretrained / from_pretrained to all backbone models.

    Subclasses must implement get_backbone_config() → dict whose entries are
    exactly the keyword arguments needed to reconstruct the model via cls(**kwargs).
    """

    def get_backbone_config(self) -> dict:
        raise NotImplementedError(f"{self.__class__.__name__} must implement get_backbone_config()")

    @classmethod
    def _build_from_config(cls, backbone_kwargs: dict):
        """Construct model from backbone_kwargs. Override if the constructor needs
        special handling (e.g. tuple conversion)."""
        return cls(**backbone_kwargs)

    def save_pretrained(self, path: str) -> None:
        """Save model_config.json + weights to *path*.

        After this call the directory is self-contained for from_pretrained().
        The HF Trainer already saves model.safetensors to this dir; this method
        overwrites it with backbone-only weights (no 'backbone.' prefix) so the
        file is directly loadable without the DualStrandWrapper.
        """
        config = {
            "model_class": self.__class__.__name__,
            "backbone_kwargs": self.get_backbone_config(),
        }
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "model_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        # Overwrite model.safetensors with backbone-only weights (no prefix).
        try:
            from safetensors.torch import save_file
            save_file(self.state_dict(), os.path.join(path, "model.safetensors"))
        except ImportError:
            torch.save(self.state_dict(), os.path.join(path, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        device: Optional[str] = None,
    ) -> "Seq2FunMixin":
        """Load backbone from a checkpoint directory.

        The directory must contain:
          * model_config.json   – written by save_pretrained()
          * model.safetensors   – also written by save_pretrained()
            (pytorch_model.bin accepted as fallback)

        Examples::

            from src.seq2fun.baseline_models import DeepSEA, DanQ, DeepFormer
            from src.seq2fun.funformer import FunFormer

            model = DeepSEA.from_pretrained("checkpoints/DeepSEA_5e-4_1000/best_model")
            model = FunFormer.from_pretrained("checkpoints/funformer_.../best_model")
        """
        config_path = os.path.join(path, "model_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"No model_config.json found in '{path}'. "
                "This checkpoint predates auto-config; re-train or create the file manually."
            )
        with open(config_path) as f:
            config = json.load(f)

        model_class_name: str = config["model_class"]
        if cls.__name__ != model_class_name:
            raise ValueError(
                f"Checkpoint contains a {model_class_name} model but you called "
                f"{cls.__name__}.from_pretrained(). "
                f"Use {model_class_name}.from_pretrained('{path}') instead."
            )

        backbone_kwargs: dict = dict(config["backbone_kwargs"])
        model = cls._build_from_config(backbone_kwargs)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        st_path = os.path.join(path, "model.safetensors")
        pt_path = os.path.join(path, "pytorch_model.bin")
        if os.path.exists(st_path):
            from safetensors.torch import load_file
            state_dict = load_file(st_path, device=device)
        elif os.path.exists(pt_path):
            state_dict = torch.load(pt_path, map_location=device, weights_only=True)
        else:
            raise FileNotFoundError(
                f"No weight file found in '{path}'. "
                "Expected model.safetensors or pytorch_model.bin."
            )

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model


class DeepSEA(Seq2FunMixin, nn.Module):
    def __init__(self, n_targets=919, sequence_length=1000):
        super().__init__()
        conv_kernel_size = 8
        pool_kernel_size = 4

        self.n_targets = n_targets
        self.sequence_length = sequence_length

        self.conv_net = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.Dropout(p=0.2),

            nn.Conv1d(320, 480, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.Dropout(p=0.2),

            nn.Conv1d(480, 960, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )

        reduce_by = conv_kernel_size - 1
        pool_kernel_size = float(pool_kernel_size)
        self.n_channels = int(
            ((
                (sequence_length - reduce_by) / pool_kernel_size
            //1 - reduce_by) / pool_kernel_size
            //1 - reduce_by)
        )

        self.classifier = nn.Sequential(
            nn.Linear(960 * self.n_channels, n_targets),
            nn.ReLU(inplace=True),
            nn.Linear(n_targets, n_targets),
        )
        
        # Register pos_weight as buffer (will move to GPU with model)
        self.register_buffer('pos_weight', None)

    def get_backbone_config(self) -> dict:
        return {"n_targets": self.n_targets, "sequence_length": self.sequence_length}

    def set_pos_weight(self, pos_weight: torch.Tensor):
        """Set pos_weight for loss calculation."""
        self.register_buffer('pos_weight', pos_weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        # Input expected as (B, L, 4) -> transpose to (B, 4, L)
        if input_ids.dim() == 3 and input_ids.shape[-1] == 4:
            input_ids = input_ids.transpose(1, 2)

        out = self.conv_net(input_ids)
        reshape_out = out.view(out.size(0), 960 * self.n_channels)
        logits = self.classifier(reshape_out)

        outputs = {"logits": logits}
        if labels is not None:
            if self.pos_weight is not None:
                loss = F.binary_cross_entropy_with_logits(
                    logits, labels.float(), 
                    pos_weight=self.pos_weight, 
                    reduction="mean"
                )
            else:
                loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="mean")
            outputs["loss"] = loss

        return outputs


class DanQ(Seq2FunMixin, nn.Module):
    def __init__(self, n_targets=919, sequence_length=1000):
        super().__init__()
        self.n_targets = n_targets
        self.sequence_length = sequence_length

        # Conv + Pool + Drop
        self.conv_pool_drop_1 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=320, kernel_size=26, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=13, stride=13),
            nn.Dropout(0.2)
        )

        # Bidirectional LSTM
        self.bdlstm = nn.LSTM(
            input_size=320,
            hidden_size=320,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.dropout_2 = nn.Dropout(0.5)

        # Compute flattened size after conv + pool
        conv_out_len = (sequence_length - 26 + 1)  # Conv1d output
        pool_out_len = conv_out_len // 13          # MaxPool1d output
        lstm_out_dim = pool_out_len * 2 * 320      # bidirectional LSTM

        # Dense layers
        self.dense_1 = nn.Sequential(
            nn.Linear(lstm_out_dim, 925),
            nn.ReLU()
        )
        self.dense_2 = nn.Linear(925, n_targets)
        
        # Register pos_weight as buffer
        self.register_buffer('pos_weight', None)

    def get_backbone_config(self) -> dict:
        return {"n_targets": self.n_targets, "sequence_length": self.sequence_length}

    def set_pos_weight(self, pos_weight: torch.Tensor):
        """Set pos_weight for loss calculation."""
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        # Transpose to (B, 4, L) if needed
        if input_ids.dim() == 3 and input_ids.shape[-1] == 4:
            input_ids = input_ids.transpose(1, 2)

        x = self.conv_pool_drop_1(input_ids)  # (B, 320, L')
        x = x.transpose(1, 2)                 # (B, L', 320)
        x, _ = self.bdlstm(x)                 # (B, L', 640)
        x = self.dropout_2(x)
        x = x.reshape(x.size(0), -1)          # Flatten
        x = self.dense_1(x)
        logits = self.dense_2(x)

        outputs = {"logits": logits}
        if labels is not None:
            if self.pos_weight is not None:
                loss = F.binary_cross_entropy_with_logits(
                    logits, labels.float(), 
                    pos_weight=self.pos_weight, 
                    reduction="mean"
                )
            else:
                loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="mean")
            outputs["loss"] = loss
        return outputs


class Flow_Attention(nn.Module):
    def __init__(self, d_input, d_model, d_output, n_heads, drop_out=0.05, eps=5e-4):
        super().__init__()
        self.n_heads = n_heads
        self.query_projection = nn.Linear(d_input, d_model)
        self.key_projection = nn.Linear(d_input, d_model)
        self.value_projection = nn.Linear(d_input, d_model)
        self.out_projection = nn.Linear(d_model, d_output)
        self.dropout = nn.Dropout(drop_out)
        self.eps = eps

    def kernel_method(self, x):
        return torch.sigmoid(x)

    def dot_product(self, q, k, v):
        kv = einsum("nhld,nhlm->nhdm", k, v)
        return einsum("nhld,nhdm->nhlm", q, kv)

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        queries = self.query_projection(queries).view(B, L, self.n_heads, -1)
        keys = self.key_projection(keys).view(B, S, self.n_heads, -1)
        values = self.value_projection(values).view(B, S, self.n_heads, -1)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
      
        queries = self.kernel_method(queries)
        keys = self.kernel_method(keys)
        
        sink_incoming = 1.0 / (einsum("nhld,nhd->nhl", queries + self.eps, keys.sum(dim=2) + self.eps))
        source_outgoing = 1.0 / (einsum("nhld,nhd->nhl", keys + self.eps, queries.sum(dim=2) + self.eps))
        
        conserved_sink = einsum("nhld,nhd->nhl", queries + self.eps,
                               (keys * source_outgoing[:, :, :, None]).sum(dim=2) + self.eps)
        conserved_source = einsum("nhld,nhd->nhl", keys + self.eps,
                                 (queries * sink_incoming[:, :, :, None]).sum(dim=2) + self.eps)
        conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)
       
        sink_allocation = torch.sigmoid(conserved_sink * (float(queries.shape[2]) / float(keys.shape[2])))
        source_competition = torch.softmax(conserved_source, dim=-1) * float(keys.shape[2])
        
        x = (self.dot_product(queries * sink_incoming[:, :, :, None],
                              keys,
                              values * source_competition[:, :, :, None])
             * sink_allocation[:, :, :, None]).transpose(1, 2)
       
        x = x.reshape(B, L, -1)
        x = self.out_projection(x)
        x = self.dropout(x)
        return x


class DeepFormer(Seq2FunMixin, nn.Module):
    def __init__(self, n_targets=919, sequence_length=1000):
        super().__init__()
        self.n_targets = n_targets
        self.sequence_length = sequence_length  # dummy for HF Trainer
        conv_kernel_size = 8
        pool_kernel_size = 4
        
        self.conv_net1 = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(320, 320, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.BatchNorm1d(320)
        )
        self.conv_net2 = nn.Sequential(
            nn.Conv1d(320, 480, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(480, 480, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.BatchNorm1d(480),
            nn.Dropout(p=0.2)
        )
        self.conv_net3 = nn.Sequential(
            nn.Conv1d(480, 960, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(960, 960, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(960),
            nn.Dropout(p=0.2)
        )
        self.attn_normal = Flow_Attention(44, 44, 44, 4)
        self.classifier = nn.Sequential(
            nn.Linear(44 * 960, n_targets),
            nn.ReLU(inplace=True),
            nn.Linear(n_targets, n_targets),
        )
        
        # Register pos_weight as buffer
        self.register_buffer('pos_weight', None)

    def get_backbone_config(self) -> dict:
        return {"n_targets": self.n_targets, "sequence_length": self.sequence_length}

    def set_pos_weight(self, pos_weight: torch.Tensor):
        """Set pos_weight for loss calculation."""
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        # transpose if input is (B, L, 4)
        if input_ids.dim() == 3 and input_ids.shape[-1] == 4:
            x = input_ids.transpose(1, 2)
        else:
            x = input_ids

        x = self.conv_net1(x)
        x = self.conv_net2(x)
        x = self.conv_net3(x)
        x = self.attn_normal(x, x, x)
        x = x.view(x.size(0), 44 * 960)
        logits = self.classifier(x)

        outputs = {"logits": logits}
        if labels is not None:
            if self.pos_weight is not None:
                loss = F.binary_cross_entropy_with_logits(
                    logits, labels.float(), 
                    pos_weight=self.pos_weight, 
                    reduction="mean"
                )
            else:
                loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="mean")
            outputs["loss"] = loss
        return outputs


# ---------------------------------------------------------------------------
# DeepATT — cross-attention model with category query encoding
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            q_d_model: int = 512,
            k_d_model: int = 512,
            v_d_model: int = 512,
            num_dimensions: int = 512,
            num_heads: int = 16,
            dropout_p: float = 0.1,
    ):
        super(MultiHeadAttention, self).__init__()
        assert num_dimensions % num_heads == 0, "num_dimensions % num_heads should be zero."
        self.num_dimensions = num_dimensions
        self.d_head = int(num_dimensions / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(num_dimensions)

        self.wq = nn.Linear(q_d_model, num_dimensions)
        self.wk = nn.Linear(k_d_model, num_dimensions)
        self.wv = nn.Linear(v_d_model, num_dimensions)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.dense = nn.Linear(num_dimensions, num_dimensions)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        query = self.wq(q).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.wk(k).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.wv(v).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        score = content_score / self.sqrt_dim

        if mask is not None:
            if mask.dtype == torch.bool:
                score.masked_fill_(mask, float('-inf'))
            else:
                score += mask

        attn = torch.softmax(score, -1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.num_dimensions)

        return self.dense(context), score


class DeepATT(Seq2FunMixin, nn.Module):
    def __init__(self, n_targets=919, sequence_length=1000):
        super().__init__()
        self.n_targets = n_targets
        self.sequence_length = sequence_length

        self.conv_pool_drop_1 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=1024, kernel_size=30, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=15, stride=15),
            nn.Dropout(0.2),
        )

        self.bidirectional_rnn = nn.LSTM(
            input_size=1024, hidden_size=512, num_layers=1,
            batch_first=True, bidirectional=True,
        )

        # Register as buffer so it moves to the correct device automatically
        self.register_buffer('category_encoding', torch.eye(n_targets).unsqueeze(0))  # (1, T, T)

        self.multi_head_attention = MultiHeadAttention(
            q_d_model=n_targets, k_d_model=1024, v_d_model=1024,
            num_dimensions=400, num_heads=4,
        )

        self.dropout_2 = nn.Dropout(0.2)

        self.point_wise_dense_1 = nn.Sequential(
            nn.Linear(400, 100),
            nn.ReLU(),
        )

        # Raw logit output — sigmoid applied inside BCEWithLogitsLoss
        self.point_wise_dense_2 = nn.Linear(100, 1)

        self.register_buffer('pos_weight', None)

    def get_backbone_config(self) -> dict:
        return {"n_targets": self.n_targets, "sequence_length": self.sequence_length}

    def set_pos_weight(self, pos_weight: torch.Tensor):
        self.register_buffer('pos_weight', pos_weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # Accept (B, L, 4) or (B, 4, L) — normalise to (B, 4, L)
        if input_ids.dim() == 3 and input_ids.shape[-1] == 4:
            input_ids = input_ids.transpose(1, 2)

        batch_size = input_ids.shape[0]

        x = self.conv_pool_drop_1(input_ids)          # (B, 1024, L')
        x = x.transpose(1, 2)                         # (B, L', 1024)
        x, _ = self.bidirectional_rnn(x)              # (B, L', 1024)

        query = self.category_encoding.expand(batch_size, -1, -1)  # (B, T, T)
        x, _ = self.multi_head_attention(q=query, k=x, v=x)       # (B, T, 400)

        x = self.dropout_2(x)
        x = self.point_wise_dense_1(x)                # (B, T, 100)
        logits = self.point_wise_dense_2(x).squeeze(-1)  # (B, T)

        outputs: Dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            if self.pos_weight is not None:
                loss = F.binary_cross_entropy_with_logits(
                    logits, labels.float(),
                    pos_weight=self.pos_weight, reduction="mean",
                )
            else:
                loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="mean")
            outputs["loss"] = loss
        return outputs
