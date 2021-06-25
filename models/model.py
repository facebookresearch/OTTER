import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ema import ExponentialMovingAverage
from loss.contrastive_loss import ContrastiveLoss
from loss.kl_div_loss import KLDivLoss
from models.model_util import (concat_all_gather, sinkhorn)
from utils.utils import get_rank


class TextModel(nn.Module):
    """
    A thin wrapper that bundles together a text model and its tokenizer.
    """

    def __init__(self, text_arch, model, tokenizer, feature_dim, out_dim, max_length=60):
        super().__init__()
        self.text_arch = text_arch
        self.model = model
        self.tokenizer = tokenizer
        self.feature_dim = feature_dim
        self.out_dim = out_dim
        self.max_length = max_length

        if self.feature_dim != self.out_dim:
            self.fc = nn.Linear(self.feature_dim, self.out_dim)
        else:
            self.fc = nn.Identity()

    def tokenize(self, texts):
        if type(texts) == tuple:
            texts = list(texts)
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        )
        return tokens

    def text_output_to_embedding(self, sequence_output, tokens):
        sequence_output = self.fc(sequence_output)
        attention_mask = tokens["attention_mask"].unsqueeze(-1)
        txt_emb = (sequence_output * attention_mask).sum(dim=1) / (
            torch.clamp(attention_mask.sum(dim=1), min=1e-9))
        return txt_emb

    def forward(self, texts):
        tokens = self.tokenize(texts).to("cuda")
        sequence_output = self.model(**tokens)[0]
        txt_emb = self.text_output_to_embedding(sequence_output, tokens)
        return {"txt_emb": txt_emb}


class ImageModel(nn.Module):
    """
    A thin wrapper around the image model. Useful for adding linear layers on
    top or modifying architecture.
    """

    def __init__(self, image_arch, model, out_dim):
        super().__init__()
        self.image_arch = image_arch
        self.model = model
        self.out_dim = out_dim

        # If out_dim is not equal to the output shape of the image model, reset the fc layer.
        if hasattr(model, 'num_classes') and getattr(model, 'num_classes') != self.out_dim:
            self.model.reset_classifier(num_classes=self.out_dim)

    def forward(self, x):
        img_emb = self.model(x)
        return {"img_emb": img_emb}


class ImageTextModel(nn.Module):
    """
    ImageTextModel bundles pre-built image and text encoders.
    """

    def __init__(self, image_encoder, text_encoder, label_smoothing=0.0):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.contrastive_loss = ContrastiveLoss(
            T=3.9, label_smoothing=label_smoothing, temp_grad=True,
        )

    def get_temperature(self):
        return self.contrastive_loss.T.item()

    def get_temperature_str(self):
        return f"{self.get_temperature():.3f}"

    def compute_logits(self, features):
        """
        Computes image and text logits from embeddings output by encoders.
        """
        img_emb = features["img_emb"]  # mc
        txt_emb = features["txt_emb"]  # mc

        img_gather = concat_all_gather(img_emb)  # nc
        txt_gather = concat_all_gather(txt_emb)  # nc

        img_logits = torch.einsum('mc,nc->mn', [img_emb, txt_gather])  # mn
        txt_logits = torch.einsum('mc,nc->mn', [txt_emb, img_gather])  # mn
        return img_logits, txt_logits

    def compute_sims(self, features):
        """
        Computes similarity matrices used in the optimal transport module.
        """
        img_emb = features["img_emb"]  # mc
        txt_emb = features["txt_emb"]  # mc

        img_gather = concat_all_gather(img_emb)  # nc
        txt_gather = concat_all_gather(txt_emb)  # nc

        vt_sim = torch.einsum('mc,nc->mn', [img_emb, txt_gather])
        tv_sim = torch.einsum('mc,nc->mn', [txt_emb, img_gather])
        vv_sim = torch.einsum("mc,nc->mn", [img_emb, img_gather])
        tt_sim = torch.einsum("mc,nc->mn", [txt_emb, txt_gather])

        return {
            "vv_sim": vv_sim,
            "tt_sim": tt_sim,
            "vt_sim": vt_sim,
            "tv_sim": tv_sim,
        }

    def compute_contrastive_loss(self, image_logits, text_logits):
        """
        Computes the infoNCE loss for both image and text logits.
        """
        image_loss = self.contrastive_loss(image_logits)
        text_loss = self.contrastive_loss(text_logits)
        contrastive_loss = image_loss + text_loss
        return contrastive_loss

    def forward_features(self, images, texts):
        """
        Returns a dictionary with keys: "img_emb" and "txt_emb".
        """
        img_output = self.image_encoder(images)
        img_emb = F.normalize(img_output["img_emb"], dim=1)

        txt_output = self.text_encoder(texts)
        txt_emb = F.normalize(txt_output["txt_emb"], dim=1)

        return {
            "img_emb": img_emb,
            "txt_emb": txt_emb,
        }

    def forward(self, images, texts):
        """
        This forward function is called in the infoNCE mode, when there is no
        distillation/OTTER.
        """
        losses = {}
        features = self.forward_features(images, texts)
        image_logits, text_logits = self.compute_logits(features)
        contrastive_loss = self.compute_contrastive_loss(image_logits, text_logits)
        losses["contrastive"] = contrastive_loss
        return losses


class DistillationModel(nn.Module):
    def __init__(
        self,
        student,
        teacher,
        alpha,
        ema=False,
        ema_decay=0.999,
        T_t=100.,
        ot_dist=False,
        sinkhorn_lambda=0.05,
        sinkhorn_iter=5,
        vv_coef=1.0,
        tt_coef=1.0,
        global_ot=True,
        remove_diag=True
    ):
        """
        Distillation model takes in pre-built student and teacher
        ImageTextModels, and perform knowledge distillation.  alpha is the
        prior probability that the default pairing is correct (set as a
        hyper-parameter).
        """
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.ema = ema
        if ema:
            self.ema_model = ExponentialMovingAverage(
                self.student.parameters(), decay=ema_decay,
            )
        self.T_t = T_t
        self.ot_dist = ot_dist
        self.sinkhorn_lambda = sinkhorn_lambda
        self.sinkhorn_iter = sinkhorn_iter
        self.alpha = alpha
        self.dist_loss = KLDivLoss()
        self.vv_coef = vv_coef
        self.tt_coef = tt_coef
        self.global_ot = global_ot
        self.remove_diag = remove_diag

    def get_temperature(self):
        return [
            self.student.contrastive_loss.T.item(),
            self.dist_loss.T_s.item(),
        ]

    def get_temperature_str(self):
        return f"T_c: {self.student.contrastive_loss.T.item():.3f} | " \
               f"T_s: {self.dist_loss.T_s.item():.3f} "

    def compute_dist_loss(self, img_logits_s, txt_logits_s, sims):
        """
        Computes distillation loss computed over image and text logits.
        """
        if self.ot_dist:
            # Compute the cost matrix for sinkhorn. Add large values to the
            # diagonal of the cost matrix to ensure the output of sinkhorn is
            # close to 0 on the diagonal. The is because OTTER uses sinkhorn
            # to model off-diagonal target probabilities.
            diag = (torch.eye(*sims["vv_sim"].shape) * self.remove_diag * 1e2).to(img_logits_s)
            vv_sim = (sims["vv_sim"] - diag) * self.vv_coef
            tt_sim = (sims["tt_sim"] - diag) * self.tt_coef
            vt_sim = sims["vt_sim"]
            tv_sim = sims["tv_sim"]

            img_cost_mat = - (vv_sim + tt_sim + vt_sim)
            txt_cost_mat = - (vv_sim + tt_sim + tv_sim)

            if self.global_ot:
                # All gather cost mat for global OT. If turned off, OT is only
                # performed locally on each GPU.
                img_cost_mat = concat_all_gather(img_cost_mat)
                txt_cost_mat = concat_all_gather(txt_cost_mat)

            # Perform sinkhorn based on the cost matrix, and then row-normalize
            # to get target probability.
            img_target_prob = sinkhorn(
                img_cost_mat, self.sinkhorn_lambda, self.sinkhorn_iter,
            )
            txt_target_prob = sinkhorn(
                txt_cost_mat, self.sinkhorn_lambda, self.sinkhorn_iter,
            )
            img_target_prob /= img_target_prob.sum(dim=1, keepdim=True)
            txt_target_prob /= txt_target_prob.sum(dim=1, keepdim=True)

            # Get the target probability corresponding to the current GPU.
            if self.global_ot:
                rank = get_rank()
                bs = vv_sim.size(0)
                img_target_prob = img_target_prob[rank * bs: (rank + 1) * bs, :]
                txt_target_prob = txt_target_prob[rank * bs: (rank + 1) * bs, :]
        else:
            # Knowledge distillation mode: logits are directly used as target
            # probabilities.
            img_target_prob = F.softmax(sims["vt_sim"] * self.T_t, dim=1)
            txt_target_prob = F.softmax(sims["tv_sim"] * self.T_t, dim=1)

        img_dist_loss = self.dist_loss(pred=img_logits_s, target_prob=img_target_prob)
        txt_dist_loss = self.dist_loss(pred=txt_logits_s, target_prob=txt_target_prob)
        dist_loss = img_dist_loss + txt_dist_loss
        return dist_loss

    def ema_step(self):
        self.ema_model.update(self.student.parameters())
        self.ema_model.copy_to(self.teacher.parameters())

    def forward(self, images, texts):
        losses = {}

        if self.ema:
            self.ema_step()
        features_s = self.student.forward_features(images, texts)
        img_logits_s, txt_logits_s = self.student.compute_logits(features_s)

        # Compute InfoNCE Loss.
        contrastive_loss = self.student.compute_contrastive_loss(img_logits_s, txt_logits_s)

        # Compute similarity matrices using the teacher model.
        with torch.no_grad():
            features_t = self.teacher.forward_features(images, texts)
            sims_t = self.teacher.compute_sims(features_t)

        # Compute distillation loss.
        dist_loss = self.compute_dist_loss(img_logits_s, txt_logits_s, sims_t)

        losses["contrastive"] = self.alpha * contrastive_loss
        losses["distillation"] = (1 - self.alpha) * dist_loss
        return losses
