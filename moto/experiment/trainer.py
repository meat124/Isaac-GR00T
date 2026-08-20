"""Trainer for latent-motion-tokenizer pretraining."""

from typing import Optional

import torch

from lara_full.experiment.trainer import DualBrainTrainer


class MotoTokenizerTrainer(DualBrainTrainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        rgb_sequence = torch.cat(
            [inputs["rgb_initial"], inputs["rgb_future"]],
            dim=1,
        )
        outputs = model(
            cond_pixel_values=rgb_sequence[:, 0],
            target_pixel_values=rgb_sequence[:, 1],
            hand_mask=inputs.get("hand_mask"),
        )
        loss = outputs.loss.mean()
        # Factorized-masked diagnostics: per-stream masked recon + per-codebook usage.
        logging_steps = max(int(self.args.logging_steps), 1)
        if self.state.global_step % logging_steps == 0:
            extras = {}
            for key in ("stream_recons_loss_head", "stream_recons_loss_hand",
                        "active_code_num_head", "active_code_num_hand",
                        "commit_loss", "recons_loss", "perceptual_loss"):
                value = getattr(outputs, key, None)
                if value is not None and torch.is_tensor(value):
                    extras[key] = value.detach().float().mean().item()
            if extras:
                self.log(extras)
        return (loss, outputs) if return_outputs else loss

    def save_model(
        self,
        output_dir: Optional[str] = None,
        _internal_call: bool = False,
    ):
        output_dir = output_dir or self.args.output_dir
        if self.is_deepspeed_enabled:
            state_dict = self.accelerator.get_state_dict(self.deepspeed)
        else:
            state_dict = self.model.get_state_dict_to_save()

        if self.args.should_save:
            return self.model.save_pretrained(output_dir, state_dict=state_dict)
