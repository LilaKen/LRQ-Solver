from typing import Dict

import torch


class R2Score:
    @torch.no_grad()
    def __call__(self, output_dict, label_dict) -> Dict[str, torch.Tensor]:
        r2score_dict = {}

        for key in label_dict:
            output = output_dict[key]
            target = label_dict[key]
            if output.shape != target.shape:
                raise ValueError(
                    f"Output and target shapes do not match for key '{key}'. "
                    f"Output shape: {output.shape}, Target shape: {target.shape}"
                )

            output = output.flatten()
            target = target.flatten()
            target_mean = target.mean(dim=-1, keepdim=True)
            ss_tot = torch.sum((target - target_mean) ** 2)
            ss_res = torch.sum((target - output) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            r2score_dict[key] = r2.unsqueeze(0).item()
        return r2score_dict
