import torch
from torch.optim import Optimizer


def is_muon_param(name, param):
    name = name.lower()
    if "emb" in name or "head" in name:
        return False
    return param.ndim == 2


def route_params(named_params):
    muon, adam, seen = [], [], set()
    for name, param in named_params:
        if not param.requires_grad or param.data_ptr() in seen:
            continue
        seen.add(param.data_ptr())
        if is_muon_param(name, param):
            muon.append(param)
        else:
            adam.append(param)
    return adam, muon


class AdamMuon(Optimizer):
    def __init__(
        self,
        named_params,
        muon_lr=0.001,
        muon_momentum=0.95,
        muon_weight_decay=0.1,
        adam_lr=0.001,
        adam_betas=(0.9, 0.999),
    ):
        named_params = list(named_params)
        super().__init__([p for n, p in named_params], defaults={})
        adam_params, muon_params = route_params(named_params)
        self.adam = torch.optim.Adam(adam_params, adam_lr, adam_betas,fused=True)
        self.muon = torch.optim.Muon(
            muon_params, muon_lr, muon_weight_decay, muon_momentum
        )

        self.param_groups = self.adam.param_groups + self.muon.param_groups

    def step(self, closure=None):
        self.muon.step()
        self.adam.step()

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none)
        self.adam.zero_grad(set_to_none)

    def state_dict(self):
        return {"adam": self.adam.state_dict(), "muon": self.muon.state_dict()}

    def load_state_dict(self, state_dict):
        self.adam.load_state_dict(state_dict["adam"])
        self.muon.load_state_dict(state_dict["muon"])
