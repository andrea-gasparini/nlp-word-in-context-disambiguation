import torch
from typing import Optional, Dict


class WiCDisambiguationClassifier(torch.nn.Module):

    def __init__(self, n_features: int, n_hidden: int) -> None:
        super().__init__()

        self.lin1 = torch.nn.Linear(n_features, n_hidden)
        self.lin2 = torch.nn.Linear(n_hidden, 1)
        self.loss_fn = torch.nn.BCELoss()

    def save_weights(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str, device: str) -> None:
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)
        self.eval()

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        out = self.lin1(x)
        out = torch.relu(out)
        out = self.lin2(out).squeeze(1)
        out = torch.sigmoid(out)

        result = {'pred': out}

        if y is not None:
            loss = self.loss(out, y)
            result['loss'] = loss

        return result

    def loss(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred, y)
