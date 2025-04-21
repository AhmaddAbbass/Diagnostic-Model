import torch.nn.functional as F
import pytorch_lightning as pl
from derm_mmdiagnosis.models.image_encoder.derm_efficientnet import DermEfficientNet

class DermContrastiveModel(pl.LightningModule):
    def __init__(self, lr=3e-4, temp=0.5):
        super().__init__()
        self.encoder   = DermEfficientNet(num_classes=0)
        self.projector = nn.Sequential(
            nn.Linear(1280,1280), nn.ReLU(inplace=True),
            nn.Linear(1280,128)
        )
        self.lr   = lr
        self.temp = temp

    import torch
    import torch.nn.functional as F

    def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
            """
            Compute the SimCLR InfoNCE loss between two batches of representations.
            
            Args:
                z1 (Tensor): Projection of first view, shape (B, D)
                z2 (Tensor): Projection of second view, shape (B, D)
                temperature (float): Softmax temperature (τ)
            
            Returns:
                Tensor: Scalar loss
            """
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)

            B = z1.size(0)
            representations = torch.cat([z1, z2], dim=0)  # (2B, D)

            # Compute similarity matrix
            sim_matrix = torch.matmul(representations, representations.T)  # (2B, 2B)
            sim_matrix = sim_matrix / temperature

            # Mask to remove similarity of samples with themselves
            mask = torch.eye(2 * B, device=sim_matrix.device).bool()
            sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

            # Positive pairs: (i, i + B) and (i + B, i)
            positives = torch.cat([
                torch.diag(sim_matrix, B),
                torch.diag(sim_matrix, -B)
            ], dim=0)  # (2B,)

            # Negatives: all rows of sim_matrix except the positive entry
            labels = torch.zeros(2 * B, dtype=torch.long, device=sim_matrix.device)
            logits = torch.cat([positives.unsqueeze(1), sim_matrix[~mask].view(2 * B, -1)], dim=1)

            return F.cross_entropy(logits, labels)


    def training_step(self, batch, batch_idx):
        x1,x2 = batch
        h1,_  = self.encoder(x1)
        h2,_  = self.encoder(x2)
        z1    = self.projector(h1)
        z2    = self.projector(h2)
        loss  = self.info_nce_loss(z1,z2)
        self.log('train/contrastive_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
