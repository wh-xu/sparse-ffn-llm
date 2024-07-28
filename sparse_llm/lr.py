import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm.auto import tqdm


def gate_LR(mlp_inp, weight_svd, threshold=0.0):
    gate_pred = mlp_inp @ weight_svd.T
    gate_pred[gate_pred.abs() < threshold] = 0.0
    return gate_pred


def gate_MLP(mlp_inp, mlp_gate_weight, act=F.silu):
    if act is not None:
        return act(mlp_inp @ mlp_gate_weight.bfloat16().T)
    else:
        return mlp_inp @ mlp_gate_weight.bfloat16().T


def measure_sparse_gate_recall(gt, inp):
    gt, inp = gt.to("cpu"), inp.to("cpu")
    density = (inp != 0).float().mean()
    val, ind = torch.topk(gt.abs(), k=int(density * gt.shape[-1]), largest=True)
    res = torch.zeros_like(gt)
    res.scatter_(-1, ind, val)
    out = res.view(*gt.size()) * gt.sign()

    n_topk = torch.sum(out != 0)
    recall = torch.sum((out != 0) * (inp != 0.0)) / n_topk
    print(f"Recall@Density={density:.2f} = {recall:.3f}")


def get_threshold_from_density(activation: torch.Tensor, keep_ratio: list):
    threshold_list = []
    for ratio in keep_ratio:
        k = int(activation.shape[-1] * ratio)
        topk_values, topk_indices = torch.topk(
            activation.cpu().squeeze().abs(), k, dim=1
        )
        threshold_list.append(topk_values[:, -1].mean().item())

    return threshold_list


class MLP_gate_LR(nn.Module):
    def __init__(self, Q, R):
        super(MLP_gate_LR, self).__init__()
        self.Q = nn.Parameter(data=Q.bfloat16())
        self.R = nn.Parameter(data=R.bfloat16())

    def forward(self, x):
        out = x @ self.R.T @ self.Q.T
        return out


def train_LR(epochs, mlp_gate_weight, mlp_gate_inp, rank):
    y_all = gate_MLP(mlp_gate_inp, mlp_gate_weight, act=None)
    n_sample = int(y_all.shape[-2] * 0.8)
    x_train, x_test = mlp_gate_inp[:, :n_sample], mlp_gate_inp[:, n_sample:]
    y_train, y_test = y_all[:, :n_sample], y_all[:, n_sample:]

    Q, R = torch.linalg.qr(mlp_gate_weight.float())
    Q, R = Q[:, :rank], R[:rank]

    model = MLP_gate_LR(Q, R)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in tqdm(range(epochs), position=0, leave=True, ascii=True):
        # Forward pass
        outputs = model(x_train)
        loss = criterion(outputs, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 1:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}", flush=True)

            test_out = model(x_test)
            test_loss = criterion(test_out, y_test)
            print(
                f"Epoch [{epoch+1}/{epochs}], Test Loss: {test_loss.item():.4f}",
                flush=True,
            )
    return model


def generate_pred_model(weight, inp, method, **kwargs):
    if method == "svd":
        U, S, Vh = torch.linalg.svd(weight.float().to("cuda"))
    elif method == "asvd":
        S = inp.squeeze().abs().mean(dim=-2)
        S = S.view(1, -1).to("cuda")
        scaled_W = weight * S
        U, S, Vh = torch.linalg.svd(scaled_W.float())
        Vh = Vh / S.view(-1, 1)
    elif method == "csvd":
        X2 = inp.squeeze().mT @ inp.squeeze()
        X2 = X2.float().cpu()

        try:
            S = torch.linalg.cholesky(X2)
        except Exception as e:
            print("Warning: eigen scaling_diag_matrix is not positive!")
            eigenvalues = torch.linalg.eigvalsh(X2)
            X2 += (-eigenvalues[0] + 1e-3) * torch.eye(X2.shape[0])
            S = torch.linalg.cholesky(X2)

        try:
            S_inv = torch.linalg.inv(S)
        except Exception as e:
            print("Warning: scaling_diag_matrix is not full rank!")
            S += 1e-6 * torch.eye(S.shape[0]).to("cuda")
            S_inv = torch.linalg.inv(S)

        scaled_W = weight.cpu().float() @ S.cpu().float()
        U, S, Vh = torch.linalg.svd(scaled_W, full_matrices=False)
        Vh = S_inv @ Vh
    elif method == "qr":
        Q, R = torch.linalg.qr(weight.float().to("cuda"))
        return Q[:, : kwargs["rank"]].cpu(), R[: kwargs["rank"]].cpu()

    elif method == "tqr":
        QR_model = train_LR(
            epochs=kwargs["epochs"],
            mlp_gate_weight=weight,
            mlp_gate_inp=inp,
            rank=kwargs["rank"],
        )

        QR_model.eval()
        return QR_model.Q.data.detach().cpu(), QR_model.R.data.detach().cpu()
    else:
        raise NotImplementedError("Wrong method")

    Q = U[:, : kwargs["rank"]] @ torch.diag(S[: kwargs["rank"]])
    R = Vh[:, : kwargs["rank"]]
    return Q.cpu(), R.cpu()
