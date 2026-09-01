import torch

class GridDecoder:
    def __init__(self, normalization_stats=None, device='cpu'):
        """
        Decoder to translate raw model output tensors back into real-world physical values 
        and flag grid violations.
        Requires the 'y_mean' and 'y_std' from the dataset statistics to un-normalize.
        """
        if normalization_stats is None:
            self.y_mean = torch.tensor([0.0, 0.0]).to(device)
            self.y_std = torch.tensor([1.0, 1.0]).to(device)
        else:
            self.y_mean = normalization_stats['y_mean'].to(device)
            self.y_std = normalization_stats['y_std'].to(device)

    def decode_voltage(self, tensor_out):
        """
        Converts Z-score normalized output [va, vm] back to physical values.
        tensor_out is expected to be [N, 2]
        """
        # Un-normalize: physical_val = (normalized_val * std) + mean
        unnormalized = (tensor_out * self.y_std) + self.y_mean
        
        va = unnormalized[:, 0]
        vm = unnormalized[:, 1]
        return va, vm

    def check_voltage_violations(self, vm, vmin, vmax):
        """
        Flags nodes where voltage magnitude violates bounds.
        Returns a boolean mask where True = safe, False = violation.
        """
        safe_mask = (vm >= vmin) & (vm <= vmax)
        violations = ~safe_mask
        return violations

    def check_thermal_violations(self, va, vm, edge_index, edge_attr):
        """
        Approximate DC power flow to check for thermal overloads on active lines.
        Power Flow P_ij approx= (va_i - va_j) / X_ij
        edge_attr contains [br_r, br_x, rate_a]
        """
        src = edge_index[0]
        dst = edge_index[1]
        
        va_src = va[src]
        va_dst = va[dst]
        
        br_x = edge_attr[:, 1]
        rate_a = edge_attr[:, 2]
        
        # Avoid division by zero for extremely small reactances
        br_x = torch.clamp(br_x, min=1e-5)
        
        # Approximate flow
        p_flow = torch.abs(va_src - va_dst) / br_x
        
        constrained_mask = rate_a > 0.0
        overloaded = (p_flow > rate_a) & constrained_mask
        return overloaded
