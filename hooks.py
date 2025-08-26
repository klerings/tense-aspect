import torch


def make_add_constant_hook_fixed(vector, alpha, position):
    def add_constant_hook(module, input, output):

        output_list = list(output)
        resid = output_list[0]  # (batch, seq_len, model_dim)

        # Apply modification
        modified_resid = resid + alpha * vector
        if position == "final":
            output_list[0][:, -1, :] = modified_resid[:, -1, :]
        elif position == "first":
            output_list[0][:, 0, :] = modified_resid[:, 0, :]
        else:
            raise Exception(f"position {position} not defined")
        return tuple(output_list)

    return add_constant_hook


def make_add_constant_hook_dynamic(vector, alpha, position):
    def add_constant_hook(module, input, output):

        output_list = list(output)
        resid = output_list[0]  # (batch, seq_len, model_dim)

        # dot product to compute alignment score
        projection_magnitudes = torch.einsum("...d,d->...", resid, vector)

        # Compute an inverse scaling factor: small projections get larger corrections
        inverse_scaling = 1.0 / (
            1.0 + projection_magnitudes.abs()
        )  # Avoid division by zero

        # Scale the steering direction inversely to alignment
        steering_adjustment = inverse_scaling.unsqueeze(-1) * vector

        # Apply modification
        modified_resid = resid + alpha * steering_adjustment
        if position == "final":
            output_list[0][:, -1, :] = modified_resid[:, -1, :]
        elif position == "first":
            output_list[0][:, 0, :] = modified_resid[:, 0, :]
        else:
            raise Exception(f"position {position} not defined")
        return tuple(output_list)

    return add_constant_hook


def make_add_constant_hook_first_step_only(vector, alpha, position="final"):
    """
    Apply steering only at the first generation step on the final token.
    """
    step_counter = [0]  # Use list as mutable counter

    def add_constant_hook(module, input, output):
        output_list = list(output)
        resid = output_list[0]  # (batch, seq_len, model_dim)

        # Only apply steering at the first step
        if step_counter[0] == 0 and position == "final":
            # Apply modification
            modified_resid = resid + alpha * vector
            output_list[0][:, -1, :] = modified_resid[:, -1, :]

        # Increment counter for next step
        step_counter[0] += 1

        return tuple(output_list)

    return add_constant_hook


def make_add_constant_hook_decreasing(
    vector, alpha, decay_factor=0.9, position="final"
):
    """
    Apply steering with decreasing intensity on final tokens at each generation step.
    decay_factor: How quickly the intensity decreases (0-1)
    """
    step_counter = [0]  # Use list as mutable counter

    def add_constant_hook(module, input, output):
        output_list = list(output)
        resid = output_list[0]  # (batch, seq_len, model_dim)

        # Calculate decayed alpha
        current_alpha = alpha * (decay_factor ** step_counter[0])

        if position == "final":
            # Apply modification with decayed alpha
            modified_resid = resid + current_alpha * vector
            output_list[0][:, -1, :] = modified_resid[:, -1, :]

        # Increment counter for next step
        step_counter[0] += 1

        return tuple(output_list)

    return add_constant_hook


def make_add_constant_hook_every_second(vector, alpha, position="final"):
    """
    Apply steering only on every second generation step.
    """
    step_counter = [0]  # Use list as mutable counter

    def add_constant_hook(module, input, output):
        output_list = list(output)
        resid = output_list[0]  # (batch, seq_len, model_dim)

        # Apply steering only on even steps (0, 2, 4, ...)
        if step_counter[0] % 2 == 0 and position == "final":
            # Apply modification
            modified_resid = resid + alpha * vector
            output_list[0][:, -1, :] = modified_resid[:, -1, :]

        # Increment counter for next step
        step_counter[0] += 1

        return tuple(output_list)

    return add_constant_hook


def make_subtract_tense_specific_hook(
    vector, sv_dict, feature_name, alpha, position="final"
):
    """
    Apply steering by subtracting tense-specific vectors based on batch_tenses.

    Args:
        sv_dict: Dictionary mapping tense names to steering vectors
        alpha: Scaling factor for the steering vectors
        position: Where to apply the steering ("final" or "first")
    """
    # Store tenses for the current batch
    current_batch_tenses = []

    def set_batch_tenses(tenses):
        current_batch_tenses.clear()
        current_batch_tenses.extend(tenses)

    def subtract_tense_specific_hook(module, input, output):
        output_list = list(output)
        resid = output_list[0]  # (batch, seq_len, model_dim)

        # Apply tense-specific modifications
        batch_size = resid.shape[0]

        # Ensure we have tense information for each item in the batch
        if len(current_batch_tenses) != batch_size:
            raise ValueError(
                f"Tense information missing: expected {batch_size}, got {len(current_batch_tenses)}"
            )

        # Apply tense-specific steering to each item in the batch
        for i in range(batch_size):
            tense_tuple = current_batch_tenses[i]
            if feature_name in ["present", "past", "future"]:
                tense = tense_tuple[0]
            elif feature_name in [
                "simple",
                "progressive",
                "perfect",
                "perfect_progressive",
            ]:
                tense = tense_tuple[1]
            else:
                raise Exception(f"unknown feature: {feature_name}")
            if tense in sv_dict:
                vector_to_subtract = sv_dict[tense]

                # Apply modification at the specified position
                if position == "final":
                    output_list[0][i, -1, :] = (
                        resid[i, -1, :] - alpha * vector_to_subtract + alpha * vector
                    )
                elif position == "first":
                    output_list[0][i, 0, :] = (
                        resid[i, 0, :] - alpha * vector_to_subtract + alpha * vector
                    )
                else:
                    raise Exception(f"Position {position} not defined")

        return tuple(output_list)

    # Return both the hook and the function to set tenses
    return subtract_tense_specific_hook, set_batch_tenses


def make_subtract_proj_tense_specific_hook(
    vector, sv_dict, feature_name, alpha, position="final"
):
    """
    Apply steering by subtracting tense-specific vectors based on batch_tenses.

    Args:
        sv_dict: Dictionary mapping tense names to steering vectors
        alpha: Scaling factor for the steering vectors
        position: Where to apply the steering ("final" or "first")
    """
    # Store tenses for the current batch
    current_batch_tenses = []

    def set_batch_tenses(tenses):
        current_batch_tenses.clear()
        current_batch_tenses.extend(tenses)

    def subtract_tense_specific_hook(module, input, output):
        output_list = list(output)
        resid = output_list[0]  # (batch, seq_len, model_dim)

        # Apply tense-specific modifications
        batch_size = resid.shape[0]

        # Ensure we have tense information for each item in the batch
        if len(current_batch_tenses) != batch_size:
            raise ValueError(
                f"Tense information missing: expected {batch_size}, got {len(current_batch_tenses)}"
            )

        # Apply tense-specific steering to each item in the batch
        for i in range(batch_size):
            tense_tuple = current_batch_tenses[i]
            if feature_name in ["present", "past", "future"]:
                tense = tense_tuple[0]
            elif feature_name in [
                "simple",
                "progressive",
                "perfect",
                "perfect_progressive",
            ]:
                tense = tense_tuple[1]
            else:
                raise Exception(f"unknown feature: {feature_name}")
            if tense in sv_dict:
                vector_to_subtract = sv_dict[tense]

                # choose the position to modify
                if position == "final":
                    pos = -1
                elif position == "first":
                    pos = 0
                else:
                    raise Exception(f"Position {position} not defined")

                # Get the current activation at the specified position
                current_activation = resid[i, pos, :]

                # calculate the projection of the activation onto vector_to_subtract
                projection_magnitude = torch.dot(
                    current_activation, vector_to_subtract
                ) / torch.dot(vector_to_subtract, vector_to_subtract)
                projection_vector = projection_magnitude * vector_to_subtract

                steered = current_activation - projection_vector + alpha * vector
                output_list[0][i, pos, :] = steered

        return tuple(output_list)

    # Return both the hook and the function to set tenses
    return subtract_tense_specific_hook, set_batch_tenses


def get_layer_norm_hook(
    name, activation_norms, activation_norms_last, current_attention_mask
):
    def hook(module, input, output):
        # compute norm per sample
        output_list = list(output)
        resid = output_list[0]  # (batch, seq_len, model_dim)

        if current_attention_mask is not None:
            # Calculate norm per token
            token_norms = resid.norm(dim=-1)  # (batch, seq_len)

            # Mask out padding tokens
            masked_token_norms = token_norms * current_attention_mask

            # Compute mean norm for each sequence (excluding padding)
            # Sum of norms divided by sequence length (sum of attention mask)
            seq_lengths = current_attention_mask.sum(dim=1)
            sum_norms = masked_token_norms.sum(dim=1)  # (batch,)
            mean_norms = sum_norms / seq_lengths  # (batch,)

            activation_norms[name].extend(mean_norms.cpu().tolist())

            # Extract the last token representation for each sequence
            last_indices = current_attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(resid.shape[0], device=resid.device)
            last_tokens = resid[batch_indices, last_indices]  # (batch, model_dim)

            # Compute norm of last token for each sequence
            last_token_norms = last_tokens.norm(dim=-1)  # (batch,)
            activation_norms_last[name].extend(last_token_norms.cpu().tolist())

    return hook
