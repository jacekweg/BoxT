import copy
import inspect
import os
import random
import time
import warnings
from datetime import datetime
from math import ceil
import json
from collections import defaultdict
from typing import Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import msgpack
import msgpack_numpy as m
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.tensorboard import SummaryWriter

from model_options import ModelOptions

from dataclasses import dataclass

from utils import metrics_helper
from utils.metrics_helper import rank_biased_precision

import cnst
import utils.data_prep

m.patch()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


zero = torch.tensor(0.0, device=device)
half = torch.tensor(0.5, device=device)
one = torch.tensor(1.0, device=device)
SANITY_EPS = 1e-8
NORM_LOG_BOUND = 1

BOTTOM_VALUE = np.nan

uniform_neg_sampling_call_count = 0


SIMPLE_BOX_LOSS = "simple_box_loss"

@dataclass
class Metrics:
    adjusted_mean_rank_index: float
    rank_biased_precision: float
    mean_first_rank: float
    hits_at_values: list


def sanitize_scatter(input_tensor):
    return torch.where(input_tensor == 0, input_tensor + SANITY_EPS, input_tensor)


def delta_time_string(delta):
    seconds = int(delta) % 60
    minutes = int(delta / 60) % 60
    hours = int(delta / 3600)
    return f"{hours}:{str(minutes).zfill(2)}:{str(seconds).zfill(2)}"

def print_or_log(input_string, log, log_file_path="log.txt"):
    if not log:
        print(input_string)
    else:
        with open(log_file_path, "a+") as log_file:
            log_file.write(input_string + "\r\n")


def transform(input_list, transformation_function):
    output_tuple = tuple([transformation_function(x) for x in input_list])
    return output_tuple


def q2b_loss(points, lower_corner, upper_corner):
    centres = 0.5 * (lower_corner + upper_corner)
    dist_outside = torch.clamp(points - upper_corner, min=0.0) + \
                   torch.clamp(lower_corner - points, min=0.0)
    dist_inside = centres - torch.min(upper_corner, torch.max(lower_corner, points))
    return dist_inside, dist_outside


def polynomial_loss(points, lower_corner, upper_corner, scale_mults):
    widths = upper_corner - lower_corner
    widths_p1 = widths + 1.0
    centres = 0.5 * (lower_corner + upper_corner)
    condition = torch.logical_and(lower_corner <= points, points <= upper_corner)
    expr_inside = torch.abs(points - centres) / widths_p1
    expr_outside = widths_p1 * torch.abs(points - centres) - (widths / 2.0) * (widths_p1 - 1.0 / widths_p1)
    return expr_inside, expr_outside, condition


def total_box_size_reg(rel_deltas, reg_lambda, log_box_size):
    rel_mean_width = torch.mean(torch.log(torch.abs(rel_deltas) + SANITY_EPS), dim=1)
    min_width = torch.min(rel_mean_width).detach()
    rel_width_ratios = torch.exp(rel_mean_width - min_width)
    total_multiplier = torch.log(torch.sum(rel_width_ratios) + SANITY_EPS)
    total_width = total_multiplier + min_width
    size_constraint_loss = reg_lambda * (total_width - log_box_size) ** 2
    return size_constraint_loss


def drpt(tensor, rate):
    if rate > 0:
        return F.dropout(tensor, p=rate, training=True)
    else:
        return tensor


def loss_function_q2b(batch_points, batch_mask, box_lows, box_highs, box_mults, typicality_scores,
                      dim_dropout_prob=0.0, order=2, alpha=0.2):




    batch_points = batch_points.squeeze(1) if batch_points.dim() > 2 else batch_points
    box_lows = box_lows.squeeze(1) if box_lows.dim() > 2 else box_lows
    box_highs = box_highs.squeeze(1) if box_highs.dim() > 2 else box_highs

    batch_mask = batch_mask.reshape(-1)

    batch_box_inside, batch_box_outside = q2b_loss(batch_points, box_lows, box_highs)

    if dim_dropout_prob > 0.0:
        batch_box_inside = drpt(batch_box_inside, rate=dim_dropout_prob)
        batch_box_outside = drpt(batch_box_outside, rate=dim_dropout_prob)

    bbi = torch.norm(batch_box_inside, p=order, dim=-1)
    bbo = torch.norm(batch_box_outside, p=order, dim=-1)

    bbi_masked = bbi * batch_mask
    bbo_masked = bbo * batch_mask

    weight_inside = torch.where(typicality_scores >= 0,
                                1 - typicality_scores,
                                torch.zeros_like(typicality_scores))

    weight_outside = torch.where(typicality_scores < 0,
                                 -typicality_scores,
                                 torch.zeros_like(typicality_scores))

    total_loss = alpha * (bbi_masked * weight_inside) + (bbo_masked * weight_outside)

    return total_loss


def loss_function_poly(batch_points, batch_mask, box_lows, box_highs, box_mults, typicality_scores,
                       dim_dropout_prob=0.0, order=1, lambda_reg=0.001):


    batch_points = batch_points.squeeze(1)
    box_lows = box_lows.squeeze(1)
    box_highs = box_highs.squeeze(1)

    expr_inside, expr_outside, condition = polynomial_loss(batch_points, box_lows, box_highs, box_mults)

    if dim_dropout_prob > 0.0:
        expr_inside = drpt(expr_inside, rate=dim_dropout_prob)
        expr_outside = drpt(expr_outside, rate=dim_dropout_prob)

    loss_inside = torch.norm(expr_inside, p=order, dim=-1)
    loss_outside = torch.norm(expr_outside, p=order, dim=-1)

    loss_inside_masked = loss_inside * batch_mask
    loss_outside_masked = loss_outside * batch_mask

    weight_inside = torch.where(
        typicality_scores >= 0,
        1 - typicality_scores,
        torch.zeros_like(typicality_scores)
    )
    weight_outside = torch.where(
        typicality_scores < 0,
        -typicality_scores,
        torch.zeros_like(typicality_scores)
    )

    weight_inside = torch.clamp(weight_inside, min=0)
    weight_outside = torch.clamp(weight_outside, min=0)

    alpha_inside = 1.0
    alpha_outside = 100.0

    total_loss_per_example = (alpha_inside * loss_inside_masked * weight_inside) + \
                             (alpha_outside * loss_outside_masked * weight_outside)




    return total_loss_per_example.mean(dim=-1)

def loss_function_t(batch_points, batch_mask, box_lows, box_highs, box_mults, typicality_scores,
                    dim_dropout_prob=0.0, order=1,
                    min_box_size=0.01,
                    box_collapse_weight=10.0,
                    box_volume_weight=0.1,
                    typicality_loss_weight=20.0,
                    penalty_weight_pos=1.0,
                    penalty_weight_neg=50.0,
                    loss_scale=1.0,
                    log_interval=100,
                    batch_counter=None):

    entity_embeds = batch_points.squeeze(1) if batch_points.dim() > 2 else batch_points
    concept_lowers = box_lows.squeeze(1) if box_lows.dim() > 2 else box_lows
    concept_uppers = box_highs.squeeze(1) if box_highs.dim() > 2 else box_highs

    box_sizes = torch.clamp(concept_uppers - concept_lowers, min=SANITY_EPS)
    loss_box_collapse = box_collapse_weight * torch.sum(torch.relu(min_box_size - box_sizes), dim=-1)
    loss_box_volume = box_volume_weight * torch.sum(box_sizes, dim=-1)
    box_regularization_energy = loss_box_collapse + loss_box_volume

    violation_upper = torch.relu(entity_embeds - concept_uppers)
    violation_lower = torch.relu(concept_lowers - entity_embeds)
    violation = violation_upper + violation_lower
    energy_positive_containment = penalty_weight_pos * torch.sum(violation ** 2, dim=-1)

    concept_centers = (concept_lowers + concept_uppers) / 2

    dist_to_center_scalar_for_log = torch.norm(entity_embeds - concept_centers, p=order, dim=-1)

    typ_clamped = torch.clamp(typicality_scores, 0, 1)


    dist_to_center_dims = torch.abs(entity_embeds - concept_centers)

    box_half_widths = box_sizes / 2.0

    target_max_dist_from_center_ratio_unsqueezed = (1.0 - typ_clamped).unsqueeze(-1)

    target_max_allowed_dist_from_center_dims = target_max_dist_from_center_ratio_unsqueezed * (box_half_widths + SANITY_EPS)


    typicality_distance_penalty_dims = (dist_to_center_dims - target_max_allowed_dist_from_center_dims)**2

    energy_positive_typicality = typicality_loss_weight * torch.norm(typicality_distance_penalty_dims, p=order, dim=-1)

    d_lower_neg = entity_embeds - concept_lowers
    d_upper_neg = concept_uppers - entity_embeds
    dist_from_boundary_towards_center_per_dim = torch.min(d_lower_neg, d_upper_neg)
    penetration_depth_per_dim = torch.clamp(dist_from_boundary_towards_center_per_dim, min=0)
    energy_negative_penalty = penalty_weight_neg * torch.sum(penetration_depth_per_dim, dim=-1)

    positive_mask = (typicality_scores >= 0)
    negative_mask = ~positive_mask

    total_energy_for_positives = energy_positive_containment + \
                                 energy_positive_typicality + \
                                 box_regularization_energy

    total_energy_for_negatives = energy_negative_penalty

    final_energy = torch.where(positive_mask, total_energy_for_positives, total_energy_for_negatives)
    final_energy = final_energy * batch_mask.squeeze()

    if batch_counter is not None and batch_counter % log_interval == 0 and False:
        print(f"--- Loss Function Internal Log (Batch {batch_counter}) ---")

        if positive_mask.any():

            pos_typicality_scores = typicality_scores[positive_mask]

            pos_dist_to_center_scalar = dist_to_center_scalar_for_log[positive_mask]
            pos_one_minus_typ = (1 - torch.clamp(pos_typicality_scores, 0, 1))

            print(f"    Avg raw positive typicality_scores: {pos_typicality_scores.float().mean().item():.4f} (Min: {pos_typicality_scores.min().item():.4f}, Max: {pos_typicality_scores.max().item():.4f})")
            print(f"    Avg (1 - clamped_pos_typ): {pos_one_minus_typ.float().mean().item():.4f}")
            print(f"    Avg pos_dist_to_center (scalar norm): {pos_dist_to_center_scalar.mean().item():.4f} (Min: {pos_dist_to_center_scalar.min().item():.4f}, Max: {pos_dist_to_center_scalar.max().item():.4f})")


            print(f"  Avg Pos Containment Energy: {energy_positive_containment[positive_mask].mean().item():.4f}")

            print(f"  Avg Pos Typicality Energy (final for loss): {energy_positive_typicality[positive_mask].mean().item():.4f}")
            print(f"  Avg Box Collapse Energy (for pos): {loss_box_collapse[positive_mask].mean().item():.4f}")
            print(f"  Avg Box Volume Energy (for pos): {loss_box_volume[positive_mask].mean().item():.4f}")
            print(f"  Avg Total Energy (Positives): {total_energy_for_positives[positive_mask].mean().item():.4f}")

        if negative_mask.any():
            print(f"  Avg Neg Penalty Energy: {energy_negative_penalty[negative_mask].mean().item():.4f}")
            print(f"  Avg Total Energy (Negatives): {total_energy_for_negatives[negative_mask].mean().item():.4f}")
        print(f"  Avg Final Energy (Overall): {final_energy.mean().item():.4f}")
        print(f"-------------------------------------------------")

    return final_energy

def loss_function_poly_v2(batch_points, batch_mask, box_lows, box_highs, box_mults, typicality_scores,
                          dim_dropout_prob=0.0, order=1, batch_counter=None):
    """
    Polynomial-based loss with refined typicality scoring for positive examples
    and distinct penalties for negative examples. Alphas are dynamic.
    """
    batch_points = batch_points.squeeze(1)
    box_lows = box_lows.squeeze(1)
    box_highs = box_highs.squeeze(1)

    widths = box_highs - box_lows
    centers = 0.5 * (box_lows + box_highs)

    avg_box_width_per_example = torch.mean(torch.abs(widths) + SANITY_EPS, dim=-1)

    dynamic_alpha_pos_containment = 10.0
    dynamic_alpha_pos_typicality = 5.0 / avg_box_width_per_example
    dynamic_alpha_neg_outside = 100.0
    dynamic_alpha_neg_inside_penalty = 20.0 / avg_box_width_per_example



    condition_poly_inside_dim = torch.logical_and(batch_points >= box_lows, batch_points <= box_highs)
    is_entity_completely_inside_box = condition_poly_inside_dim.all(dim=-1)

    expr_inside_poly, expr_outside_poly, _ = polynomial_loss(batch_points, box_lows, box_highs, box_mults)

    if dim_dropout_prob > 0.0:
        expr_inside_poly = drpt(expr_inside_poly, rate=dim_dropout_prob)
        expr_outside_poly = drpt(expr_outside_poly, rate=dim_dropout_prob)

    loss_poly_outside = torch.norm(expr_outside_poly, p=order, dim=-1)

    total_loss_per_example = torch.zeros_like(typicality_scores, device=batch_points.device)

    positive_mask = (typicality_scores >= 0)

    dist_outside_positive_dims = torch.clamp(batch_points - box_highs, min=0.0) + \
                                 torch.clamp(box_lows - batch_points, min=0.0)
    loss_containment_positive = torch.norm(dist_outside_positive_dims, p=order, dim=-1)

    current_positive_loss = dynamic_alpha_pos_containment * loss_containment_positive

    dist_to_center_dims = torch.abs(batch_points - centers)
    clamped_typicality = torch.clamp(typicality_scores, 0.0, 1.0)
    target_max_dist_from_center_ratio = (1.0 - clamped_typicality).unsqueeze(-1)
    target_max_dist_from_center_dims = target_max_dist_from_center_ratio * (widths / 2.0 + SANITY_EPS)
    typicality_distance_penalty_dims = torch.relu(dist_to_center_dims - target_max_dist_from_center_dims)
    loss_typicality_positive = torch.norm(typicality_distance_penalty_dims, p=order, dim=-1)

    current_positive_loss = torch.where(
        is_entity_completely_inside_box,
        dynamic_alpha_pos_typicality * loss_typicality_positive,
        current_positive_loss
    )

    total_loss_per_example = torch.where(
        positive_mask,
        current_positive_loss,
        total_loss_per_example
    )

    negative_mask = (typicality_scores < 0)

    dist_inside_negative_dims = centers - torch.min(box_highs, torch.max(box_lows, batch_points))
    dist_inside_negative_dims = torch.clamp(dist_inside_negative_dims, min=0.0)
    norm_dist_inside_negative = torch.norm(dist_inside_negative_dims, p=order, dim=-1)
    penalty_negative_inside = dynamic_alpha_neg_inside_penalty * norm_dist_inside_negative

    penalty_negative_outside = dynamic_alpha_neg_outside * loss_poly_outside

    current_negative_loss = torch.where(
        is_entity_completely_inside_box,
        penalty_negative_inside,
        penalty_negative_outside
    )

    total_loss_per_example = torch.where(
        negative_mask,
        current_negative_loss,
        total_loss_per_example
    )

    total_loss_per_example = total_loss_per_example * batch_mask.squeeze(-1)

    return total_loss_per_example


def compute_box(box_base, box_delta):
    box_delta = torch.abs(box_delta)
    box_second = box_base + half * box_delta
    box_first = box_base - half * box_delta
    box_low = torch.min(box_first, box_second)
    box_high = torch.max(box_first, box_second)
    return box_low, box_high


def compute_box_np(box_base, box_delta):
    box_second = box_base + 0.5 * box_delta
    box_first = box_base - 0.5 * box_delta
    box_low = np.minimum(box_first, box_second)
    box_high = np.maximum(box_first, box_second)
    return box_low, box_high

def uniform_neg_sampling(nb_neg_examples_per_pos, batch_components,
                         nb_entities, nb_concepts, return_replacements=False,
                         entity_id_to_name_map=None, concept_id_to_name_map=None):
    global uniform_neg_sampling_call_count

    if nb_neg_examples_per_pos == 0:
        if return_replacements:
            return batch_components, None
        else:
            return batch_components

    batch_size = batch_components.shape[0]

    entity_pos = 0
    concept_pos = 2
    val_pos = 3

    random_count = batch_size * nb_neg_examples_per_pos

    device = batch_components.device
    replace_entity = torch.randint(0, 2, (random_count,), device=device)

    negative_samples_original_repeated = batch_components.repeat(nb_neg_examples_per_pos, 1)
    negative_samples = negative_samples_original_repeated.clone()
    indices_to_replace = torch.arange(random_count, device=device)

    replacements_list = []

    if uniform_neg_sampling_call_count < 0:
        print(f"\n--- uniform_neg_sampling LOG (Call")



        print(f"  Config: nb_neg_per_pos={nb_neg_examples_per_pos}, batch_size={batch_size}, total_neg_to_gen={random_count}")

        if batch_components.numel() > 0:
            print("  Original Batch (first sample):")
            orig_fact = batch_components[0]
            ent_id = orig_fact[entity_pos].item()
            rel_id = orig_fact[1].item()
            con_id = orig_fact[concept_pos].item()
            typ_val = orig_fact[val_pos].item()
            ent_name = entity_id_to_name_map.get(ent_id, f"ID:{ent_id}") if entity_id_to_name_map else f"ID:{ent_id}"
            con_name = concept_id_to_name_map.get(con_id, f"ID:{con_id}") if concept_id_to_name_map else f"ID:{con_id}"
            print(f"    Fact: (Entity: '{ent_name}', Rel_ID: {rel_id}, Concept: '{con_name}', Typ: {typ_val})")
        else:
            print("  Original Batch: EMPTY")


    entity_replacement_indices = indices_to_replace[replace_entity == 0]
    if entity_replacement_indices.numel() > 0:
        new_entity_ids = torch.randint(0, nb_entities, (entity_replacement_indices.numel(),), device=device)

        original_entity_values_for_neg = negative_samples_original_repeated[entity_replacement_indices, entity_pos]

        if uniform_neg_sampling_call_count < 0 and entity_replacement_indices.numel() > 0:
            log_limit = min(2, entity_replacement_indices.numel())
            print(f"  Entity Replacements (showing first {log_limit} of {entity_replacement_indices.numel()} actual replacements):")
            for i in range(log_limit):



                idx_in_repeated_batch = entity_replacement_indices[i].item()
                original_fact_context = negative_samples_original_repeated[idx_in_repeated_batch]

                orig_ent_id_val = original_entity_values_for_neg[i].item()
                new_ent_id_val = new_entity_ids[i].item()











                orig_ent_name = entity_id_to_name_map.get(orig_ent_id_val, f"ID:{orig_ent_id_val}") if entity_id_to_name_map else f"ID:{orig_ent_id_val}"
                new_ent_name = entity_id_to_name_map.get(new_ent_id_val, f"ID:{new_ent_id_val}") if entity_id_to_name_map else f"ID:{new_ent_id_val}"

                orig_rel_id_context = original_fact_context[1].item()
                orig_conc_id_context = original_fact_context[concept_pos].item()
                orig_conc_name_context = concept_id_to_name_map.get(orig_conc_id_context, f"ID:{orig_conc_id_context}") if concept_id_to_name_map else f"ID:{orig_conc_id_context}"

                print(f"    Sample")
                print(f"      Original Fact Context: (Entity: '{orig_ent_name}', Rel_ID: {orig_rel_id_context}, Concept: '{orig_conc_name_context}')")
                print(f"      Entity Corrupted: FROM '{orig_ent_name}' (ID:{orig_ent_id_val}) TO '{new_ent_name}' (ID:{new_ent_id_val})")


        negative_samples[entity_replacement_indices, entity_pos] = new_entity_ids
        entity_replacements = torch.stack([
            batch_size + entity_replacement_indices,
            torch.full_like(entity_replacement_indices, entity_pos, device=device),
            original_entity_values_for_neg,
            new_entity_ids
        ], dim=1)
        replacements_list.append(entity_replacements)

    concept_replacement_indices = indices_to_replace[replace_entity == 1]
    if concept_replacement_indices.numel() > 0:
        new_concept_ids = torch.randint(0, nb_concepts, (concept_replacement_indices.numel(),), device=device)
        original_concept_values = negative_samples[concept_replacement_indices, concept_pos].clone()
        negative_samples[concept_replacement_indices, concept_pos] = new_concept_ids
        concept_replacements = torch.stack([
            batch_size + concept_replacement_indices,
            torch.full_like(concept_replacement_indices, concept_pos, device=device),
            original_concept_values,
            new_concept_ids
        ], dim=1)
        replacements_list.append(concept_replacements)

    negative_samples[:, val_pos] = -1

    batch_components_neg = torch.cat([batch_components, negative_samples], dim=0)

    if return_replacements:
        if replacements_list:
            replacements = torch.cat(replacements_list, dim=0)
        else:
            replacements = None
        return batch_components_neg, replacements
    else:
        return batch_components_neg


def create_uniform_var(name, shape, min_val, max_val):
    var = nn.Parameter(torch.empty(shape, device=device).uniform_(min_val, max_val))
    return var

def create_normal_var(name, shape):


    std = 1.0 / np.sqrt(shape[1])
    var = nn.Parameter(torch.empty(shape, device=device).normal_(0, std))
    return var

def instantiate_box_embeddings(name: str, scale_mult_shape, rel_tbl_shape, base_norm_shapes, sqrt_dim,
                               hard_size: bool, total_size: float, relation_stats, fixed_width: bool):
    with torch.no_grad():
        if relation_stats is not None:
            scale_multiples = relation_stats.detach()
        else:
            if fixed_width:
                scale_multiples = torch.ones(scale_mult_shape, device=device)
            else:
                scale_multiples = create_uniform_var("scale_multiples_" + name, scale_mult_shape, -1.0, 1.0)

            if hard_size:
                scale_multiples = total_size * F.softmax(scale_multiples, dim=0)
            else:
                scale_multiples = F.softplus(scale_multiples)

        base_norm_shapes = torch.abs(base_norm_shapes.detach() if base_norm_shapes.requires_grad else base_norm_shapes)

        embedding_base_points = create_uniform_var(name + "_base_point", rel_tbl_shape, -0.5 / sqrt_dim,
                                                   0.5 / sqrt_dim)

        embedding_deltas = torch.abs(scale_multiples) * torch.abs(base_norm_shapes)

    embedding_base_points = nn.Parameter(embedding_base_points)
    embedding_deltas = nn.Parameter(embedding_deltas)
    scale_multiples = nn.Parameter(scale_multiples)

    return embedding_base_points, embedding_deltas, scale_multiples


def corrupt_batch(batch, idx, nb_entities, hash_table, filtered=True):
    nb_batch_facts = batch.shape[0]
    replacement_ents = torch.full((nb_batch_facts,), int(idx % nb_entities), dtype=torch.long, device=device)
    rep_ar_pos = torch.full((nb_batch_facts,), int(idx // nb_entities), dtype=torch.long, device=device)
    replacement_idx = torch.stack([torch.arange(nb_batch_facts, device=device), rep_ar_pos + 1], dim=1)
    replacement_mask = torch.zeros_like(batch, dtype=torch.long)
    replacement_mask[replacement_idx[:, 0], replacement_idx[:, 1]] = replacement_ents + 1
    new_batch = torch.where(replacement_mask > 0, replacement_mask - 1, batch)
    if filtered:
        input_keys = ['{}{}'.format(cnst.FACT_DELIMITER.join(map(str, row[:-1].tolist()))) for row in new_batch]
        fact_exists = hash_table.lookup(input_keys)
        fact_exists_bool = fact_exists > 0
        original_ents = batch[:, idx // nb_entities + 1]
        original_ents_filt = original_ents[fact_exists_bool]
        replacement_idx_filt = replacement_idx[fact_exists_bool]
        replacement_mask = torch.zeros_like(new_batch, dtype=torch.long)
        replacement_mask[replacement_idx_filt[:, 0], replacement_idx_filt[:, 1]] = original_ents_filt + 1
        new_batch = torch.where(replacement_mask > 0, replacement_mask - 1, new_batch)
    return new_batch

class BoxT(nn.Module):
    def __init__(self, kb_name, options: ModelOptions, suffix: str = ""):

        torch.autograd.set_detect_anomaly(True)
        print("Device used:", device)

        super(BoxT, self).__init__()

        self.options = options
        self.embedding_dim = options.embedding_dim
        self.neg_sampling_opt = options.neg_sampling_opt

        self.adv_temp = options.adversarial_temp
        self.nb_neg_examples_per_pos = options.nb_neg_examples_per_pos
        self.nb_neg = self.nb_neg_examples_per_pos
        self.learning_rate = options.learning_rate

        self.stop_gradient = options.stop_gradient
        self.replace_idx = options.replace_indices
        self.margin = options.loss_margin
        self.reg_lambda = options.regularisation_lambda
        self.reg_points = options.regularisation_points
        self.total_log_box_size = options.total_log_box_size
        self.batch_size = options.batch_size

        self.hard_total_size = options.hard_total_size

        self.shared_shape = options.shared_shape
        self.learnable_shape = options.learnable_shape
        self.fixed_width = options.fixed_width

        self.param_directory = "weights/" + kb_name + "/values.ckpt"

        self.bounded_pt_space = options.bounded_pt_space

        self.bounded_box_space = options.bounded_box_space

        self.bound_scale = options.space_bound

        self.obj_fct = options.obj_fct
        self.loss_fct = options.loss_fct
        self.loss_ord = options.loss_norm_ord
        self.dim_dropout_prob = options.dim_dropout_prob
        self.dim_dropout_flt = options.dim_dropout_prob

        self.viz_mode = options.viz
        self.save_pos = options.pos

        self.kb_name = kb_name
        kb_metadata = utils.data_prep.load_kb_metadata_multi(kb_name)
        self.nb_entities = kb_metadata[0]

        self.nb_relations = kb_metadata[1]

        self.nb_concepts = kb_metadata[2]

        self.hard_code_size = options.hard_code_size
        self.sqrt_dim = np.sqrt(self.embedding_dim)

        self.gradient_clip = options.gradient_clip

        self.bounded_norm = options.bounded_norm

        self.max_arity = kb_metadata[3]

        self.augment_inv = options.augment_inv

        self.original_nb_rel = self.nb_concepts

        if self.augment_inv:
            if self.max_arity > 2:
                print("Unable to use data augmentation, dataset is not a knowledge graph. Setting Aug to False")
                self.augment_inv = False
            else:
                self.nb_concepts = 2 * self.nb_concepts

        if options.hard_code_size:
            relation_stats = utils.data_prep.compute_statistics(kb_name)
            relation_stats = relation_stats ** (1 / self.embedding_dim)
        else:
            relation_stats = None

        self.lr_decay = options.learning_rate_decay
        self.lr_decay_period = options.decay_period

        concept_tbl_shape = [self.nb_concepts, self.embedding_dim]
        scale_multiples_shape = [self.nb_concepts, 1]


        self.total_size = np.exp(options.total_log_box_size) if self.hard_total_size else -1

        self.entity_points = nn.Embedding(self.nb_entities, self.embedding_dim)
        self.entity_points.to(device)

        self.concept_bases = nn.Embedding(self.nb_concepts, self.embedding_dim)
        nn.init.uniform_(self.concept_bases.weight, -0.5 / self.sqrt_dim, 0.5 / self.sqrt_dim)
        self.concept_bases.to(device)

        self.concept_deltas_raw = nn.Embedding(self.nb_concepts, self.embedding_dim)

        nn.init.uniform_(self.concept_deltas_raw.weight, -1.0, 1.0)
        self.concept_deltas_raw.to(device)

        self.concept_multiples = nn.Embedding(self.nb_concepts, 1)

        nn.init.ones_(self.concept_multiples.weight)
        self.concept_multiples.to(device)

        self.total_size = np.exp(options.total_log_box_size) if self.hard_total_size else -1

        self.softplus = nn.Softplus()

        tr_np_arr = utils.data_prep.load_kb_file(cnst.DEFAULT_KB_MULTI_DIR + str(kb_name) + "/train" + cnst.KB_FORMAT)
        if not options.restricted_training:
            self.nb_training_facts = tr_np_arr.shape[0]
        else:
            tr_np_arr = tr_np_arr[:options.restriction, :]
            self.nb_training_facts = options.restriction
        if self.augment_inv:
            tr_np_arr_augmentation = np.zeros_like(tr_np_arr)
            tr_np_arr_augmentation[:, 0] = tr_np_arr[:, 0] + self.original_nb_rel
            tr_np_arr_augmentation[:, 1] = tr_np_arr[:, 2]
            tr_np_arr_augmentation[:, 2] = tr_np_arr[:, 1]
            tr_np_arr_augmentation[:, 3] = tr_np_arr[:, 3]
            tr_np_arr = np.concatenate([tr_np_arr, tr_np_arr_augmentation], axis=0)
            self.nb_training_facts = 2 * self.nb_training_facts

        self.nb_tr_batches = ceil(self.nb_training_facts / self.batch_size)
        tr_tensor = torch.tensor(tr_np_arr, dtype=torch.long, device=device)
        self.tr_dataset = torch.utils.data.TensorDataset(tr_tensor)

        if self.neg_sampling_opt == cnst.UNIFORM or self.neg_sampling_opt == cnst.SELFADV:
            def collate_fn(facts):
                facts = torch.stack([f[0] for f in facts])
                if self.replace_idx and self.nb_neg_examples_per_pos > 0:
                    batch_components, replaced_indices = uniform_neg_sampling(
                        self.nb_neg_examples_per_pos,
                        batch_components=facts,
                        nb_entities=self.nb_entities,
                        nb_concepts=self.nb_concepts,
                        return_replacements=True,
                        entity_id_to_name_map=self.entity_id_to_name,
                        concept_id_to_name_map=self.concept_id_to_name
                    )
                    return batch_components, replaced_indices
                else:
                    batch_components = uniform_neg_sampling(
                        self.nb_neg_examples_per_pos,
                        batch_components=facts,
                        nb_entities=self.nb_entities,
                        nb_concepts=self.nb_concepts,
                        entity_id_to_name_map=self.entity_id_to_name,
                        concept_id_to_name_map=self.concept_id_to_name
                    )
                    replaced_indices = None
                    return batch_components, replaced_indices

            self.tr_loader = torch.utils.data.DataLoader(self.tr_dataset, batch_size=self.batch_size,
                                                         shuffle=True, collate_fn=collate_fn)
        else:
            self.tr_loader = torch.utils.data.DataLoader(self.tr_dataset, batch_size=self.batch_size,
                                                         shuffle=True)

        self.hash_tbl = utils.data_prep.create_kb_filter_torch(self.kb_name)

        vl_np_arr = utils.data_prep.load_kb_file(cnst.DEFAULT_KB_MULTI_DIR + str(kb_name) + "/valid" + cnst.KB_FORMAT)
        self.nb_vl_facts = vl_np_arr.shape[0]
        vl_tensor = torch.tensor(vl_np_arr, dtype=torch.long, device=device)
        self.vl_dataset = torch.utils.data.TensorDataset(vl_tensor)
        self.vl_loader = torch.utils.data.DataLoader(self.vl_dataset, batch_size=self.nb_vl_facts)

        self.vl_corr_loader = self.generate_corrupted_loader(vl_np_arr)

        tr_ts_np_arr = tr_np_arr[:3 * self.batch_size, :]
        self.nb_tr_ts_facts = tr_ts_np_arr.shape[0]
        tr_ts_tensor = torch.tensor(tr_ts_np_arr, dtype=torch.long, device=device)
        self.tr_ts_dataset = torch.utils.data.TensorDataset(tr_ts_tensor)
        self.tr_ts_loader = torch.utils.data.DataLoader(self.tr_ts_dataset, batch_size=self.nb_tr_ts_facts)

        self.tr_ts_corr_loader = self.generate_corrupted_loader(tr_ts_np_arr)

        ts_np_arr = utils.data_prep.load_kb_file(cnst.DEFAULT_KB_MULTI_DIR + str(kb_name) + "/test" + cnst.KB_FORMAT)
        self.nb_ts_facts = ts_np_arr.shape[0]
        ts_tensor = torch.tensor(ts_np_arr, dtype=torch.long, device=device)
        self.ts_dataset = torch.utils.data.TensorDataset(ts_tensor)
        self.ts_loader = torch.utils.data.DataLoader(self.ts_dataset, batch_size=self.nb_ts_facts)

        self.ts_corr_loader = self.generate_corrupted_loader(ts_np_arr)

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        if self.lr_decay > 0:
            decay_step = self.lr_decay_period * self.nb_tr_batches
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda epoch: 1 / (1 + self.lr_decay * (epoch / decay_step)))
        else:
            self.lr_scheduler = None

        self.batch_components = None
        self.replaced_indices = None
        self.original_batch_size = None

        self.verbose = options.verbose

        self.typicality_threshold = options.typ_thresh

        self.load_kb_dicts(kb_name, cnst.DEFAULT_KB_MULTI_DIR)

        self.now_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

        self.build_known_facts()

        self.generate_report = options.generate_report
        if self.generate_report:
            summary_descriptor = (
                f"{self.kb_name}_{self.stop_gradient}_{self.learning_rate}_nb_neg-{self.nb_neg_examples_per_pos}"
                f"_loss_margin-{self.margin}_emb_dim-{self.embedding_dim}_neg_opt-{self.neg_sampling_opt}"
                f"_{self.now_str}")
            self.summary_dir = os.path.join('summaries/', summary_descriptor)
            self.summary_writer = SummaryWriter(log_dir=self.summary_dir)

        dir_name = f"{self.kb_name}_lr-{self.learning_rate}_emb_dim-{self.embedding_dim}_{self.now_str}"
        if self.save_pos:
            self.positions_dir = os.path.join('positions/', dir_name)
            os.makedirs(self.positions_dir, exist_ok=True)
        if self.viz_mode:
            self.viz_dir = os.path.join('plots/', dir_name)
            os.makedirs(self.viz_dir, exist_ok=True)

        self.hits_at = [1, 3, 5, 10]

        self.parent_of = {}
        self.children_of = defaultdict(set)
        self._load_concept_hierarchy()

        self.noun_to_base_concept_id = {}
        self._load_noun_concept_mapping()

        self.current_batch_count = 0

        self.criterion = nn.BCEWithLogitsLoss()

        self.to(device)

    def _load_noun_concept_mapping(self):
        """
        Loads noun_concept_mapping.json for the current KB.
        """
        mapping_path = os.path.join(cnst.DEFAULT_KB_DIR, self.kb_name, "noun_concept_mapping.json")

        if not os.path.exists(mapping_path):
            mapping_path_multi = os.path.join(cnst.DEFAULT_KB_MULTI_DIR, self.kb_name, "noun_concept_mapping.json")
            if os.path.exists(mapping_path_multi):
                mapping_path = mapping_path_multi
            else:
                mapping_path_datagen = os.path.join("DataGen", "noun_concept_mapping.json")
                if os.path.exists(mapping_path_datagen):
                    mapping_path = mapping_path_datagen
                else:
                    return

        if not hasattr(self, 'c2id_dict') or not self.c2id_dict:
            return

        try:
            with open(mapping_path, 'r') as f:
                noun_to_concept_name = json.load(f)

            for noun, concept_name in noun_to_concept_name.items():
                if concept_name in self.c2id_dict:
                    self.noun_to_base_concept_id[noun] = self.c2id_dict[concept_name]
                else:

                    pass

            if not self.noun_to_base_concept_id:
                print(f"Warning: Noun to base concept mapping for {self.kb_name} is empty after processing.", flush=True)

        except Exception as e:
            print(f"Warning: An unexpected error occurred while loading noun_concept_mapping for KB '{self.kb_name}: {e}'", flush=True)


    def _get_noun_from_entity_id(self, entity_id: int) -> Optional[str]:
        if not hasattr(self, 'entity_id_to_name'):
            return None
        entity_name = self.entity_id_to_name.get(entity_id)
        if entity_name:
            parts = entity_name.split('_')
            if parts:
                return parts[-1]

        return None

    def _get_base_concept_id_for_noun(self, noun_str: str) -> Optional[int]:
        if not hasattr(self, 'noun_to_base_concept_id'):

            return None
        return self.noun_to_base_concept_id.get(noun_str)

    def _is_ancestor(self, potential_ancestor_id: int, concept_id: int) -> bool:
        if potential_ancestor_id == concept_id:
            return True

        if not hasattr(self, 'parent_of'):
            return False

        current_concept_id = concept_id

        max_depth = getattr(self, 'nb_concepts', 100)

        for _ in range(max_depth):
            parent_id = self.parent_of.get(current_concept_id)
            if parent_id is None:
                return False
            if parent_id == potential_ancestor_id:
                return True
            current_concept_id = parent_id

        return False

    def _load_concept_hierarchy(self):
        """
        Loads concept hierarchy (parent-child relationships) from concept_relations.json.
        """
        concept_relations_path = os.path.join(cnst.DEFAULT_KB_DIR, self.kb_name, "concept_relations.json")

        if not os.path.exists(concept_relations_path):
            concept_relations_path_multi = os.path.join(cnst.DEFAULT_KB_MULTI_DIR, self.kb_name, "concept_relations.json")
            if os.path.exists(concept_relations_path_multi):
                concept_relations_path = concept_relations_path_multi
            else:
                concept_relations_path_datagen = os.path.join("DataGen", "concept_relations.json")
                if os.path.exists(concept_relations_path_datagen):
                    print(f"Warning: Using concept_relations.json from DataGen/ for {self.kb_name}.", flush=True)
                    concept_relations_path = concept_relations_path_datagen


        if not hasattr(self, 'c2id_dict') or not self.c2id_dict:
            print(f"Cannot load concept hierarchy for {self.kb_name}.", flush=True)
            return

        try:
            with open(concept_relations_path, 'r') as f:
                relations_by_name = json.load(f)

            for concept_name, details in relations_by_name.items():
                if concept_name not in self.c2id_dict:

                    continue
                child_id = self.c2id_dict[concept_name]

                parent_name = details.get("parent")
                if parent_name:
                    if parent_name not in self.c2id_dict:

                        continue
                    parent_id = self.c2id_dict[parent_name]
                    self.parent_of[child_id] = parent_id
                    self.children_of[parent_id].add(child_id)
        except Exception as e:
            print(f"Cannot open file {concept_relations_path}. Reason: {e}", flush=True)

    def load_kb_dicts(self, kb_name, kb_multi_dir):
        kb_directory = os.path.join(kb_multi_dir, kb_name)

        path_to_e2id = os.path.join(kb_directory, "Ent2ID.dict")
        path_to_r2id = os.path.join(kb_directory, "Rel2ID.dict")
        path_to_c2id = os.path.join(kb_directory, "Con2ID.dict")

        with open(path_to_e2id, 'rb') as f:
            self.e2id_dict = msgpack.unpack(f, raw=False)
        with open(path_to_r2id, 'rb') as f:
            self.r2id_dict = msgpack.unpack(f, raw=False)
        with open(path_to_c2id, 'rb') as f:
            self.c2id_dict = msgpack.unpack(f, raw=False)

        self.entity_id_to_name = {v: k for k, v in self.e2id_dict.items()}
        self.concept_id_to_name = {v: k for k, v in self.c2id_dict.items()}
        self.relation_id_to_name = {v: k for k, v in self.r2id_dict.items()}

    def generate_corrupted_loader(self, np_arr):
        def generate_corrupted_data():
            for idx in range(self.max_arity * self.nb_entities):
                batch = torch.tensor(np_arr, dtype=torch.long, device=device)
                corrupted_batch = corrupt_batch(batch, idx, self.nb_entities, self.hash_tbl)
                yield corrupted_batch
        return generate_corrupted_data()

    def build_known_facts(self):
        self.known_facts = set()

        for dataset_name in ["train", "valid", "test"]:
            path = f"{cnst.DEFAULT_KB_MULTI_DIR}{self.kb_name}/{dataset_name}{cnst.KB_FORMAT}"
            try:
                data = utils.data_prep.load_kb_file(path)
                for row in data:
                    self.known_facts.add((int(row[0]), int(row[1]), int(row[2])))
            except:
                print(f"Could not load {path}")

    def _loss_computation(
        self, batch_points, batch_mask, batch_concept_bases, batch_concept_deltas, batch_concept_mults,
        typicality_scores_raw,
        batch_counter=None
    ):

        obs = self.original_batch_size if self.original_batch_size is not None \
                                  else batch_points.size(0)

        if self.loss_fct == cnst.POLY_V2_LOSS:
            loss_function = loss_function_poly_v2
        elif self.loss_fct == cnst.RECONSTRUCTION_LOSS:
            loss_function = self.loss_function_reconstruction
        elif self.loss_fct == cnst.SIMPLE_LOSS:
            loss_function = loss_function_simple_geometric_distance
        else:
            warnings.warn(f"Warning: Unrecognized loss_fct '{self.loss_fct}'. Defaulting to loss_function_t.")
            loss_function = loss_function_t

        concept_bx_low, concept_bx_high = compute_box(batch_concept_bases, batch_concept_deltas)

        if self.bounded_pt_space:
            batch_points = self.bound_scale * torch.tanh(batch_points)

        if self.bounded_box_space:
            concept_bx_low = self.bound_scale * torch.tanh(concept_bx_low)
            concept_bx_high = self.bound_scale * torch.tanh(concept_bx_high)

        if self.dim_dropout_prob > 0.0:

            dropout_mask = (torch.rand_like(batch_points) > self.dim_dropout_prob).float()
            batch_points = batch_points * dropout_mask

        max_typicality_value_from_storage = 1000.0

        typicality_scores_raw_float = typicality_scores_raw.float()

        typicality_scores = torch.where(
            typicality_scores_raw_float >= 0,
            typicality_scores_raw_float / max_typicality_value_from_storage,
            typicality_scores_raw_float
        )

        if self.stop_gradient == cnst.NO_STOPS or obs is None:
            loss_pos_neg = loss_function(
                batch_points=batch_points,
                batch_mask=batch_mask,
                box_lows=concept_bx_low,
                box_highs=concept_bx_high,
                box_mults=batch_concept_mults,
                typicality_scores=typicality_scores,
                order=self.loss_ord,
                batch_counter=batch_counter
            )

            if obs < loss_pos_neg.shape[0]:
                pos_loss = loss_pos_neg[:obs]
                neg_loss = loss_pos_neg[obs:]
            else:
                pos_loss = loss_pos_neg
                neg_loss = torch.zeros(loss_pos_neg.numel() * self.nb_neg_examples_per_pos, device=loss_pos_neg.device)
        else:
            batch_points_pos = batch_points[:obs]
            batch_mask_pos = batch_mask[:obs]
            batch_points_neg = batch_points[obs:]
            batch_mask_neg = batch_mask[obs:]

            concept_bases_pos = batch_concept_bases[:obs]
            concept_deltas_pos = batch_concept_deltas[:obs]
            concept_mults_pos = batch_concept_mults[:obs]

            concept_bases_neg = batch_concept_bases[obs:]
            concept_deltas_neg = batch_concept_deltas[obs:]
            concept_mults_neg = batch_concept_mults[obs:]

            typicality_scores_pos = typicality_scores[:obs]
            typicality_scores_neg = typicality_scores[obs:]

            if self.stop_gradient[0] == cnst.STOP:
                concept_bases_neg = concept_bases_neg.detach()
            if self.stop_gradient[1] == cnst.STOP:
                concept_deltas_neg = concept_deltas_neg.detach()

            concept_bx_low_pos, concept_bx_high_pos = compute_box(concept_bases_pos, concept_deltas_pos)

            if self.bounded_pt_space:
                batch_points_pos = self.bound_scale * torch.tanh(batch_points_pos)
            if self.bounded_box_space:
                concept_bx_low_pos = self.bound_scale * torch.tanh(concept_bx_low_pos)
                concept_bx_high_pos = self.bound_scale * torch.tanh(concept_bx_high_pos)

            pos_loss = loss_function(
                batch_points=batch_points_pos,
                batch_mask=batch_mask_pos,
                box_lows=concept_bx_low_pos,
                box_highs=concept_bx_high_pos,
                box_mults=concept_mults_pos,
                typicality_scores=typicality_scores_pos,
                order=self.loss_ord,
                batch_counter=batch_counter
            )

            concept_bx_low_neg, concept_bx_high_neg = compute_box(concept_bases_neg, concept_deltas_neg)

            if self.bounded_pt_space:
                batch_points_neg = self.bound_scale * torch.tanh(batch_points_neg)
            if self.bounded_box_space:
                concept_bx_low_neg = self.bound_scale * torch.tanh(concept_bx_low_neg)
                concept_bx_high_neg = self.bound_scale * torch.tanh(concept_bx_high_neg)

            neg_loss = loss_function(
                batch_points=batch_points_neg,
                batch_mask=batch_mask_neg,
                box_lows=concept_bx_low_neg,
                box_highs=concept_bx_high_neg,
                box_mults=concept_mults_neg,
                typicality_scores=typicality_scores_neg,
                order=self.loss_ord,
                batch_counter=batch_counter
            )

        return pos_loss, neg_loss

    def forward(self):
        indices = self.batch_components[:, 0:self.max_arity - 1]
        batch_points = self.entity_points(indices[:, 0])

        self.batch_point_representations = batch_points

        batch_components_ent = self.batch_components[:, :1]
        raw_mask = torch.where(
            batch_components_ent == self.nb_entities,
            torch.zeros_like(batch_components_ent, dtype=torch.float32),
            torch.ones_like(batch_components_ent, dtype=torch.float32)
        )
        self.batch_mask = raw_mask.squeeze(-1)

        concept_indices = self.batch_components[:, 2].long()
        self.batch_concept_bases = self.concept_bases(concept_indices)
        self.batch_concept_deltas = F.softplus(self.concept_deltas_raw(concept_indices))

        batch_concept_mults = self.concept_multiples(concept_indices)

        positive_loss, negative_loss = self._loss_computation(
            batch_points=self.batch_point_representations,
            batch_mask=self.batch_mask,
            batch_concept_bases=self.batch_concept_bases,
            batch_concept_deltas=self.batch_concept_deltas,
            batch_concept_mults=batch_concept_mults,
            typicality_scores_raw=self.batch_components[:, 3],
            batch_counter=self.current_batch_count
        )

        computed_loss = torch.tensor(0.0, device=device)

        if self.loss_fct == cnst.SIMPLE_BOX_LOSS:
            computed_loss += positive_loss.sum() + (negative_loss.sum() / self.nb_neg_examples_per_pos)
            self.scores = positive_loss.unsqueeze(-1)

            concept_bx_low, concept_bx_high = compute_box(self.batch_concept_bases, self.batch_concept_deltas)

            entity_points = self.batch_point_representations[:self.original_batch_size]
            low_corners = concept_bx_low[:self.original_batch_size]
            high_corners = concept_bx_high[:self.original_batch_size]

            dist_outside_upper = torch.relu(entity_points - high_corners)
            dist_outside_lower = torch.relu(low_corners - entity_points)
            dist_outside = dist_outside_upper + dist_outside_lower
            total_distance = torch.sum(dist_outside, dim=-1)

            containment_score = torch.exp(-total_distance)

            self.scores = containment_score.unsqueeze(-1)

        if self.loss_fct == cnst.RECONSTRUCTION_LOSS:

            positive_loss = 0.1 * positive_loss.sum()
            negative_loss = self.options.lambda_neg_reconstruction * negative_loss.sum() /self.nb_neg_examples_per_pos


            with torch.no_grad():
                current_batch_points = self.batch_point_representations
                current_concept_bases = self.batch_concept_bases
                current_concept_deltas = self.batch_concept_deltas


                concept_lows = current_concept_bases - current_concept_deltas * 0.5
                concept_highs = current_concept_bases + current_concept_deltas * 0.5

                dist_outside_upper = torch.relu(current_batch_points - concept_highs)**2
                dist_outside_lower = torch.relu(concept_lows - current_batch_points)**2
                d_e_C = torch.sum(dist_outside_upper + dist_outside_lower, dim=-1)

                self.scores = torch.exp(-d_e_C).unsqueeze(-1)

        if self.loss_fct == cnst.POLY_V2_LOSS:

            with torch.no_grad():
                self.scores = -positive_loss.unsqueeze(-1)

            positive_loss = torch.log1p(positive_loss.sum())
            negative_loss = torch.log1p(negative_loss.sum() / self.nb_neg_examples_per_pos)

        concept_overlap_penalty = self.calculate_concept_box_dim_penalty(
            self.batch_components,
            self.batch_concept_bases,
            self.batch_concept_deltas,
            self.original_batch_size
        )

        monotonicity_penalty = self.box_size_monotonicity_penalty(
            self.batch_components,
            self.batch_concept_bases,
            self.batch_concept_deltas,
            self.original_batch_size,
            monotonicity_weight=1.0,
        )

        volume_reg_loss = total_box_size_regular(
            all_concept_deltas=F.softplus(self.concept_deltas_raw.weight),
            reg_lambda=self.reg_lambda,
            target_log_box_size=self.options.total_log_box_size
        )
        computed_loss = positive_loss + negative_loss + concept_overlap_penalty + monotonicity_penalty + volume_reg_loss

        if self.options.regularisation_points > 0:
            point_reg = self.options.regularisation_points * (
                    torch.nn.functional.mse_loss(self.batch_point_representations, torch.zeros_like(self.batch_point_representations), reduction='sum') +
                    torch.nn.functional.mse_loss(self.batch_concept_bases, torch.zeros_like(self.batch_concept_bases), reduction='sum')
            )
            computed_loss += point_reg

        if self.epoch % 100 == 0:
            print("epoch", self.epoch)
            print("computed_loss", computed_loss)
            print()
        return computed_loss

    def check_reg_loss(self, reload_params=False, param_loc=None):
        if reload_params:
            self.load_params(param_loc)
        scores = self.reg_loss.item()
        return scores

    def score_forward_pass(self, batch_components, reload_params=False, param_loc=None):
        if reload_params:
            self.load_params(param_loc)
        self.batch_components = batch_components
        self.original_batch_size = self.batch_components.shape[0]
        self.forward()
        scores = self.scores.detach().cpu()
        return scores

    def get_mask(self, batch_components, reload_params=False, param_loc=None):
        if reload_params:
            self.load_params(param_loc)
        self.batch_components = batch_components
        self.forward()
        mask = self.batch_mask.detach().cpu()
        return mask

    def compute_box_volume(self, reload_params=False, param_loc=None):
        if reload_params:
            self.load_params(param_loc)
        r_bases = self.concept_bases.detach().cpu().numpy()
        r_deltas = self.concept_deltas.detach().cpu().numpy()
        r_low, r_high = compute_box_np(r_bases, r_deltas)
        if self.bounded_box_space:
            r_low = self.bound_scale * np.tanh(r_low)
            r_high = self.bound_scale * np.tanh(r_high)
        r_log_widths = np.mean(np.log(r_high - r_low + SANITY_EPS), axis=-1)
        r_geom_width = np.exp(r_log_widths)
        return r_geom_width

    def load_params(self, param_loc=None):
        if param_loc is None:
            param_loc = self.param_directory
        if os.path.exists(param_loc):
            self.load_state_dict(torch.load(param_loc, map_location=device, weights_only=True))
            self.eval()
        else:
            print(f"No checkpoint found at '{param_loc}'. Starting from scratch.")

    def get_relation_average_norm_width(self):
        rel_batch = torch.arange(self.nb_concepts, device=device)
        batch_components = torch.zeros((self.nb_concepts, self.max_arity + 2), dtype=torch.long, device=device)
        batch_components[:, 0] = rel_batch
        self.batch_components = batch_components
        self.forward()
        concept_deltas = self.concept_deltas.detach().cpu().numpy()
        width_arith_mean = np.mean(np.mean(concept_deltas, axis=2), axis=1)
        return width_arith_mean

    def forward_pass(self, batch_components, reload_params=False, param_loc=None):
        if reload_params:
            self.load_params(param_loc)
        self.batch_components = batch_components
        self.forward()
        batch_points = self.batch_points.detach().cpu().numpy()
        batch_mask = self.batch_mask.detach().cpu().numpy()
        concept_bases = self.batch_concept_bases.detach().cpu().numpy()
        concept_deltas = self.batch_concept_deltas.detach().cpu().numpy()
        return batch_points, concept_bases, concept_deltas

    def compute_ranks(self, batch, ref_scores, position_to_replace):
        """
        Computes filtered ranks.
        """
        batch_size = batch.size(0)
        ranks = []
        candidate_sizes = []

        for i in range(batch_size):
            fact = batch[i:i + 1].clone()
            ref_score = ref_scores[i:i + 1]

            if position_to_replace == 0:
                replacement_range = torch.arange(self.nb_entities, device=batch.device)
            elif position_to_replace == 2:
                replacement_range = torch.arange(self.nb_concepts, device=batch.device)
            else:
                raise ValueError("Invalid position_to_replace. Should be 0 (entity) or 2 (concept).")

            corrupted_facts = fact.repeat(len(replacement_range), 1)
            corrupted_facts[:, position_to_replace] = replacement_range

            if self.known_facts is not None:
                mask = []
                for corrupted_fact in corrupted_facts:
                    candidate = (int(corrupted_fact[0]), int(corrupted_fact[1]), int(corrupted_fact[2]))
                    mask.append(candidate not in self.known_facts or candidate == (
                    int(fact[0][0]), int(fact[0][1]), int(fact[0][2])))
                corrupted_facts = corrupted_facts[mask]
            else:
                warnings.warn("Filtering is disabled, ranks may be inaccurate.")

            corrupted_scores = self.score_forward_pass(corrupted_facts)
            rank = 1 + torch.sum((corrupted_scores < ref_score).float()).item()
            ranks.append(rank)
            candidate_sizes.append(len(corrupted_facts))

        return ranks, candidate_sizes

    def compute_metrics(self, relevant_ranks_list, exact_ranks_list, candidate_sizes_list):
        relevant_ranks_np = np.array(relevant_ranks_list)
        exact_ranks_np = np.array(exact_ranks_list)
        candidate_sizes_np = np.array(candidate_sizes_list)

        adjusted_mean_rank_index = metrics_helper.adjusted_mean_rank_index(exact_ranks_np, candidate_sizes_np)

        rank_biased_precision_value = rank_biased_precision(relevant_ranks_np, p=0.9)
        mean_first_rank = metrics_helper.mean_first_relevant(relevant_ranks_np)

        hits_at_values = [np.mean(exact_ranks_np <= k) for k in self.hits_at]

        if self.verbose:
            print(f"AMRI: {adjusted_mean_rank_index}")
            print(f"RBP (p=0.9): {rank_biased_precision_value}")
            print(f"MFR: {mean_first_rank}")
            for k, hit_value in zip(self.hits_at, hits_at_values):
                print(f"Hits@{k}: {hit_value}")

        return Metrics(adjusted_mean_rank_index, rank_biased_precision_value, mean_first_rank, hits_at_values)

    def _calculate_first_relevant_rank(self, original_fact_cpu, original_score_cpu,
                                       candidate_facts_cpu, candidate_scores_cpu,
                                       arity_pos_to_replace):
        """
        Calculates the rank of the first relevant candidate fact.
        """
        true_entity_id = original_fact_cpu[0, 0].item()
        true_relation_id = original_fact_cpu[0, 1].item()
        true_concept_id = original_fact_cpu[0, 2].item()

        all_considered_facts_tuples = [(true_entity_id, true_relation_id, true_concept_id)]
        all_considered_scores = [original_score_cpu.item()]

        if candidate_facts_cpu.nelement() > 0:
            for i in range(candidate_facts_cpu.shape[0]):
                all_considered_facts_tuples.append(
                    (candidate_facts_cpu[i, 0].item(),
                     candidate_facts_cpu[i, 1].item(),
                     candidate_facts_cpu[i, 2].item())
                )
            all_considered_scores.extend(candidate_scores_cpu.tolist())

        if not all_considered_scores:
            return 1.0

        sorted_indices = sorted(range(len(all_considered_scores)), key=lambda k: all_considered_scores[k], reverse=True)

        true_noun_str = self._get_noun_from_entity_id(true_entity_id)
        base_concept_id_for_true_noun = self._get_base_concept_id_for_noun(true_noun_str) if true_noun_str else None

        for rank_minus_1, original_list_idx in enumerate(sorted_indices):
            rank = float(rank_minus_1 + 1)
            cand_entity_id, cand_relation_id, cand_concept_id = all_considered_facts_tuples[original_list_idx]

            if (cand_entity_id == true_entity_id and
                cand_relation_id == true_relation_id and
                cand_concept_id == true_concept_id):
                return rank

            if original_list_idx > 0:
                if arity_pos_to_replace == 2:
                    if base_concept_id_for_true_noun is not None:
                        if cand_concept_id == base_concept_id_for_true_noun:
                            return rank

                        if self._is_ancestor(cand_concept_id, base_concept_id_for_true_noun):
                            return rank

                        if self._is_ancestor(base_concept_id_for_true_noun, cand_concept_id):
                            return rank

                    if self._is_ancestor(cand_concept_id, true_concept_id):
                        return rank

                    if self._is_ancestor(true_concept_id, cand_concept_id):
                        return rank

                    true_parent = self.parent_of.get(true_concept_id)
                    cand_parent = self.parent_of.get(cand_concept_id)
                    if true_parent is not None and true_parent == cand_parent:
                        return rank

                elif arity_pos_to_replace == 0:


                    cand_noun_str = self._get_noun_from_entity_id(cand_entity_id)
                    if cand_noun_str == true_noun_str and cand_noun_str is not None:
                        return rank

                    base_concept_id_for_cand_noun = self._get_base_concept_id_for_noun(cand_noun_str) if cand_noun_str else None

                    if base_concept_id_for_cand_noun is not None:

                        if base_concept_id_for_cand_noun == true_concept_id:
                            return rank

                        if self._is_ancestor(true_concept_id, base_concept_id_for_cand_noun):
                            return rank
                        if self._is_ancestor(base_concept_id_for_cand_noun, true_concept_id):
                            return rank

                    if (cand_entity_id, true_relation_id, true_concept_id) in self.known_facts:
                        return rank

                    if base_concept_id_for_cand_noun is not None and base_concept_id_for_true_noun is not None:
                        if base_concept_id_for_cand_noun == base_concept_id_for_true_noun:
                            return rank

        return float(len(all_considered_scores) + 1) if all_considered_scores else 1.0

    def validate(self, dataset=cnst.VALID, debug_candidates=False, log_details=True, max_facts_to_process=None,
                 max_facts_per_position=None, batch_facts_for_corruption=32, corruption_batch_size=2000):
        """
        Validate the model on a chosen dataset.
        """

        original_training_state = self.training
        original_dropout_prob = self.dim_dropout_prob
        original_nb_neg = self.nb_neg

        self.dim_dropout_prob = 0.0
        self.nb_neg = 0
        self.eval()

        if dataset == cnst.VALID:
            data_loader = self.vl_loader
        elif dataset == cnst.TEST:
            data_loader = self.ts_loader
        else:
            data_loader = self.tr_ts_loader

        all_relevant_ranks_list = []
        all_exact_ranks_list = []
        all_candidate_sizes_list = []
        log_data = []

        entity_pos = 0
        concept_pos = 2

        if self.verbose:
            val_start_time = time.time()
            log_data.append(f"\n=== Starting Validation ({dataset}) ===")

        facts_processed = 0
        total_corruption_time = 0
        total_scoring_time = 0
        total_ranking_time = 0

        facts_processed_by_pos = {entity_pos: 0, concept_pos: 0}

        with torch.no_grad():
            for batch_idx, batch_data_cpu in enumerate(data_loader):
                current_batch_tensor = batch_data_cpu[0].to(device)

                current_batch_size = current_batch_tensor.size(0)
                if current_batch_size == 0:
                    continue

                score_start = time.time()
                ref_scores_full = self.score_forward_pass(current_batch_tensor)
                ref_scores = ref_scores_full.squeeze(-1)
                total_scoring_time += time.time() - score_start

                if ref_scores.dim() == 0:
                    ref_scores = ref_scores.unsqueeze(0)

                if max_facts_to_process is not None and facts_processed + current_batch_size > max_facts_to_process:

                    current_batch_tensor = current_batch_tensor[:max_facts_to_process - facts_processed]
                    ref_scores = ref_scores[:max_facts_to_process - facts_processed]
                    current_batch_size = current_batch_tensor.size(0)

                for arity_pos_to_replace in [entity_pos, concept_pos]:

                    if max_facts_per_position is not None and facts_processed_by_pos[arity_pos_to_replace] >= max_facts_per_position:
                        log_data.append(f"Processed maximum {max_facts_per_position} facts for position {arity_pos_to_replace}, skipping remaining in this position.")
                        continue

                    remaining_facts_for_pos = (
                        max_facts_per_position - facts_processed_by_pos[arity_pos_to_replace]
                        if max_facts_per_position is not None
                        else current_batch_size
                    )

                    facts_to_process = min(current_batch_size, remaining_facts_for_pos)

                    for start_idx in range(0, facts_to_process, batch_facts_for_corruption):
                        end_idx = min(start_idx + batch_facts_for_corruption, facts_to_process)
                        facts_in_minibatch = end_idx - start_idx

                        minibatch_tensor = current_batch_tensor[start_idx:end_idx]
                        minibatch_scores = ref_scores[start_idx:end_idx]

                        corruption_start = time.time()

                        all_candidate_facts_cpu, all_candidate_scores_cpu, all_candidate_counts, original_fact_indices = (
                            self._get_corruptions_for_fact_batch(
                                minibatch_tensor, arity_pos_to_replace, corruption_batch_size
                            )
                        )

                        total_corruption_time += time.time() - corruption_start

                        ranking_start = time.time()

                        for i in range(facts_in_minibatch):
                            original_fact = minibatch_tensor[i:i+1]
                            original_fact_cpu = original_fact.cpu()
                            original_fact_score = minibatch_scores[i].item()
                            fact_mask = original_fact_indices == i
                            if torch.any(fact_mask):
                                candidate_facts = all_candidate_facts_cpu[fact_mask]
                                candidate_scores = all_candidate_scores_cpu[fact_mask]
                                candidate_count = all_candidate_counts[i].item()
                                exact_rank, relevant_rank = self._calculate_ranks(
                                    original_fact_cpu, original_fact_score,
                                    candidate_facts, candidate_scores,
                                    arity_pos_to_replace
                                )
                            else:
                                exact_rank = 1.0
                                relevant_rank = 1.0
                                candidate_count = 0

                            all_relevant_ranks_list.append(relevant_rank)
                            all_exact_ranks_list.append(exact_rank)
                            all_candidate_sizes_list.append(candidate_count)

                            if log_details:
                                self._generate_rank_logs(
                                    log_data, original_fact, arity_pos_to_replace,
                                    original_fact_score, exact_rank, relevant_rank, candidate_count,
                                    candidate_facts if fact_mask.any() else None,
                                    candidate_scores if fact_mask.any() else None
                                )

                        total_ranking_time += time.time() - ranking_start

                        facts_processed_by_pos[arity_pos_to_replace] += facts_in_minibatch

                    if max_facts_per_position is not None and facts_processed_by_pos[arity_pos_to_replace] >= max_facts_per_position:
                        log_data.append(f"Processed maximum {max_facts_per_position} facts for position {arity_pos_to_replace}.")

                facts_processed += current_batch_size

                if max_facts_to_process is not None and facts_processed >= max_facts_to_process:
                    log_data.append(f"Processed maximum {max_facts_to_process} facts total, stopping validation.")
                    break

        if self.verbose:
            validation_time = time.time() - val_start_time
            log_data.append(f"Validation complete in {validation_time:.2f}s")
            log_data.append(f"Corruption generation: {total_corruption_time:.2f}s")
            log_data.append(f"Scoring time: {total_scoring_time:.2f}s")
            log_data.append(f"Ranking time: {total_ranking_time:.2f}s")
            log_data.append(f"Facts processed by position: Entity: {facts_processed_by_pos[entity_pos]}, Concept: {facts_processed_by_pos[concept_pos]}")

        if log_details and hasattr(self, '_bad_rank_logs'):
            self._add_worst_rank_logs(log_data)

        if log_details:
            for line in log_data:
                print(line)

        if not all_relevant_ranks_list:
            warnings.warn("No ranks computed during validation. Returning default/empty metrics.")
            final_metrics = Metrics(0.0, 0.0, 0.0, [0.0] * len(self.hits_at))
        else:
            final_metrics = self.compute_metrics(all_relevant_ranks_list, all_exact_ranks_list, all_candidate_sizes_list)

        self.dim_dropout_prob = original_dropout_prob
        self.nb_neg = original_nb_neg
        self.train(original_training_state)

        return final_metrics

    def _get_corruptions_for_fact_batch(self, facts_to_corrupt, arity_pos_to_replace, corruption_batch_size):
        """
        Get corruptions and scores for a batch of facts, using batch processing.

        This version processes multiple facts at once rather than sequentially, reducing validation time.

        1. Processes multiple facts (batch_facts_for_corruption) at once
        2. Creates corruptions in large batches (corruption_batch_size)
        3. Uses efficient filtering with a single pass over the corruptions
        4. Scores all corruptions in a single batch for greater throughput
        5. Returns a mapping that associates each corruption with its original fact
        """
        batch_size = facts_to_corrupt.size(0)

        if batch_size == 0:
            return (torch.empty(0, facts_to_corrupt.shape[1], dtype=facts_to_corrupt.dtype, device='cpu'),
                    torch.empty(0, dtype=torch.float, device='cpu'),
                    torch.zeros(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.long, device=device))

        if arity_pos_to_replace == 0:
            max_corruptions = self.nb_entities
        else:
            max_corruptions = self.nb_concepts

        replacement_values = torch.arange(max_corruptions, device=device)

        original_facts_tuples = set()
        for i in range(batch_size):
            original_facts_tuples.add((
                facts_to_corrupt[i, 0].item(),
                facts_to_corrupt[i, 1].item(),
                facts_to_corrupt[i, 2].item()
            ))

        all_filtered_corruptions = []
        all_fact_indices = []
        candidate_counts = torch.zeros(batch_size, dtype=torch.long, device=device)

        num_batches = (max_corruptions + corruption_batch_size - 1) // corruption_batch_size

        for b_idx in range(num_batches):
            start_idx = b_idx * corruption_batch_size
            end_idx = min((b_idx + 1) * corruption_batch_size, max_corruptions)
            current_batch_size = end_idx - start_idx

            if current_batch_size <= 0:
                continue

            current_replacements = replacement_values[start_idx:end_idx]

            all_batch_corruptions = []
            all_batch_fact_indices = []

            for fact_idx in range(batch_size):

                fact_corruptions = facts_to_corrupt[fact_idx:fact_idx+1].repeat(current_batch_size, 1)
                fact_corruptions[:, arity_pos_to_replace] = current_replacements

                all_batch_corruptions.append(fact_corruptions)
                all_batch_fact_indices.extend([fact_idx] * current_batch_size)

            batch_corruptions = torch.cat(all_batch_corruptions, dim=0)
            batch_fact_indices = torch.tensor(all_batch_fact_indices, device=device)

            keep_mask = torch.ones(batch_corruptions.size(0), dtype=torch.bool, device=device)

            batch_triples = [
                (batch_corruptions[k, 0].item(),
                 batch_corruptions[k, 1].item(),
                 batch_corruptions[k, 2].item())
                for k in range(batch_corruptions.size(0))
            ]

            filter_triples = set(batch_triples) & (self.known_facts - original_facts_tuples)

            if filter_triples:
                for k, triple in enumerate(batch_triples):
                    if triple in filter_triples:
                        keep_mask[k] = False

            filtered_corruptions = batch_corruptions[keep_mask]
            filtered_fact_indices = batch_fact_indices[keep_mask]

            if filtered_fact_indices.numel() > 0:
                unique_indices, counts = torch.unique(filtered_fact_indices, return_counts=True)
                candidate_counts.index_add_(0, unique_indices, counts)

            if filtered_corruptions.size(0) == 0:
                continue

            corruptions_scores = self.score_forward_pass(filtered_corruptions).squeeze(-1)

            all_filtered_corruptions.append((filtered_corruptions.cpu(), corruptions_scores.cpu(), filtered_fact_indices.cpu()))

        if all_filtered_corruptions:
            all_candidate_facts = []
            all_candidate_scores = []
            all_candidate_indices = []

            for corruptions, scores, indices in all_filtered_corruptions:
                all_candidate_facts.append(corruptions)
                all_candidate_scores.append(scores)
                all_candidate_indices.append(indices)

            all_candidate_facts_cpu = torch.cat(all_candidate_facts, dim=0)
            all_candidate_scores_cpu = torch.cat(all_candidate_scores, dim=0)
            original_fact_indices = torch.cat(all_candidate_indices, dim=0)
        else:
            all_candidate_facts_cpu = torch.empty(0, facts_to_corrupt.shape[1], dtype=facts_to_corrupt.dtype, device='cpu')
            all_candidate_scores_cpu = torch.empty(0, dtype=torch.float, device='cpu')
            original_fact_indices = torch.empty(0, dtype=torch.long, device='cpu')

        return all_candidate_facts_cpu, all_candidate_scores_cpu, candidate_counts, original_fact_indices

    def _calculate_ranks(self, original_fact_cpu, original_score, candidate_facts_cpu, candidate_scores_cpu, arity_pos_to_replace):
        """
        Calculate exact and relevant ranks for a fact.
        """

        exact_rank = 1.0
        relevant_rank = 1.0

        if len(candidate_scores_cpu) > 0:

            original_score_tensor = torch.tensor(original_score)
            exact_rank = 1.0 + torch.sum(candidate_scores_cpu > original_score_tensor).item()

            relevant_rank = self._calculate_first_relevant_rank(
                original_fact_cpu,
                original_score_tensor,
                candidate_facts_cpu,
                candidate_scores_cpu,
                arity_pos_to_replace
            )

        return exact_rank, relevant_rank


    def set_up_valid_net(self):
        options_no_neg = copy.deepcopy(self.options)
        options_no_neg.nb_neg_examples_per_pos = 0
        valid_net = BoxT(self.kb_name, options_no_neg, "_val")

        valid_net.dim_dropout_prob = 0.0
        valid_net.training = False
        return valid_net

    def train_with_valid(self, separate_valid_model=True, epoch_ckpt=50, save_period=1000,
                         num_epochs=1000, reset_weights=True, log_to_file=True, log_file_name="training_log.txt"):
        self.log_to_file = log_to_file
        self.log_to_file_name = log_file_name
        self.separate_valid_model = separate_valid_model
        self.epoch_checkpoint = epoch_ckpt

        if self.separate_valid_model:
            valid_model = self.set_up_valid_net()
        else:
            valid_model = self

        if log_to_file:
            with open(log_file_name, "w") as f:
                f.write("")

        losses = []
        if reset_weights:
            self.apply(self._weights_init)
        else:
            self.load_params()

        batch_total_count = 0
        try:
            if not os.path.exists('training_ckpts'):
                os.mkdir('training_ckpts')

            self._log_initial_configuration(log_to_file, log_file_name, num_epochs)

            for epoch_index in range(num_epochs):
                self.epoch = epoch_index + 1

                average_epoch_loss = 0
                print_or_log(f"Epoch {epoch_index + 1}", log_to_file, log_file_name)

                for batch_index, batch in enumerate(self.tr_loader):
                    self.optimizer.zero_grad()

                    self.batch_components, self.replaced_indices = batch
                    self.batch_components = self.batch_components.to(device)

                    if self.replaced_indices is not None:
                        self.replaced_indices = self.replaced_indices.to(device)

                    self.original_batch_size = self.batch_components.shape[0] // (1 + self.nb_neg)
                    self.current_batch_count = batch_total_count
                    loss = self.forward()
                    loss.backward()

                    if self.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip)
                    self.optimizer.step()

                    average_epoch_loss += loss.item()
                    batch_total_count += 1

                average_epoch_loss /= len(self.tr_loader)
                print_or_log(f"Epoch {epoch_index + 1} Complete. Average Epoch Loss: {average_epoch_loss}", log_to_file,
                             log_file_name)
                losses.append(average_epoch_loss)

                if self.generate_report:
                    self.summary_writer.add_scalar('Average Epoch Loss', average_epoch_loss, epoch_index + 1)

                if epoch_index % save_period == 0 and epoch_index > 0:
                    self.save_params()
                    print_or_log(f"Model checkpointed at epoch {epoch_index}", log_to_file, log_file_name)

                if epoch_index % epoch_ckpt == 0 and epoch_index > 0:
                    self.perform_validation(valid_model, epoch_index)
                    print(f"Loss value: {loss}")

            self.perform_validation(valid_model, num_epochs)

        except KeyboardInterrupt:
            print_or_log("Training Stopped", log_to_file, log_file_name)

        self.save_params()
        print_or_log(f"Weights saved to {self.param_directory}", log_to_file, log_file_name)
        if self.generate_report:
            self.summary_writer.close()

    def perform_validation(self, valid_model, epoch_index):
        if self.verbose:
            print("\nEpoch:", epoch_index)

        if self.separate_valid_model:

            with torch.no_grad():
                valid_model.load_state_dict(self.state_dict())
                valid_model.eval()

        print_or_log("Checkpoint Reached. Evaluating Metrics...", self.log_to_file, self.log_to_file_name)

        validation_start_time = time.time()
        with torch.no_grad():
            metrics = valid_model.validate(
                log_details=self.verbose,
                max_facts_to_process=4,
                max_facts_per_position=4,
                batch_facts_for_corruption=32,
                corruption_batch_size=2000
            )
        validation_time = time.time() - validation_start_time
        print_or_log(f"Validation completed in {validation_time:.2f} seconds", self.log_to_file, self.log_to_file_name)

        if self.generate_report:
            self.summary_writer.add_scalar('Adjusted Mean Rank Index', metrics.adjusted_mean_rank_index, epoch_index)
            self.summary_writer.add_scalar('Rank Biased Precision', metrics.rank_biased_precision, epoch_index)
            self.summary_writer.add_scalar('Mean First Relevant', metrics.mean_first_rank, epoch_index)
            self.summary_writer.add_scalar('Hits@1', metrics.hits_at_values[0], epoch_index)
            self.summary_writer.add_scalar('Hits@3', metrics.hits_at_values[1], epoch_index)
            self.summary_writer.add_scalar('Hits@5', metrics.hits_at_values[2], epoch_index)
            self.summary_writer.add_scalar('Hits@10', metrics.hits_at_values[3], epoch_index)

        if self.save_pos:
            self.save_positions(epoch_index)

        if self.viz_mode:
            self.visualize_embeddings(epoch_index)


    def save_positions(self, epoch):
        file_path = os.path.join(self.positions_dir, f"positions_epoch_{epoch}.txt")

        with open(file_path, "w") as f:
            f.write("Entity Positions:\n")
            for entity, idx in self.e2id_dict.items():
                embedding = self.entity_points.weight[idx].detach().cpu().numpy()
                f.write(f"{entity}: {embedding.tolist()}\n")

            f.write("\nConcept Positions (as Boxes):\n")
            for concept, idx in self.c2id_dict.items():
                base = self.concept_bases.weight[idx].detach().cpu().numpy()
                delta = F.softplus(self.concept_deltas_raw.weight[idx]).detach().cpu().numpy()
                f.write(f"{concept}:\n\t Center={base.tolist()},\n\t Width={delta.tolist()}\n")

    def visualize_embeddings(self, epoch, data_set='train', method='pca', n_components=2,
                             max_entities=20, max_concepts=10, label_entities=True, label_concepts=True):
        """
        Visualize entity embeddings and concept embeddings (boxes) in 2D space using dimensionality reduction.
        """

        if self.verbose:
            print("Visualizing embeddings...")

        entity_embeddings = self.entity_points.weight.detach().cpu().numpy()
        concept_bases = self.concept_bases.weight.detach().cpu().numpy()
        concept_deltas = F.softplus(self.concept_deltas_raw.weight).detach().cpu().numpy()

        concept_lows = concept_bases - concept_deltas * 0.5
        concept_highs = concept_bases + concept_deltas * 0.5

        entity_labels = [self.entity_id_to_name.get(i, f'Entity_{i}') for i in range(len(entity_embeddings))]
        concept_labels = [self.concept_id_to_name.get(i, f'Concept_{i}') for i in range(len(concept_bases))]

        if data_set == 'train':
            data_loader = self.tr_ts_loader
        elif data_set == 'valid':
            data_loader = self.vl_loader
        elif data_set == 'test':
            data_loader = self.ts_loader
        else:
            raise ValueError(f"Unknown data_set: {data_set}. Choose from 'train', 'valid', or 'test'.")

        entity_indices = set()
        concept_indices = set()
        for batch in data_loader:
            batch_tensor = batch[0]
            entities_in_batch = batch_tensor[:, 0].cpu().numpy()
            concepts_in_batch = batch_tensor[:, 2].cpu().numpy()
            entity_indices.update(entities_in_batch.flatten())
            concept_indices.update(concepts_in_batch.flatten())

        entity_indices = sorted(entity_indices)
        concept_indices = sorted(concept_indices)

        if max_entities < len(entity_indices):
            entity_indices = random.sample(entity_indices, max_entities)
        if max_concepts < len(concept_indices):
            concept_indices = random.sample(concept_indices, max_concepts)

        entity_embeddings = entity_embeddings[entity_indices]
        concept_bases = concept_bases[concept_indices]
        concept_lows = concept_lows[concept_indices]
        concept_highs = concept_highs[concept_indices]

        entity_labels = [entity_labels[i] for i in entity_indices]
        concept_labels = [concept_labels[i] for i in concept_indices]

        all_embeddings = np.vstack((entity_embeddings, concept_bases, concept_lows, concept_highs))

        if method == 'pca':
            reducer = PCA(n_components=n_components)
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}. Choose 'pca' or 'tsne'.")

        embeddings_2d = reducer.fit_transform(all_embeddings)

        n_entities = len(entity_embeddings)
        n_concepts = len(concept_bases)
        entity_embeddings_2d = embeddings_2d[:n_entities]
        concept_bases_2d = embeddings_2d[n_entities:n_entities + n_concepts]
        concept_lows_2d = embeddings_2d[n_entities + n_concepts:n_entities + 2 * n_concepts]
        concept_highs_2d = embeddings_2d[n_entities + 2 * n_concepts:]

        plt.figure(figsize=(10, 8))
        scatter_plot = plt.scatter(entity_embeddings_2d[:, 0], entity_embeddings_2d[:, 1], label='Entities', alpha=0.7, s=10)

        ax = plt.gca()

        for i in range(n_concepts):

            low = concept_lows_2d[i]
            high = concept_highs_2d[i]
            width = high[0] - low[0]
            height = high[1] - low[1]
            rect = patches.Rectangle((low[0], low[1]), width, height, linewidth=1, edgecolor='red', facecolor='none',
                                     alpha=0.7)
            ax.add_patch(rect)

            if label_concepts:
                center = concept_bases_2d[i]
                plt.text(center[0], center[1], concept_labels[i], fontsize=9, ha='center', va='center', color='red')

        if label_entities:
            for i in range(len(entity_labels)):
                plt.text(entity_embeddings_2d[i, 0], entity_embeddings_2d[i, 1], entity_labels[i], fontsize=8,
                         ha='right',
                         va='bottom')

        entity_legend_proxy = plt.Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor=scatter_plot.get_facecolor()[0][:3], markersize=6)
        concept_legend_proxy = patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='red',
                                                 facecolor='none', alpha=0.7)
        plt.legend(
            handles=[entity_legend_proxy, concept_legend_proxy],
            labels=['Entities', 'Concepts'],
            loc='upper left'
        )

        plt.autoscale()
        plt.grid(True)

        plt.title(f"Embeddings Visualization (Epoch {epoch}) - Data Set: {data_set.capitalize()}")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")

        file_name = f"epoch_{epoch}.png"
        plt.savefig(os.path.join(self.viz_dir, file_name))
        plt.close()

    def save_params(self, param_loc=None):
        if param_loc is None:
            param_loc = self.param_directory
        directory = os.path.dirname(param_loc)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        torch.save(self.state_dict(), param_loc)

    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _log_initial_configuration(self, log_to_file, log_file_name, num_epochs):
        config_strings = [
            f"BoxEMulti:",
            f"Training for {self.kb_name}:",
            f"Embedding Dimension: {self.embedding_dim}",
            f"Checkpoint Frequency: {self.lr_decay_period}",
            f"Number of Epochs: {num_epochs}",
            f"Learning Rate: {self.learning_rate}",
            f"LR Decay: {self.lr_decay}/Period: {self.lr_decay_period}",
            f"Loss Margin: {self.margin}",
            f"Shared Shape: {self.shared_shape}",
            f"Learnable Shape: {self.learnable_shape}",
            f"Fixed Width: {self.fixed_width}",
            f"Stop Gradients: {self.stop_gradient}",
            f"Negative Sampling: {self.neg_sampling_opt}",
            f"Reg Lambda: {self.reg_lambda}",
            f"Reg Points: {self.reg_points}",
            f"Bounded Pt: {self.bounded_pt_space}",
            f"Bounded Box: {self.bounded_box_space}",
            f"Bound Scale: {self.bound_scale}" if self.bounded_pt_space or self.bounded_box_space else "",
            f"Hard Size: {self.total_size}" if self.hard_total_size else "Hard Size: NO",
            f"Hard Code Size: {self.hard_code_size}",
            f"Objective Function: {self.obj_fct}",
            f"Loss Function: {self.loss_fct}",
            f"Loss Order: {self.loss_ord}",
            f"Dim_Dropout Probability: {self.dim_dropout_flt}",
            f"Gradient Clip: {self.gradient_clip}",
            f"Bounded Norm: {self.bounded_norm}",
            f"Data Augmentation: {self.augment_inv}",
            f"WARNING, Regularising with hard constraints. Disable one." if self.reg_lambda > 0 and self.hard_total_size else ""
        ]
        for line in config_strings:
            if line:
                print_or_log(line, log_to_file, log_file_name)

    def save_data_for_export(self, directory_path):
        """
        Saves data required by standalone export functions to the specified directory.
        This includes concept_bases, concept_deltas, concept_id_to_name, and kb_name.
        """
        os.makedirs(directory_path, exist_ok=True)

        try:

            concept_bases_np = self.concept_bases.weight.detach().cpu().numpy()
            concept_deltas_np = F.softplus(self.concept_deltas_raw.weight).detach().cpu().numpy()
            entity_points_np = self.entity_points.weight.detach().cpu().numpy()

            np.save(os.path.join(directory_path, "concept_bases.npy"), concept_bases_np)
            np.save(os.path.join(directory_path, "concept_deltas.npy"), concept_deltas_np)
            np.save(os.path.join(directory_path, "entity_points.npy"), entity_points_np)

            if hasattr(self, 'concept_id_to_name') and self.concept_id_to_name:
                with open(os.path.join(directory_path, "concept_id_to_name.json"), 'w') as f:
                    json.dump(self.concept_id_to_name, f, indent=2)
            else:
                print(f"Warning: self.concept_id_to_name not found or empty. Cannot save it.")

            export_info = {
                "kb_name": self.kb_name,
                "embedding_dim": self.embedding_dim,
                "nb_entities": self.nb_entities,
                "nb_concepts": self.nb_concepts,
                "summary_dir": self.summary_dir if hasattr(self, 'summary_dir') else None
                }
            with open(os.path.join(directory_path, "export_info.json"), 'w') as f:
                json.dump(export_info, f, indent=2)

        except Exception as e:
            print(f"Error saving data for export to {directory_path}: {e}")

    def loss_function_reconstruction(self, batch_points, batch_mask, box_lows, box_highs, box_mults,
                                     typicality_scores,
                                     order=2,
                                     batch_counter=None):
        """
        Implements the reconstruction loss: L_typ + lambda_neg * L_neg. (Not used in the end)
        """
        entity_embeds = batch_points.squeeze(1) if batch_points.dim() > 2 else batch_points
        concept_lowers = box_lows.squeeze(1) if box_lows.dim() > 2 else box_lows
        concept_uppers = box_highs.squeeze(1) if box_highs.dim() > 2 else box_highs


        dist_outside_upper = torch.relu(entity_embeds - concept_uppers)**2
        dist_outside_lower = torch.relu(concept_lowers - entity_embeds)**2

        d_e_C = torch.sum(dist_outside_upper + dist_outside_lower, dim=-1)

        p_e_C = torch.exp(-d_e_C)

        positive_mask = (typicality_scores >= 0)

        loss_positive_terms = torch.zeros_like(d_e_C)
        if positive_mask.any():
            loss_positive_terms[positive_mask] = d_e_C[positive_mask]

        loss_negative_terms = torch.zeros_like(p_e_C)
        negative_indices = torch.where(typicality_scores < 0)[0]
        if negative_indices.numel() > 0:
          loss_negative_terms[negative_indices] = p_e_C[negative_indices]**2

        final_loss_per_example = torch.where(positive_mask, loss_positive_terms, loss_negative_terms)

        return final_loss_per_example * batch_mask.squeeze()

    def calculate_concept_box_dim_penalty(self, batch_components, batch_concept_bases, batch_concept_deltas,
                                          original_batch_size):
        """
        Compute regularization terms for geometric constraints.
        """
        
        if original_batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        if not self.training:
            return torch.tensor(0.0, device=device, requires_grad=True)

        w1, w2 = 5.0, 1.0

        xi = batch_components[:original_batch_size, 0]
        yi = batch_components[:original_batch_size, 2].long()
        zi = batch_components[:original_batch_size, 3].float()

        zn = torch.where(zi >= 0, zi / 1000.0, torch.zeros_like(zi))
        m = zn >= 0
        nm = torch.sum(m)

        if nm <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        fx = xi[m]
        fy = yi[m]
        fz = zn[m]

        uy, _ = torch.unique(fy, return_counts=True)
        ku = len(uy)

        if ku <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        reg = self._build_reg(fx, fy, fz)

        vl = list(reg.keys())
        np = ku * (ku - 1) // 2
        pt = []
        at = []
        pc = 0
        for i in range(len(vl)):
            for j in range(i + 1, len(vl)):
                a, b = vl[i], vl[j]
                ca = reg[a]['ents']
                cb = reg[b]['ents']
                
                if not ca or not cb:
                    continue
                eta = len(ca.intersection(cb)) / min(len(ca), len(cb))
                nu = self._comp_nu(a, b, ca, cb, reg)
                pa = self._ext_p(batch_concept_bases, batch_concept_deltas, yi, m, a)
                pb = self._ext_p(batch_concept_bases, batch_concept_deltas, yi, m, b)
                
                if pa is None or pb is None:
                    continue

                mu = self._comp_mu(pa, pb)
                delta = self._apply_delta(eta, mu)
                if delta is not None:
                    pt.append(delta)
                if nu > 0:
                    at.append(nu)
                pc += 1

        if pc == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        tp = torch.tensor(0.0, device=device, requires_grad=True)
        ta = torch.tensor(0.0, device=device, requires_grad=True)

        if pt:
            tp = torch.mean(torch.stack(pt))

        if at:
            ta = torch.mean(torch.stack(at))

        res = (w1 * tp) - (w2 * ta)

        return res

    def _build_reg(self, fx, fy, fz):
        """Build entity-concept mapping with auxiliary data."""
        reg = defaultdict(lambda: {'ents': set(), 'aux': {}})
        
        x_cpu = fx.cpu().numpy()
        y_cpu = fy.cpu().numpy()
        z_cpu = fz.cpu().numpy()

        for i in range(len(x_cpu)):
            ck = int(y_cpu[i])
            ek = int(x_cpu[i])
            av = float(z_cpu[i])

            reg[ck]['ents'].add(ek)
            reg[ck]['aux'][ek] = av

        return reg

    def _comp_nu(self, a, b, ca, cb, reg):
        """Compute auxiliary attraction metric based on shared elements."""
        sf = ca.intersection(cb)
        nu = 0.0
        
        if sf:
            sa = sum(reg[a]['aux'].get(e, 0.0) for e in sf) / len(sf)
            sb = sum(reg[b]['aux'].get(e, 0.0) for e in sf) / len(sf)

            pt = sa * sb
            if pt > 0:
                nu = torch.tensor(
                    pt * len(sf) / (len(ca) + len(cb)),
                    device=device,
                    dtype=torch.float32
                )
        
        return nu

    def _ext_p(self, batch_concept_bases, batch_concept_deltas, yi, m, nid):
        """Extract geometric representation for concept."""
        ti = (yi == nid) & m
        
        if not torch.any(ti):
            return None
            
        tidx = torch.nonzero(ti, as_tuple=True)[0][0].item()
        
        c = batch_concept_bases[tidx]
        e = batch_concept_deltas[tidx]
        
        return {
            'lo': c - e * 0.5,
            'hi': c + e * 0.5,
            'c': c,
            'e': e
        }

    def _comp_mu(self, pa, pb):
        """Compute geometric intersection coefficient."""
        ilo = torch.max(pa['lo'], pb['lo'])
        ihi = torch.min(pa['hi'], pb['hi'])

        has_int = torch.all(ihi > ilo)

        if has_int:
            # Compute volumes using softplus for numerical stability
            va = torch.prod(F.softplus(pa['hi'] - pa['lo']))
            vb = torch.prod(F.softplus(pb['hi'] - pb['lo']))
            vi = torch.prod(F.softplus(ihi - ilo))
            vm = torch.min(va, vb)
            mu = vi / (vm + SANITY_EPS)
            return mu
        else:
            return torch.tensor(0.0, device=device)

    def _apply_delta(self, eta, mu):
        """Apply penalty function based on expected vs actual intersection."""
        if eta == 0:
            bt = mu ** 2 * 0.5
            et = torch.exp(torch.clamp(mu * 10.0, max=10.0)) - 1.0
            delta = et + bt

        else:
            if mu > 0:
                delta = F.relu(torch.tensor(eta, device=device) - mu) ** 2 * 10.0
            else:
                delta = torch.tensor(eta * 15.0, device=device, requires_grad=True)
        
        return delta

    def box_size_monotonicity_penalty(self, batch_components, batch_concept_bases, batch_concept_deltas,
                                      original_batch_size, monotonicity_weight=1.0):
        """
        Encourage size ordering constraint based on entity counts.
        """

        if original_batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        ent_idx = batch_components[:original_batch_size, 0]
        con_idx = batch_components[:original_batch_size, 2].long()
        typ_vals = batch_components[:original_batch_size, 3].float()
        pos_mask = typ_vals >= 0

        if torch.sum(pos_mask) <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        pos_ents = ent_idx[pos_mask]
        pos_cons = con_idx[pos_mask]

        phi_map = defaultdict(set)
        ents_cpu = pos_ents.cpu().numpy()
        cons_cpu = pos_cons.cpu().numpy()

        for i in range(len(ents_cpu)):
            cid = int(cons_cpu[i])
            eid = int(ents_cpu[i])
            phi_map[cid].add(eid)

        unique_cons = list(phi_map.keys())

        if len(unique_cons) <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Build index mapping
        con_to_idx = {}
        for idx, c in enumerate(con_idx[:original_batch_size].cpu().numpy()):
            if c in phi_map and c not in con_to_idx:
                con_to_idx[c] = idx

        penalty = torch.tensor(0.0, device=device, requires_grad=True)
        viol_cnt = 0


        for i, alpha in enumerate(unique_cons):
            for j, beta in enumerate(unique_cons):
                if i == j:
                    continue

                n_alpha = len(phi_map[alpha])
                n_beta = len(phi_map[beta])

                if n_alpha > n_beta:
                    idx_alpha = con_to_idx[alpha]
                    idx_beta = con_to_idx[beta]

                    delta_alpha = batch_concept_deltas[idx_alpha]
                    delta_beta = batch_concept_deltas[idx_beta]
                    sigma_alpha = F.softplus(delta_alpha)
                    sigma_beta = F.softplus(delta_beta)
                    vol_alpha = torch.prod(sigma_alpha)
                    vol_beta = torch.prod(sigma_beta)

                    rho = F.relu(vol_beta - vol_alpha)

                    if rho > 0:
                        viol_cnt += 1
                        penalty = penalty + rho

        final_pen = monotonicity_weight * penalty

        return final_pen


def loss_function_simple_geometric_distance(batch_points, batch_mask, box_lows, box_highs,
                                            box_mults,
                                            typicality_scores,
                                            order=1,
                                            batch_counter=None
                                            ):
    """
    Calculates the sum of distances for dimensions where the entity is outside the box.
    This is equivalent to the box_loss in the user's simple example.
    """
    entity_embeds = batch_points.squeeze(1) if batch_points.dim() > 2 else batch_points
    concept_lowers = box_lows.squeeze(1) if box_lows.dim() > 2 else box_lows
    concept_uppers = box_highs.squeeze(1) if box_highs.dim() > 2 else box_highs

    dist_out = torch.relu(entity_embeds - concept_uppers) + torch.relu(concept_lowers - entity_embeds)
    geometric_distance = torch.sum(dist_out, dim=-1)

    return geometric_distance * batch_mask.squeeze()


def total_box_size_regular(all_concept_deltas, reg_lambda, target_log_box_size):
    """
    Penalizes the deviation of the aggregated log geometric mean width of boxes from a target_log_box_size.
    """
    if all_concept_deltas.numel() == 0:
        return torch.tensor(0.0, device=all_concept_deltas.device)

    log_epsilon = 1e-20
    log_abs_deltas = torch.log(torch.abs(all_concept_deltas) + log_epsilon)
    mean_log_widths = torch.mean(log_abs_deltas, dim=1)

    if mean_log_widths.numel() == 0:
        return torch.tensor(0.0, device=all_concept_deltas.device)

    min_mean_log_width = torch.min(mean_log_widths).detach()
    width_ratios = torch.exp(mean_log_widths - min_mean_log_width)
    total_multiplier = torch.log(torch.sum(width_ratios) + SANITY_EPS)
    current_total_log_width = total_multiplier + min_mean_log_width
    size_constraint_loss = reg_lambda * (current_total_log_width - target_log_box_size)**2
    
    return size_constraint_loss

def loss_function_simple_box(batch_points, batch_mask, box_lows, box_highs, box_mults, typicality_scores,
                           dim_dropout_prob=0.0, order=1, batch_counter=None):
    """
    A simpler box-embedding loss function that prevents box collapse by using a direct implementation
    similar to the approach in main.py with the box_embeddings library. (Not used in the end)
    """

    entity_embeds = batch_points.squeeze(1) if batch_points.dim() > 2 else batch_points
    concept_lowers = box_lows.squeeze(1) if box_lows.dim() > 2 else box_lows
    concept_uppers = box_highs.squeeze(1) if box_highs.dim() > 2 else box_highs

    batch_mask = batch_mask.reshape(-1)

    dist_outside_upper = torch.relu(entity_embeds - concept_uppers)
    dist_outside_lower = torch.relu(concept_lowers - entity_embeds)
    dist_outside = dist_outside_upper + dist_outside_lower

    total_distance = torch.sum(dist_outside, dim=-1)

    containment_score = torch.exp(-total_distance)
    binary_labels = (typicality_scores >= 0).float()
    loss = F.binary_cross_entropy(containment_score, binary_labels, reduction='none')
    loss = loss * batch_mask
    
    return loss
            