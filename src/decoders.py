from time import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import random
from typing import Tuple


@torch.inference_mode()
def baseline_decode(model, tokenizer, prompt, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    start = time()
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    end = time()

    generated_tokens = output.shape[-1] - inputs['input_ids'].shape[-1]
    tps = generated_tokens / (end - start)

    metrics = {
        "time": end - start,
        "tokens": generated_tokens,
        "tps": tps
    }

    return tokenizer.decode(output[0], skip_special_tokens=True), metrics


@torch.inference_mode()
def speculative_decode(draft_model: str, target_model, tokenizer, prompt, k, max_new_tokens, top_p=1.0, temperature=1.0):
    """
    Approach based on DeepMind's paper(https://arxiv.org/pdf/2302.01318.pdf)
    """

    device = target_model.device

    # prefill for draft model
    inputs = tokenizer([prompt], return_tensors='pt').to(device)
    draft_output_prefill = draft_model(input_ids=inputs['input_ids'], use_cache=True)
    prefix_token_list = inputs['input_ids'][0].cpu().numpy().tolist()
    draft_past_key_values, logits = draft_output_prefill.past_key_values, draft_output_prefill.logits
    logits_prev_step = logits[:, -1, :]
    prob_prev_step = logits_adapter(logits_prev_step, temperature=temperature, top_p=top_p)

    # prefill for target model
    target_output_prefill = target_model(input_ids=inputs['input_ids'][:, :-1], use_cache=True)
    target_past_key_values = target_output_prefill.past_key_values
    target_prev_token_id = inputs['input_ids'][0, -1].cpu().item()

    output_ids = []

    n = 0
    accepted_tokens = 0
    draft_count = 0
    s_time = time()
    while n < max_new_tokens:
        # Every step/case features adding tokens to output, sampling next tokens, updating cache

        # -- Draft Tokens -- #
        draft_tokens = []
        draft_tokens_prob = []
        draft_prob = [prob_prev_step]

        for _ in range(k):
            # Sample a token from the multinomial distribution of draft model given current prefix
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(prob_prev_step)))
            next_draft_token = torch.argmax(torch.log(prob_prev_step) + gumbel_noise, dim=-1, keepdim=True)
            next_draft_token_prob = torch.gather(prob_prev_step, -1, next_draft_token)
            draft_tokens.append(next_draft_token[0, 0].cpu().item())
            draft_tokens_prob.append(next_draft_token_prob[0, 0].cpu().item())
            draft_outputs = draft_model(input_ids=next_draft_token, 
                past_key_values=draft_past_key_values,
                attention_mask=torch.ones(next_draft_token.shape[0], 1+draft_past_key_values[0][0].shape[2], dtype=torch.long, device=next_draft_token.device),
                position_ids=torch.LongTensor([draft_past_key_values[0][0].shape[2]]).to(device).view(-1, 1),
                use_cache=True)
            
            draft_past_key_values = draft_outputs.past_key_values
            draft_logits = draft_outputs.logits[:, -1, :]
            prob_prev_step = logits_adapter(draft_logits, temperature=temperature, top_p=top_p)
            draft_prob.append(prob_prev_step)
        draft_count += 1

        # -- Verify tokens -- #
        # We provide the last output token from target model + k drafted tokens
        target_input_ids = torch.tensor([[target_prev_token_id, *draft_tokens]], device=target_model.device)
        target_attention_mask = torch.ones(1, target_past_key_values[0][0].shape[2]+k+1, dtype=torch.long, device=next_draft_token.device)
        target_position_ids = torch.arange(target_past_key_values[0][0].shape[2], target_past_key_values[0][0].shape[2]+k+1).unsqueeze(0).to(next_draft_token.device)
        target_outputs = target_model(input_ids=target_input_ids, attention_mask=target_attention_mask, position_ids=target_position_ids, past_key_values=target_past_key_values, use_cache=True)
        target_past_key_values = target_outputs.past_key_values
        target_prob = logits_adapter(target_outputs.logits, temperature=temperature, top_p=top_p)
        
        # -- Acceptance Loop -- #
        # Check if draft token is accepted by rejection sampling
        all_accept = True 
        for i in range(k):
            target_token_prob = target_prob[0, i, draft_tokens[i]].cpu().item()

            # Perform rejection sampling
            if random.random() <= (target_token_prob / draft_tokens_prob[i]): pass
            else:
                all_accept = False
                accepted_tokens += i # i tokens were accepted before rejection
                n += (i+1)

                # Reject current token --> Add accepted prefix tokens to output, resample rejected token via relu
                modified_dist = relu_normalize(target_prob[0, i], draft_prob[i][0]) # (vocab_size, )
                resampled_token = modified_dist.multinomial(num_samples=1).unsqueeze(0)
                output_ids.extend([target_prev_token_id, *draft_tokens[:i]])
                target_prev_token_id = resampled_token[0,0].cpu().item()
                # Update draft kv cache and remove unverified tokens from KV cache
                target_past_key_values = truncate_kv_cache(target_past_key_values, truncate_size=k-i)
                draft_past_key_values = truncate_kv_cache(draft_past_key_values, truncate_size=k-i)
                
                # Resync draft model to use new accepted sequence and produce new tokens for next sampling
                draft_outputs = draft_model(input_ids=resampled_token, 
                    attention_mask=torch.ones(next_draft_token.shape[0], 1+draft_past_key_values[0][0].shape[2], dtype=torch.long, device=next_draft_token.device),
                    position_ids=torch.LongTensor([draft_past_key_values[0][0].shape[2]]).to(device).view(-1, 1),
                    past_key_values=draft_past_key_values,
                    use_cache=True)
                draft_past_key_values = draft_outputs.past_key_values
                draft_logits = draft_outputs.logits[:, -1, :]
                prob_prev_step = logits_adapter(draft_logits, temperature=temperature, top_p=top_p)
                break

        # Route for all drafted tokens being accepted
        if all_accept:
            accepted_tokens += k
            output_ids.extend([target_prev_token_id, *draft_tokens])
            target_next_token = target_prob[0, -1].multinomial(num_samples=1).unsqueeze(0)
            draft_outputs = draft_model(input_ids=target_next_token, 
                past_key_values=draft_past_key_values,
                attention_mask=torch.ones(next_draft_token.shape[0], 1+draft_past_key_values[0][0].shape[2], dtype=torch.long, device=next_draft_token.device),
                position_ids=torch.LongTensor([draft_past_key_values[0][0].shape[2]]).to(device).view(-1, 1),
                use_cache=True)
            draft_past_key_values = draft_outputs.past_key_values
            draft_logits = draft_outputs.logits[:, -1, :]
            prob_prev_step = logits_adapter(draft_logits, temperature=temperature, top_p=top_p)
            target_prev_token_id = target_next_token[0, 0].cpu().item()
            n += (k+1)
    e_time = time()
    run_time = e_time - s_time
    tps = n/(e_time-s_time)
    acceptance_rate = accepted_tokens / (draft_count*k)*100

    metrics = {
        "time": run_time,
        "tps": tps,
        "acceptance_rate": acceptance_rate
    }

    decoded_output = tokenizer.decode(prefix_token_list + output_ids, skip_special_tokens=True)
    return decoded_output, metrics

def relu_normalize(p, q):
    """Modify sampling distribution"""
    temp_dist = torch.relu(p-q)
    return temp_dist / temp_dist.sum(dim=-1, keepdim=True)

def truncate_kv_cache(cache, truncate_size):
    kv_cache = list(cache)
    for i in range(len(kv_cache)):
        kv_cache[i] = list(kv_cache[i])
        kv_cache[i][0] = kv_cache[i][0][:, :, :-truncate_size, :]
        kv_cache[i][1] = kv_cache[i][1][:, :, :-truncate_size, :]
    return kv_cache

def logits_adapter(logits, temperature, top_p):
    """
    Apply logits transformation 
    1. Logit to probability
    2. Apply temperature
    3. Apply top-p filter
    4. Normalize probabilities
    5. Restore shape
    """

    flag = False
    if logits.ndim==3:
        bsz = logits.shape[0]
        l = logits.shape[1]
        logits = logits.view(-1, logits.shape[-1])
        flag = True
    prob = torch.softmax(logits / temperature, dim=-1)
    sorted_prob, sorted_prob_idx = torch.sort(prob, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_prob, dim=-1)
    mask = (cumsum - sorted_prob) > top_p
    sorted_prob[mask] = 0.0
    sorted_prob.div_(sorted_prob.sum(dim=-1, keepdim=True))
    _, gather_pos = torch.sort(sorted_prob_idx, descending=False, dim=-1)
    final_prob = torch.gather(sorted_prob, -1, gather_pos)
    if flag: final_prob = final_prob.view(bsz, l, -1)
    return final_prob
