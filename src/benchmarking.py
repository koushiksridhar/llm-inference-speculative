import yaml
import pandas as pd
import torch
from decoders import baseline_decode, speculative_decode
from utils import load_models, load_tokenizer, measure_time

def run_benchmark():
    # Load configs
    print('[INFO] Starting Benchmarking...')

    c_models = yaml.safe_load(open("configs/models.yaml"))
    c_benchmarks = yaml.safe_load(open("configs/benchmarks.yaml"))
    print('[INFO] Loaded configs...')


    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'[INFO] Device + {device}')

    tokenizer = load_tokenizer(c_models['tokenizer'])
    print('[INFO] Loaded tokenizer...')

    draft_model, target_model = load_models( c_models['draft_model'], c_models['target_model'], device)
    print('[INFO] Loaded models...')

    # -- Warmup --
    print('[INFO] Running warmup inference...')
    warmup_prompt = "Hello world!"
    warmup_inputs = tokenizer(warmup_prompt, return_tensors='pt').to(device)

    with torch.inference_mode():
        # Target model warmup
        target_model.generate(**warmup_inputs, max_new_tokens=5)
        # Draft model warmup
        draft_model.generate(**warmup_inputs, max_new_tokens=5)
    print('[INFO] Warmup done.\n')


    records = []

    for prompt in (c_benchmarks['prompts']):
        print(f'[INFO] Prompt: {prompt}')

        baseline_out, b_metrics = baseline_decode(target_model, tokenizer, prompt, c_benchmarks['max_new_tokens'])
        print(f"Ran Baseline @ {b_metrics['time']}.\n"
              f"Output: {baseline_out} \n"
              f"TPS: {b_metrics['tps']:.2f}")


        records.append({
            'prompt': prompt,
            'method': 'baseline',
            'draft_k': 0,
            'time': b_metrics['time'],
            'tokens_per_second': b_metrics['tps'],
            'output': baseline_out
        })

        for k in c_benchmarks['draft_lengths']:
            spec_out, s_metrics = speculative_decode(
                draft_model,
                target_model,
                tokenizer,
                prompt,
                k,
                c_benchmarks['max_new_tokens']
            )

            print(
                f"Ran Speculative {k} @ {s_metrics['time']:.4f}s.\n"
                f"Output: {spec_out}\n"
                f"Token Acceptance Rate: {s_metrics['acceptance_rate']:.2%}\n"
                f"TPS: {s_metrics['tps']:.2f}"
)
            records.append({
                'prompt': prompt,
                'method': 'speculative',
                'draft_k': k,
                'output': spec_out,
                'time': s_metrics["time"],
                'tokens_per_second': s_metrics["tps"],
                'token_acceptance_rate': s_metrics["acceptance_rate"]
            })

    df = pd.DataFrame(records)
    df.to_csv('results/benchmarks.csv', index=False)
    print('[INFO] Saved results...')


if __name__ == '__main__':
    run_benchmark()
