"""Frozen batch=1 latency and separately timed batched throughput on one device.

Measures text -> candidate extraction -> threshold/non-overlap spans, including
CPU tokenization, transfers and postprocessing. Model loading and literal mask
replacement are excluded. CUDA is synchronized at every timing boundary.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from evaluate import Predictor, load_frozen, read_jsonl, sha, synchronize, validate_rows, path_at


def percentile(values, q):
    if not values:
        raise ValueError('Empty timing sample')
    ordered = sorted(values); position = (len(ordered)-1)*q
    lo = int(position); hi = min(lo+1, len(ordered)-1)
    return ordered[lo] + (ordered[hi]-ordered[lo])*(position-lo)


def main():
    from common import write_json, seed_all
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--freeze', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args()
    root = Path(args.root); out = path_at(root, args.output)
    args.freeze = str(path_at(root, args.freeze))
    if args.batch_size < 1:
        raise ValueError('batch-size must be positive')
    # Verify every selection/weight set before creating a timing result.
    selections = {}
    for variant in ('base','full','lora'):
        frozen, selections[variant] = load_frozen(root, args.freeze, variant)
    if args.batch_size != frozen['benchmark_batch_size']:
        raise ValueError('Throughput batch size must be chosen on dev and recorded in freeze')
    rows = read_jsonl(root/'data/corpus-v1/test.jsonl'); validate_rows(rows, 'test', 1000)
    dev = read_jsonl(root/'data/corpus-v1/dev.jsonl'); validate_rows(dev, 'dev', 500)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'invocation.json', vars(args))
    write_json(out/'freeze-reference.json', {'path': str(args.freeze), 'sha256': sha(args.freeze)})
    results = []
    try:
        for variant, selection in selections.items():
            seed_all(20261114)
            selected = selection['selected']
            predictor = Predictor(root, selected['model_path'], args.device, frozen['precision'], selected.get('adapter_path'))
            try:
                for i in range(30):
                    predictor.predict([dev[i]['text']], selected['threshold'], 1)
                synchronize(args.device)
                timings = []
                with (out/f'latency-{variant}.jsonl').open('x') as stream:
                    for row in rows:
                        synchronize(args.device); started = time.perf_counter()
                        predictions = predictor.predict([row['text']], selected['threshold'], 1)
                        synchronize(args.device); seconds = time.perf_counter()-started
                        timing = {'id': row['id'], 'seconds': seconds, 'predicted_spans': len(predictions[0])}
                        timings.append(seconds); stream.write(json.dumps(timing)+'\n'); stream.flush()
                # Warm the chosen throughput batch shape on dev, not on test.
                for offset in range(0, 30*args.batch_size, args.batch_size):
                    batch = [dev[i % len(dev)]['text'] for i in range(offset, offset+args.batch_size)]
                    predictor.predict(batch, selected['threshold'], args.batch_size)
                batches = [[r['text'] for r in rows[i:i+args.batch_size]] for i in range(0, len(rows), args.batch_size)]
                synchronize(args.device); started = time.perf_counter()
                processed = 0
                for batch in batches:
                    predictions = predictor.predict(batch, selected['threshold'], args.batch_size)
                    processed += len(predictions)
                synchronize(args.device); throughput_seconds = time.perf_counter()-started
                if processed != len(rows):
                    raise ValueError('Throughput output count changed')
                result = {
                    'variant': variant, 'checkpoint': selected['name'], 'threshold': selected['threshold'],
                    'device': args.device, 'precision': frozen['precision'], 'adapter_merged': predictor.adapter_merged,
                    'load_seconds': predictor.load_seconds, 'latency_batch_size': 1, 'latency_documents': len(rows),
                    'warmup_calls': 30, 'latency_p50_ms': percentile(timings, .5)*1000,
                    'latency_p95_ms': percentile(timings, .95)*1000, 'latency_total_seconds': sum(timings),
                    'throughput_batch_size': args.batch_size, 'throughput_documents': processed,
                    'throughput_seconds': throughput_seconds, 'documents_per_second': processed/throughput_seconds,
                    'timing_scope': 'text to selected non-overlapping spans; tokenizer, transfers, inference, postprocessing; CUDA synchronized; mask replacement excluded',
                    'boundary_settings': predictor.settings,
                }
                results.append(result); write_json(out/'results.json', results)
                print(json.dumps(result), flush=True)
            finally:
                predictor.close()
        write_json(out/'complete.json', {'status': 'complete', 'created_utc': datetime.now(timezone.utc).isoformat(),
                   'freeze_sha256': sha(args.freeze), 'variants': ['base','full','lora'],
                   'note': 'Repeated calls measure speed only; they do not replace frozen test-quality predictions.'})
    except Exception as exc:
        write_json(out/'failure.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise


if __name__ == '__main__':
    main()
