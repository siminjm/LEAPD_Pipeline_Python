import argparse, os, json
import numpy as np
from leapd import load_data, evaluate_classification, evaluate_correlation
from leapd import filter_data, compute_lpc_librosa, build_hyperplanes, compute_leapd_scores
from leapd import count_subjects, read_labels_table, fetch_targets
from leapd import combine_scores, generate_combinations

def _compute_lpc_dispatch(x, order, backend):
    from leapd import compute_lpc_librosa
    from leapd.signal import compute_lpc_burg
    if backend == 'burg':
        return compute_lpc_burg(x, order)
    else:
        return compute_lpc_librosa(x, order)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['classification','correlation'])
    ap.add_argument('--data_test', required=True)
    ap.add_argument('--trained_model', required=True)  # results/train_results/BestParamsAll.npz or .json
    ap.add_argument('--labels_file', default='')
    ap.add_argument('--combo_sizes', nargs='*', type=int, default=list(range(1,11)))
    ap.add_argument('--max_full_combos', type=int, default=5)
    ap.add_argument('--Fs', type=float, default=500.0)
    ap.add_argument('--lpc_backend', choices=['burg','librosa'], default='burg')
    ap.add_argument('--n_jobs', type=int, default=-1)
    ap.add_argument('--is_norm_proj', type=int, default=0)
    ap.add_argument('--save_dir', default='results/test_results')
    ap.add_argument('--lpc_backend', choices=['burg','librosa'], default='burg')
    ap.add_argument('--n_jobs', type=int, default=-1)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Load trained params
    BestParamsAll = None
    if args.trained_model.endswith('.npz'):
        BestParamsAll = np.load(args.trained_model, allow_pickle=True)['BestParamsAll'].tolist()
    else:
        with open(args.trained_model, 'r') as f:
            BestParamsAll = json.load(f)

    DataTest, _, SubjectIDs, _ = load_data(args.data_test, [], [])
    channels_all = set(DataTest.keys())
    trained_chs = [bp['channel'] for bp in BestParamsAll]
    avail_mask = [ch in channels_all for ch in trained_chs]
    BestParamsAll = [bp for bp, ok in zip(BestParamsAll, avail_mask) if ok]
    channels = [bp['channel'] for bp in BestParamsAll]
    Nch = len(channels)

    totalSubs = count_subjects(DataTest)
    ScoresMatrix = np.full((totalSubs, Nch), np.nan, dtype=float)
    TrueLabels = None
    TargetVec = None

    labels_map = {}
    if args.mode == 'correlation' and args.labels_file:
        T = read_labels_table(args.labels_file)
        labels_map = dict(zip(T['ID'].astype(str), T['Target'].astype(float)))

    chan_index_map = []
    for chIdx, ch in enumerate(channels):
        P = BestParamsAll[chIdx]
        chData = DataTest[ch]

        # Build test subjects list and labels
        if len(chData['group2']) > 0:
            AllData = chData['group1'] + chData['group2']
            TL = np.concatenate([np.ones(len(chData['group1'])), np.zeros(len(chData['group2']))])
            if args.mode == 'correlation':
                grp_ids = np.concatenate([SubjectIDs[ch]['group1'], SubjectIDs[ch]['group2']])
                TargetVec = fetch_targets(grp_ids, labels_map)
        else:
            AllData = chData['group1']
            TL = np.ones(len(AllData))
            if args.mode == 'correlation':
                grp_ids = SubjectIDs[ch]['group1']
                TargetVec = fetch_targets(grp_ids, labels_map)

        if TrueLabels is None:
            TrueLabels = TL

        # Filter + LPC
        Filt = np.array([[0, P['cutoff'][0], 0], [P['cutoff'][1], np.inf, 0]], dtype=float)
        Xf = filter_data(AllData, Filt, args.Fs)
        LPC_list = Parallel(n_jobs=args.n_jobs)(delayed(_compute_lpc_dispatch)(x, int(P['order']), args.lpc_backend) for x in Xf)
        LPC_all = np.vstack([np.asarray(v)[None, :] for v in LPC_list])

        if len(chData['group2']) > 0:
            n1 = len(chData['group1'])
            P0, m0 = build_hyperplanes(LPC_all[n1:, :], int(P['dim']))
            P1, m1 = build_hyperplanes(LPC_all[:n1, :], int(P['dim']))
        else:
            P0, m0 = build_hyperplanes(LPC_all, int(P['dim']))
            P1, m1 = build_hyperplanes(LPC_all, int(P['dim']))

        s = compute_leapd_scores(LPC_all, P0, m0, P1, m1, None, int(args.is_norm_proj))
        if args.mode == 'correlation' and int(P.get('polarity', 0)) == -1:
            s = 1 - s

        ScoresMatrix[:len(s), chIdx] = s.reshape(-1)
        chan_index_map.append(ch)

    results = {'single': [], 'combos': {}}

    # Single-channel metrics
    for j in range(Nch):
        if args.mode == 'classification':
            metrics = evaluate_classification(ScoresMatrix[:, j], TrueLabels)
        else:
            metrics = evaluate_correlation(ScoresMatrix[:, j], TargetVec)
        results['single'].append({'channel': chan_index_map[j], 'metrics': metrics})

    # Evaluate combinations
    best_prev = {}
    for k in args.combo_sizes:
        if k <= args.max_full_combos:
            from itertools import combinations
            combos = np.array(list(combinations(range(Nch), k)), dtype=int)
        else:
            combos = generate_combinations(best_prev, Nch, k)

        best_k = {'indices': [], 'channels': [], 'metrics': None, 'score': -np.inf}
        for idx in combos:
            s = combine_scores(ScoresMatrix[:, idx])
            if args.mode == 'classification':
                metrics = evaluate_classification(s, TrueLabels)
                score = metrics['ACC']
            else:
                metrics = evaluate_correlation(s, TargetVec)
                score = abs(metrics['Rho']) if metrics['Rho'] == metrics['Rho'] else -np.inf
            if score > best_k['score']:
                best_k = {'indices': idx.tolist(),
                          'channels': [chan_index_map[t] for t in idx.tolist()],
                          'metrics': metrics,
                          'score': score}
        results['combos'][int(k)] = {'best': best_k}
        if k <= args.max_full_combos:
            best_prev[int(k)] = {'indices': best_k['indices']}
        if args.mode == 'classification':
            print(f"k={k} best: {', '.join(best_k['channels'])} | ACC={best_k['metrics']['ACC']:.2f}%")
        else:
            rho = best_k['metrics']['Rho']
            p = best_k['metrics']['PValue']
            print(f"k={k} best: {', '.join(best_k['channels'])} | Rho={rho:.3f} (p={p:.3g})")

    # Save
    out_path = os.path.join(args.save_dir, 'test_results.json')
    with open(out_path, 'w') as f:
        json.dump({'results': results, 'cfg': vars(args)}, f, indent=2)
    print(f"Saved test results to {out_path}")

if __name__ == '__main__':
    main()
