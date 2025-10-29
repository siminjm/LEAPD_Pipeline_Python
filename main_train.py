import argparse, json, os
from joblib import Parallel, delayed
import numpy as np
from leapd import load_data, read_labels_table, filter_data, compute_lpc_librosa, build_hyperplanes
from leapd import evaluate_classification, evaluate_correlation, pick_polarity_and_rho
from leapd import count_subjects
from itertools import product

def local_loocv_accuracy(LPC_all, Classes, d, is_norm):
    Classes = np.asarray(Classes).astype(int).reshape(-1)
    if np.unique(Classes).size < 2:
        return np.nan
    n = len(Classes)
    preds = np.zeros(n, dtype=bool)
    for i in range(n):
        tr = np.setdiff1d(np.arange(n), [i])
        c1 = tr[Classes[tr]==1]
        c0 = tr[Classes[tr]==0]
        P1, m1 = build_hyperplanes(LPC_all[c1, :], d)
        P0, m0 = build_hyperplanes(LPC_all[c0, :], d)
        from leapd import compute_leapd_scores
        s = compute_leapd_scores(LPC_all[i, :], P0, m0, P1, m1, 1, is_norm)
        preds[i] = (s >= 0.5)[0]
    return (preds == (Classes==1)).mean() * 100.0

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
    ap.add_argument('--data_train', required=True)
    ap.add_argument('--labels_file', default='')
    ap.add_argument('--channel_list', nargs='*', default=[])
    ap.add_argument('--exclude_chans', nargs='*', default=['FT9','FT10','TP9','TP10'])
    ap.add_argument('--f1_grid', nargs='*', type=float, default=[0.1,0.2,0.3,0.4,0.5])
    ap.add_argument('--f2_grid', nargs='*', type=float, default=list(range(1,101,1)))
    ap.add_argument('--orders', nargs='*', type=int, default=list(range(2,11)))
    ap.add_argument('--save_dir', default='results/train_results')
    ap.add_argument('--Fs', type=float, default=500.0)
    ap.add_argument('--lpc_backend', choices=['burg','librosa'], default='burg')
    ap.add_argument('--n_jobs', type=int, default=-1)
    ap.add_argument('--target_group', choices=['group1','group2'], default='group1')
    ap.add_argument('--kfold', default='loocv')
    ap.add_argument('--is_norm_proj', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    DataTrain, ChannelLoc, SubjectIDs, GroupNames = load_data(args.data_train, args.channel_list, args.exclude_chans)
    channels = list(DataTrain.keys())

    BestParamsAll = []
    labels_map = {}
    if args.mode == 'correlation' and args.labels_file:
        T = read_labels_table(args.labels_file)
        labels_map = dict(zip(T['ID'].astype(str), T['Target'].astype(float)))

    for ch in channels:
        chData = DataTrain[ch]
        if len(chData['group2']) > 0:
            AllData = chData['group1'] + chData['group2']
            Classes = np.concatenate([np.ones(len(chData['group1'])), np.zeros(len(chData['group2']))])
        else:
            AllData = chData['group1']
            Classes = np.ones(len(AllData))

        bestMetric = -np.inf
        bestRec = {}

        for f1 in args.f1_grid:
            for f2 in args.f2_grid:
                if (f2 - f1) < 4: 
                    continue
                Filt = np.array([[0, f1, 0], [f2, np.inf, 0]], dtype=float)
                Xf = filter_data(AllData, Filt, args.Fs)

                # Precompute LPC per order
                for ord_ in args.orders:
                    LPC_list = Parallel(n_jobs=args.n_jobs)(delayed(_compute_lpc_dispatch)(x, ord_, args.lpc_backend) for x in Xf)
                    LPC_all = np.vstack([np.asarray(v)[None, :] for v in LPC_list])

                    if args.mode == 'classification':
                        dims = list(range(1, ord_))
                        for d in dims:
                            acc = local_loocv_accuracy(LPC_all, Classes, d, args.is_norm_proj)
                            if acc > bestMetric:
                                bestMetric = acc
                                bestRec = dict(channel=ch, cutoff=[f1, f2], order=ord_, dim=d, metric=acc,
                                               mode='classification', polarity=0)
                    else:
                        # Correlation
                        if args.target_group == 'group1':
                            tgt_ids = SubjectIDs[ch]['group1']
                            tgt_idx = np.arange(len(tgt_ids))
                        else:
                            tgt_ids = SubjectIDs[ch]['group2']
                            base = len(chData['group1'])
                            tgt_idx = np.arange(base, base + len(tgt_ids))

                        y = np.array([labels_map.get(str(sid), np.nan) for sid in tgt_ids], dtype=float)
                        if np.all(np.isnan(y)) or y.size < 3:
                            continue

                        dims = list(range(1, ord_))
                        for d in dims:
                            if len(chData['group2']) > 0:
                                if args.target_group == 'group1':
                                    P_ref, m_ref = build_hyperplanes(LPC_all[len(chData['group1']):, :], d)
                                else:
                                    P_ref, m_ref = build_hyperplanes(LPC_all[:len(chData['group1']), :], d)
                            else:
                                P_ref, m_ref = build_hyperplanes(LPC_all[tgt_idx, :], d)

                            scores = np.zeros(len(tgt_idx))
                            for i, ti in enumerate(tgt_idx):
                                if args.target_group == 'group1':
                                    tr_idx = np.setdiff1d(np.arange(len(chData['group1'])), [ti])
                                    P_tar, m_tar = build_hyperplanes(LPC_all[tr_idx, :], d)
                                else:
                                    base = len(chData['group1'])
                                    loc = ti - base
                                    tr_idx = base + np.setdiff1d(np.arange(len(chData['group2'])), [loc])
                                    P_tar, m_tar = build_hyperplanes(LPC_all[tr_idx, :], d)
                                from leapd import compute_leapd_scores
                                scores[i] = compute_leapd_scores(LPC_all[ti, :], P_ref, m_ref, P_tar, m_tar, 1)[0]

                            rho, p, pol, _ = pick_polarity_and_rho(scores, y)
                            if np.isnan(rho):
                                continue
                            if abs(rho) > abs(bestMetric):
                                bestMetric = rho
                                bestRec = dict(channel=ch, cutoff=[f1, f2], order=ord_, dim=d, metric=rho,
                                               mode='correlation', polarity=int(pol))

        if bestRec:
            bestRec['GroupNames'] = [str(g) for g in GroupNames]
            bestRec['Fs'] = float(args.Fs)
            BestParamsAll.append(bestRec)

    # Save BestParamsAll to NPZ/JSON
    np.savez(os.path.join(args.save_dir, 'BestParamsAll.npz'), BestParamsAll=np.array(BestParamsAll, dtype=object))
    with open(os.path.join(args.save_dir, 'BestParamsAll.json'), 'w') as f:
        json.dump(BestParamsAll, f, indent=2)
    print(f"Saved training results to {os.path.join(args.save_dir, 'BestParamsAll.npz')}" )

if __name__ == '__main__':
    main()
