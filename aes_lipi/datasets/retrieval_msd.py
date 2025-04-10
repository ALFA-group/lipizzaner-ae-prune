# %% [markdown]
# # Autoencoders that don't overfit towards the Identity
#
# This notebook provides an implementation in Python 3.7.7 (and Tensorflow 1.15.0) of the algorithms outlined in the paper
# "Autoencoders that don't overfit towards the Identity"
# at the 34th Conference on Neural Information Processing Systems (NeurIPS 2020).
#
# For reproducibility, the experiments utilize publicly available [code](https://github.com/dawenl/vae_cf) for pre-processing three popular data-sets and for evaluating the learned models. That code accompanies the paper "[Variational autoencoders for collaborative filtering](https://arxiv.org/abs/1802.05814)" by Dawen Liang et al. at The Web Conference 2018. While the code for the Movielens-20M data-set was made publicly available, the code for pre-processing the other two data-sets can easily be obtained by modifying their code as described in their paper.
# The experiments were run on an AWS instance with 128 GB RAM and 16 vCPUs.

import os
import sys

import numpy as np
from scipy import sparse
import pandas as pd
import torch


DATA_DIR = "/Users/ehemberg/Lipi_AE/lipi_ae/data/msd"

itemId = "songId"  # for MSD data


def main():
    raw_data = pd.read_csv(
        os.path.join(DATA_DIR, "train_triplets.txt"),
        sep="\t",
        header=None,
        names=["userId", "songId", "playCount"],
    )

    raw_data, user_activity, item_popularity = filter_triplets(
        raw_data, min_uc=20, min_sc=200
    )  # for MSD data
    print(user_activity, item_popularity)

    sparsity = (
        1.0 * raw_data.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])
    )

    print(
        "After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)"
        % (
            raw_data.shape[0],
            user_activity.shape[0],
            item_popularity.shape[0],
            sparsity * 100,
        )
    )

    unique_uid = user_activity.index

    # ## Pre-processing of the Data
    #
    # Utilizing the publicly available [code](https://github.com/dawenl/vae_cf), which is copied below (with kind permission of Dawen Liang). Note that the following code is modified as to pre-process the [MSD data-set](https://labrosa.ee.columbia.edu/millionsong/tasteprofile). For pre-processing the [MovieLens-20M data-set](https://grouplens.org/datasets/movielens/20m/), see their original publicly-available [code](https://github.com/dawenl/vae_cf).
    #
    # ### Data splitting procedure
    # - Select 50K users as heldout users, 50K users as validation users, and the rest of the users for training
    # - Use all the items from the training users as item set
    # - For each of both validation and test user, subsample 80% as fold-in data and the rest for prediction

    np.random.seed(98765)
    idx_perm = np.random.permutation(unique_uid.size)
    unique_uid = unique_uid[idx_perm]

    # create train/validation/test users
    n_users = unique_uid.size
    n_heldout_users = 50000  # for MSD data

    tr_users = unique_uid[: (n_users - n_heldout_users * 2)]
    vd_users = unique_uid[(n_users - n_heldout_users * 2) : (n_users - n_heldout_users)]
    te_users = unique_uid[(n_users - n_heldout_users) :]

    train_plays = raw_data.loc[raw_data["userId"].isin(tr_users)]

    unique_sid = pd.unique(train_plays[itemId])

    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

    pro_dir = os.path.join(DATA_DIR, "pro_sg")

    if not os.path.exists(pro_dir):
        os.makedirs(pro_dir)

    with open(os.path.join(pro_dir, "unique_sid.txt"), "w") as f:
        for sid in unique_sid:
            f.write("%s\n" % sid)

    vad_plays = raw_data.loc[raw_data["userId"].isin(vd_users)]
    vad_plays = vad_plays.loc[vad_plays[itemId].isin(unique_sid)]

    vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)

    test_plays = raw_data.loc[raw_data["userId"].isin(te_users)]
    test_plays = test_plays.loc[test_plays[itemId].isin(unique_sid)]

    test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)

    # ### Save the data into (user_index, item_index) format

    train_data = numerize(train_plays, profile2id=profile2id, show2id=show2id)
    train_data.to_csv(os.path.join(pro_dir, "train.csv"), index=False)

    vad_data_tr = numerize(vad_plays_tr, profile2id=profile2id, show2id=show2id)
    vad_data_tr.to_csv(os.path.join(pro_dir, "validation_tr.csv"), index=False)

    vad_data_te = numerize(vad_plays_te, profile2id=profile2id, show2id=show2id)
    vad_data_te.to_csv(os.path.join(pro_dir, "validation_te.csv"), index=False)

    test_data_tr = numerize(test_plays_tr, profile2id=profile2id, show2id=show2id)
    test_data_tr.to_csv(os.path.join(pro_dir, "test_tr.csv"), index=False)

    test_data_te = numerize(test_plays_te, profile2id=profile2id, show2id=show2id)
    test_data_te.to_csv(os.path.join(pro_dir, "test_te.csv"), index=False)

    # ## Load the pre-processed training and test data

    unique_sid = get_unique_sid(pro_dir)
    n_items = len(unique_sid)

    # load training data
    train_data = load_train_data(os.path.join(pro_dir, "train.csv"), n_items=n_items)

    test_data_tr, test_data_te = load_tr_te_data(
        os.path.join(pro_dir, "test_tr.csv"),
        os.path.join(pro_dir, "test_te.csv"),
        n_items=n_items,
    )


def get_unique_sid(pro_dir):
    unique_sid = list()
    with open(os.path.join(pro_dir, "unique_sid.txt"), "r") as f:
        for line in f:
            unique_sid.append(line.strip())

    return unique_sid


def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=True)
    count = playcount_groupbyid.size()
    return count


def filter_triplets(tp, min_uc=5, min_sc=0):
    # Only keep the triplets for items which were clicked on by at least min_sc users.
    if min_sc > 0:
        itemcount = get_count(tp, itemId)
        print(itemcount.head(2))
        tp = tp[tp[itemId].isin(itemcount.index[itemcount >= min_sc])]

    # Only keep the triplets for users who clicked on at least min_uc items
    # After doing this, some of the items will have less than min_uc users, but should only be a small proportion
    if min_uc > 0:
        usercount = get_count(tp, "userId")
        tp = tp[tp["userId"].isin(usercount.index[usercount >= min_uc])]

    # Update both usercount and itemcount after filtering
    usercount, itemcount = get_count(tp, "userId"), get_count(tp, itemId)
    return tp, usercount, itemcount


def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby("userId")
    tr_list, te_list = list(), list()

    np.random.seed(98765)

    for i, (_, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype="bool")
            idx[
                np.random.choice(
                    n_items_u, size=int(test_prop * n_items_u), replace=False
                ).astype("int64")
            ] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        else:
            tr_list.append(group)

        if i % 5000 == 0:
            print("%d users sampled" % i)
            sys.stdout.flush()

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te


def numerize(tp, profile2id, show2id):
    uid = map(lambda x: profile2id[x], tp["userId"])
    sid = map(lambda x: show2id[x], tp[itemId])
    return pd.DataFrame(
        data={"uid": list(uid), "sid": list(sid)}, columns=["uid", "sid"]
    )


def load_train_data(csv_file, n_items, sparse_m: bool = True):
    tp = pd.read_csv(csv_file)
    n_users = tp["uid"].max() + 1

    rows, cols = tp["uid"].to_numpy(), tp["sid"].to_numpy()
    data = sparse.csr_matrix(
        (np.ones_like(rows), (rows, cols)), dtype="float64", shape=(n_users, n_items)
    )
    if not sparse_m:
        data = torch.sparse_coo_tensor(data.nonzero(), data.data, data.shape)
        data = data.to_dense().to(torch.float32)

    return data


def load_tr_te_data(csv_file_tr, csv_file_te, n_items):
    tp_tr = pd.read_csv(csv_file_tr)
    tp_te = pd.read_csv(csv_file_te)

    start_idx = min(tp_tr["uid"].min(), tp_te["uid"].min())
    end_idx = max(tp_tr["uid"].max(), tp_te["uid"].max())

    rows_tr, cols_tr = tp_tr["uid"] - start_idx, tp_tr["sid"]
    rows_te, cols_te = tp_te["uid"] - start_idx, tp_te["sid"]

    data_tr = sparse.csr_matrix(
        (np.ones_like(rows_tr), (rows_tr, cols_tr)),
        dtype="float64",
        shape=(end_idx - start_idx + 1, n_items),
    )
    data_te = sparse.csr_matrix(
        (np.ones_like(rows_te), (rows_te, cols_te)),
        dtype="float64",
        shape=(end_idx - start_idx + 1, n_items),
    )
    return data_tr, data_te


def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100):
    """
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    """
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1.0 / np.log2(np.arange(2, k + 2))

    DCG = (
        heldout_batch[np.arange(batch_users)[:, np.newaxis], idx_topk].toarray() * tp
    ).sum(axis=1)
    IDCG = np.array([(tp[: min(n, k)]).sum() for n in heldout_batch.getnnz(axis=1)])
    return DCG / IDCG


def Recall_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall


def evaluate(BB, test_data_tr, test_data_te):
    print("evaluating ...")
    N_test = test_data_tr.shape[0]
    idxlist_test = range(N_test)

    batch_size_test = 5000
    n100_list, r20_list, r50_list = [], [], []
    for bnum, st_idx in enumerate(range(0, N_test, batch_size_test)):
        end_idx = min(st_idx + batch_size_test, N_test)
        X = test_data_tr[idxlist_test[st_idx:end_idx]]

        if sparse.isspmatrix(X):
            X = X.toarray()
        X = X.astype("float32")

        pred_val = X.dot(BB)
        # exclude examples from training and validation (if any)
        pred_val[X.nonzero()] = -np.inf
        n100_list.append(
            NDCG_binary_at_k_batch(
                pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=100
            )
        )
        r20_list.append(
            Recall_at_k_batch(
                pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=20
            )
        )
        r50_list.append(
            Recall_at_k_batch(
                pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=50
            )
        )

    n100_list = np.concatenate(n100_list)
    r20_list = np.concatenate(r20_list)
    r50_list = np.concatenate(r50_list)
    print(
        "Test NDCG@100=%.5f (%.5f)"
        % (np.mean(n100_list), np.std(n100_list) / np.sqrt(len(n100_list)))
    )
    print(
        "Test Recall@20=%.5f (%.5f)"
        % (np.mean(r20_list), np.std(r20_list) / np.sqrt(len(r20_list)))
    )
    print(
        "Test Recall@50=%.5f (%.5f)"
        % (np.mean(r50_list), np.std(r50_list) / np.sqrt(len(r50_list)))
    )


if __name__ == "__main__":
    main()
