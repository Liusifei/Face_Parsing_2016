#
#   util_eval.py
#
#   Author: Huang Chang (huangchang@baidu.com)
#   Created on Feb. 10, 2014
#
import os, sys, time
import numpy as np
import mpi4py.MPI as MPI
#from Common import LDOT, LEculidean

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()


def my_Allgather(comm, local_data, data=None):
    if not isinstance(local_data, np.ndarray):
        local_data = np.array(local_data)

    if data==None:
        data = np.empty(local_data.size*comm.Get_size(), dtype=local_data.dtype)
    else:
        assert(data.size==local_data.size*comm.Get_size() and data.dtype==local_data.dtype)

    comm.Allgather(local_data, data)
    return data

def my_Allgatherv(local_data, comm=comm, data=None):
    if not isinstance(local_data, np.ndarray):
        local_data = np.array(local_data)

    sz_arr = my_Allgather(comm, local_data.size)
    os_arr = np.empty_like(sz_arr)
    os_arr[0] = 0; os_arr[1:] = sz_arr[:-1].cumsum()

    if data==None:
        data = np.empty(sz_arr.sum(), dtype=local_data.dtype)
    else:
        assert(data.size==sz_arr.sum() and data.dtype==local_data.dtype)

    comm.Allgatherv(local_data, (data, (sz_arr, os_arr)))
    return data.reshape(-1, *local_data.shape[1:])

def LDOT(feat, proj):
    coord = np.empty((feat.shape[0], proj.shape[1]), dtype=proj.dtype)
    item_size = max(feat.dtype.itemsize, proj.itemsize)
    chunk_sz = (1<<30)/item_size/feat.shape[1]
    for si in range(0, feat.shape[0], chunk_sz):
        ei = min(feat.shape[0], si+chunk_sz)
        coord[si:ei] = np.dot(feat[si:ei], proj)
    return coord


def LEculidean(qry, ref):
    qry_sp = (qry**2).sum(axis=1).reshape(qry.shape[0],1)
    ref_sp = (ref**2).sum(axis=1).reshape(1,ref.shape[0])
    dist = -2*LDOT(qry, ref.T)
    dist += qry_sp
    dist += ref_sp
    return dist

def RetrieveTopN(N, qry_feat, ref_feat, L, dist=None):
    #compute dist
    if dist==None:
        dist = LEculidean(qry_feat, ref_feat)
    indices = np.argsort(dist)
    top_indices = indices[:, :N]
    top_dists = []
    for d, i in zip(dist, top_indices):
        top_dists.append(d[i])
    return top_indices, np.vstack(top_dists)
    
def CalSoftMinAccuracy_MPI(local_qry_feat, local_qry_label, ref_feat, ref_label, gamma_list, top_hit, dist=None, fid=None):
    #compute dist
    if dist==None:
        dist = LEculidean(local_qry_feat, ref_feat)

    #get offsets of categories
    ref_cate_num = int(ref_label.max()+1)
    dl = ref_label[1:] - ref_label[:-1]
    if dl.max()!=1 or dl.min()<0 or ref_label[0]!=0 or ref_cate_num!=dl.sum()+1:
        print dl.max(), dl.min(), ref_label[0], ref_cate_num, dl.sum()+1
        raise ValueError("Ref Label list is not valid")
    ref_cate_offset = np.hstack([0, np.where(dl==1)[0] + 1, ref_label.size]).astype('int32')

    local_qry_num = local_qry_feat.shape[0]
    qry_num = comm.allreduce(local_qry_num, op=MPI.SUM)
    top_hit_rate_report = {}

    for gamma in gamma_list:
        hit_rank = np.zeros(ref_cate_num, dtype='int')
        dist2cate = np.empty((local_qry_num, ref_cate_num))
        if gamma==0:
            #hard min
            for ci in range(ref_cate_num):
                dist2cate[:,ci] = dist[:,ref_cate_offset[ci]:ref_cate_offset[ci+1]].min(axis=1)
        else:
            #soft min
            for ci in range(ref_cate_num):
                si = ref_cate_offset[ci]
                ei = ref_cate_offset[ci+1]
                #dist2cate[:,ci] = -np.log((np.exp(-gamma*dist[:,si:ei])).sum(axis=1))
                dist2cate[:,ci] = -(np.exp(-gamma*dist[:,si:ei])).sum(axis=1)

        for qry_i in range(local_qry_num):
            d2c = dist2cate[qry_i]
            cor_dist = d2c[local_qry_label[qry_i]]
            rank = (d2c<cor_dist).sum()
            hit_rank[rank] += 1

        max_dist = dist.max()
        sum_dist = dist.sum()

        hit_rank = comm.allreduce(hit_rank, op=MPI.SUM)
        max_dist = comm.allreduce(max_dist, op=MPI.MAX)
        sum_dist = comm.allreduce(sum_dist, op=MPI.SUM)
        avg_dist = sum_dist/(qry_num*ref_label.size)

        hit_rank = hit_rank[:top_hit]
        top_hit_rate = hit_rank.cumsum().astype('float')/qry_num

        log = "{:.3G}-MIN, {}x{}: avg-dist {:.4G} max-dist {:.4G}\n\t".format(gamma, qry_num, ref_label.size, avg_dist, max_dist)
        for idx, hit in enumerate(top_hit_rate):
            log += '{}:{:.5G} '.format(idx, hit)
        #log += '\n\t'
        #for idx, hit in enumerate(hit_rank.cumsum()):
        #    log += '{}:{} '.format(idx, hit)
        if comm_rank==0:
            print log
            if fid!=None:
                print>>fid, log
        top_hit_rate_report['gamma={}'.format(gamma)] = top_hit_rate

    return top_hit_rate_report



def CalSoftMaxPredAccuracy_MPI(local_pred, local_label, cate_num, prototype_per_cate, top_hit, fid=None):
    local_qry_num, dim = local_pred.shape
    assert dim==cate_num*prototype_per_cate

    nx = local_pred - local_pred.max(axis=1).reshape(local_qry_num, 1)
    enx = np.exp(nx)
    senx = enx.reshape(local_qry_num, cate_num, prototype_per_cate).sum(axis=2)

    local_hit_rank = np.zeros(top_hit, dtype='int')
    for qry_i in range(local_qry_num):
        cor_pred = senx[qry_i, local_label[qry_i]]
        rank = (cor_pred<senx[qry_i]).sum()
        if rank<top_hit:
            local_hit_rank[rank] += 1

    hit_rank = comm.reduce(local_hit_rank, root=0, op=MPI.SUM)
    qry_num = comm.reduce(local_qry_num, root=0, op=MPI.SUM)
    if comm_rank==0:
        top_hit_rate = hit_rank.cumsum().astype('float')/qry_num
        log = "\nSoftMax Pred, {}x{}x{}: \n\t".format(qry_num, cate_num, prototype_per_cate)
        for idx, hit in enumerate(top_hit_rate):
            log += '{}:{:.5G} '.format(idx, hit)
        print log
        if fid!=None:
            print>>fid, log






def CalTheKthNNAccuracy_MPI(local_qry_feat, local_qry_label, ref_feat, ref_label, k_list, top_hit, dist=None, fid=None):
    #compute dist
    if dist==None:
        dist = LEculidean(local_qry_feat, ref_feat)

    #get offsets of categories
    ref_cate_num = int(ref_label.max()+1)
    dl = ref_label[1:] - ref_label[:-1]
    if dl.max()!=1 or dl.min()<0 or ref_label[0]!=0 or ref_cate_num!=dl.sum()+1:
        print dl.max(), dl.min(), ref_label[0], ref_cate_num, dl.sum()+1
        raise ValueError("Ref Label list is not valid")
    ref_cate_offset = np.hstack([0, np.where(dl==1)[0] + 1, ref_label.size]).astype('int32')

    local_qry_num = local_qry_feat.shape[0]
    qry_num = comm.allreduce(local_qry_num, op=MPI.SUM)
    top_hit_rate_report = {}

    for k in k_list:
        hit_rank = np.zeros(ref_cate_num, dtype='int')
        dist2cate = np.empty((local_qry_num, ref_cate_num))
        for ci in range(ref_cate_num):
            dist2cate[:,ci] = np.sort(dist[:,ref_cate_offset[ci]:ref_cate_offset[ci+1]], axis=1)[:,k]

        for qry_i in range(local_qry_num):
            d2c = dist2cate[qry_i]
            cor_dist = d2c[local_qry_label[qry_i]]
            rank = (d2c<cor_dist).sum()
            hit_rank[rank] += 1

        max_dist = dist.max()
        sum_dist = dist.sum()

        hit_rank = comm.allreduce(hit_rank, op=MPI.SUM)
        max_dist = comm.allreduce(max_dist, op=MPI.MAX)
        sum_dist = comm.allreduce(sum_dist, op=MPI.SUM)
        avg_dist = sum_dist/(qry_num*ref_label.size)

        hit_rank = hit_rank[:top_hit]
        top_hit_rate = hit_rank.cumsum().astype('float')/qry_num

        log = "{}th-MIN, {}x{}: avg-dist {:.4G} max-dist {:.4G}\n\t".format(k, qry_num, ref_label.size, avg_dist, max_dist)
        for idx, hit in enumerate(top_hit_rate):
            log += '{}:{:.5G} '.format(idx, hit)
        #log += '\n\t'
        #for idx, hit in enumerate(hit_rank.cumsum()):
        #    log += '{}:{} '.format(idx, hit)
        if comm_rank==0:
            print log
            if fid!=None:
                print>>fid, log
        top_hit_rate_report['k={}'.format(k)] = top_hit_rate

    return top_hit_rate_report



def CalMeanAvgPrecNDCG_MPI(local_qry_feat, local_qry_label, ref_feat, ref_label, dist=None, fid=None):
    if dist==None:
        dist = LEculidean(local_qry_feat, ref_feat)
    cate_num = ref_label.max()+1
    local_qry_num, ref_num = dist.shape

    avg_prec_list = []
    nDCG_list = []
    for lqi in range(local_qry_num):
        qry_label = local_qry_label[lqi]
        dist_row = dist[lqi,:]
        sort_idx = np.argsort(dist_row)
        sorted_label = ref_label[sort_idx]
        target_pos = np.where(sorted_label==qry_label)[0]

        avg_prec = (np.arange(1,target_pos.size+1).astype('float')/(target_pos+1)).mean()
        avg_prec_list.append(avg_prec)

        DCG = (np.log(2.0)/np.log(target_pos+1+1)).sum()
        IDCG = (np.log(2.0)/np.log(np.arange(1,target_pos.size+1)+1)).sum()
        nDCG = DCG/IDCG
        nDCG_list.append(nDCG)

    local_avg_prec_sum = np.array(avg_prec_list).sum()
    local_nDCG_sum = np.array(nDCG_list).sum()

    avg_prec_sum = comm.allreduce(local_avg_prec_sum, op=MPI.SUM)
    nDCG_sum = comm.allreduce(local_nDCG_sum, op=MPI.SUM)
    qry_sum = comm.allreduce(local_qry_num, op=MPI.SUM)
    mean_avg_prec = avg_prec_sum/qry_sum
    mean_nDCG = nDCG_sum/qry_sum

    log = 'mean AvgPrec {:.6f}, mean NDCG {:.6f}'.format(mean_avg_prec, mean_nDCG)
    if comm_rank == 0:
        print log
    if fid != None:
        print>>fid, log

    CalDistSTD_MPI(dist, fid)

    return mean_avg_prec, mean_nDCG


def CalAccuracyTopN_MPI(local_qry_coord, local_qry_label, ref_coord, ref_label, topN_l, gamma_l=None, mAP_NDCG=True, leave_one_out=False, fid=None):
    #from MetricLearningCPP import RetrieveTopN
    N = max(topN_l)
    if leave_one_out:
        N += 1
    indices, dist = RetrieveTopN(N, local_qry_coord.astype('float'), ref_coord.astype('float'), np.empty(0))
    if leave_one_out:
        indices = indices[:, 1:]
        dist = dist[:, 1:]
    total_recall_num = np.array([(ref_label==y).sum() for y in local_qry_label])
    if leave_one_out:
        total_recall_num = np.maximum(total_recall_num-1, 1)
    top_labels = ref_label[indices]
    LQN = local_qry_label.size
    QN = comm.allreduce(LQN, op=MPI.SUM)
    log = 'TopK accuracy and hit rate: qry num {}, ref num {}, ref cate num {}\n'.format(QN, ref_label.size, np.unique(ref_label).size)
    for topN in topN_l:
        local_correct_mat = np.equal(local_qry_label.reshape(LQN, 1), top_labels[:,:topN])
        local_correct_num = local_correct_mat.sum()
        local_recall_rate = local_correct_mat.sum(axis=1).astype('float') / total_recall_num
        local_hit_num = local_correct_mat.max(axis=1).sum()
        correct_num = comm.allreduce(local_correct_num, op=MPI.SUM)
        recall_rate_sum = comm.allreduce(local_recall_rate.sum(), op=MPI.SUM)
        hit_num = comm.allreduce(local_hit_num, op=MPI.SUM)
        log += 'Top-{}: accuracy {:.4f}, hit rate {:.4f}, recall rate {:.4f}\n'.format(topN, float(correct_num)/(QN*topN), float(hit_num)/QN, float(recall_rate_sum)/QN)
    if comm_rank==0:
        print log,
        if fid!=None:
            print>>fid, log,

    if gamma_l != None:
        K = 5
        lqn = local_qry_coord.shape[0]
        local_hit_rank = np.empty((lqn, len(gamma_l), 2))
        gamma_arr = np.array(gamma_l)
        for topN in topN_l:
            for qi in xrange(lqn):
                d = dist[qi, :topN] - dist[qi, :topN].min()
                gd = - np.exp(-gamma_arr.reshape(1,-1)*d.reshape(-1,1))
                d2c = {}
                for ri in xrange(d.size):
                    rlabel = ref_label[indices[qi,ri]]
                    if rlabel in d2c:
                        d2c[rlabel] += gd[ri]
                    else:
                        d2c[rlabel] = gd[ri]

                d2c_label = d2c.keys()
                qlabel = local_qry_label[qi]
                if qlabel not in d2c_label:
                    local_hit_rank[qi] = 1E6
                else:
                    d2c_arr = np.vstack([d2c[rlabel] for rlabel in d2c_label])
                    cor_d = d2c_arr[d2c_label.index(qlabel)]
                    local_hit_rank[qi, :, 0] = (d2c_arr<cor_d).sum(axis=0)
                    local_hit_rank[qi, :, 1] = (d2c_arr<=cor_d).sum(axis=0) - 1

            local_err_num_gamma = np.vstack([0.5*((local_hit_rank[:,:,0]>k).sum(axis=0) + (local_hit_rank[:,:,1]>k).sum(axis=0)) for k in xrange(K)])
            err_num_gamma = comm.reduce(local_err_num_gamma, root=0, op=MPI.SUM)
            qn = comm.reduce(lqn, root=0, op=MPI.SUM)
            if comm_rank == 0:
                log = '\nSoftMin NN: {}x{}, with top-{} NN\n'.format(qn, ref_coord.shape[0], topN)
                log += 'Gamma\t'
                for k in xrange(K):
                    log += 'Top-{}\t'.format(k+1)
                log += '\n'
                err_rate_gamma = err_num_gamma.astype('float') / qn
                for gi, gamma in enumerate(gamma_l):
                    log += '{}\t'.format(gamma)
                    for k in xrange(K):
                        log += '{:.5f}\t'.format(err_rate_gamma[k, gi])
                    log += '\n'
                print log
                if fid != None:
                    print>>fid, log

    if mAP_NDCG:
        local_qry_num = local_qry_coord.shape[0]
        avg_prec_list = []
        nDCG_list = []
        for lqi in xrange(local_qry_num):
            qry_label = local_qry_label[lqi]
            sorted_label = ref_label[indices[lqi]]
            target_pos = np.where(sorted_label==qry_label)[0]
            pos_num = (ref_label==qry_label).sum()

            avg_prec = (np.arange(1,target_pos.size+1).astype('float')/(target_pos+1)).sum()/pos_num
            avg_prec_list.append(avg_prec)

            DCG = (np.log(2.0)/np.log(target_pos+1+1)).sum()
            IDCG = (np.log(2.0)/np.log(np.arange(1,min(N, pos_num)+1)+1)).sum()
            nDCG = DCG/IDCG
            nDCG_list.append(nDCG)

        local_avg_prec_sum = np.array(avg_prec_list).sum()
        local_nDCG_sum = np.array(nDCG_list).sum()

        avg_prec_sum = comm.allreduce(local_avg_prec_sum, op=MPI.SUM)
        nDCG_sum = comm.allreduce(local_nDCG_sum, op=MPI.SUM)
        qry_sum = comm.allreduce(local_qry_num, op=MPI.SUM)
        mean_avg_prec = avg_prec_sum/qry_sum
        mean_nDCG = nDCG_sum/qry_sum

        log = 'Top-{}: mean AvgPrec {:.6f}, mean NDCG {:.6f}'.format(N, mean_avg_prec, mean_nDCG)
        if comm_rank == 0:
            print log
        if fid != None:
            print>>fid, log



def CalSoftMaxErrorMultiPrototype_MPI(x, y, cate_num, prototype_per_cate, topK_l, fid=None):
    assert x.shape[1]==cate_num*prototype_per_cate
    num, dim = x.shape
    nx = x - x.max(axis=1).reshape(num, 1)
    enx = np.exp(nx)
    enx = enx.reshape(enx.shape[0], cate_num, prototype_per_cate)
    enx_cate_sum = enx.sum(axis=2)
    thres = np.empty((num,1), dtype=x.dtype)
    for i in xrange(num):
        thres[i] = enx_cate_sum[i, y[i]]
    hit_rank = (enx_cate_sum >= thres).sum(axis=1)

    total_num = comm.reduce(num, root=0, op=MPI.SUM)
    dx = x.astype('double')
    log = 'SoftMaxError: cate_num {}, prototype_per_cate {}, total_smp_num {}; dist {:.5f}+/{:.5f}, {:.5f}~{:.5f}\n'.format(cate_num, prototype_per_cate, total_num, dx.mean(), dx.std(), dx.min(), dx.max())
    for topK in topK_l:
        err_num = (hit_rank > topK).sum()
        total_err_num = comm.reduce(err_num, root=0, op=MPI.SUM)
        if comm_rank == 0:
            log += '\t{}:{:.5f}'.format(topK, float(total_err_num)/total_num)
    log += '\n'

    if fid != None:
        print>>fid, log
    if comm_rank == 0:
        print log




def CalKNNError_MPI(local_qry_feat, local_qry_label, ref_feat, ref_label, K, dist=None, leave_one_out=False, fid=None):
    if dist==None:
        dist = LEculidean(local_qry_feat, ref_feat)
    else:
        dist = np.array(dist)

    lqn = local_qry_feat.shape[0]
    if leave_one_out:
        lq_offset = np.hstack([0, np.array(comm.allgather(lqn)).cumsum()])
        assert lq_offset[-1]==ref_feat.shape[0]
        lq_si = lq_offset[comm_rank]
        for lqi in range(lqn):
            dist[lqi, lqi+lq_si] = 1E20

    local_NN_l = []
    for k in range(K):
        local_NN = np.argmin(dist, axis=1).reshape(lqn,1)
        for i, j in enumerate(local_NN):
            dist[i,j] = 1E20
        local_NN_l.append(local_NN)
    local_NN = np.hstack(local_NN_l)
    local_NN_label = ref_label[local_NN]

    local_match_idx_l = []
    for k in range(K):
        local_match_idx = (local_NN_label[:,k:k+1]==local_NN_label).sum(axis=1).reshape(lqn,1)
        local_match_idx_l.append(local_match_idx)

    local_match_idx = np.hstack(local_match_idx_l)
    local_max = local_match_idx.argmax(axis=1)
    local_pred_label = np.array([local_NN_label[i, j] for i, j in enumerate(local_max)])
    local_err_num = (local_pred_label!=local_qry_label).sum()

    err_num = comm.allreduce(local_err_num, op=MPI.SUM)
    qry_num = comm.allreduce(lqn, op=MPI.SUM)
    err_rate = float(err_num)/qry_num

    log = "{}-LeaveOneOutKNN, err {:.6G} {}/{}".format(K, err_rate, err_num, qry_num)
    if comm_rank==0:
        print log
    if fid:
        print>>fid, log

    return err_rate


def CalDistSTD_MPI(local_dist, fid=None):
    local_dist = local_dist.astype('float')
    if local_dist.size==0:
        local_num = local_sum = local_sum2 = 0
        local_min = 1E20
        local_max = 0
    else:
        local_num = local_dist.size
        local_sum = local_dist.sum()
        local_sum2 = (local_dist**2).sum()
        local_min = local_dist.min()
        local_max = local_dist.max()
    qry_num = comm.allreduce(local_dist.shape[0], op=MPI.SUM)
    num = comm.allreduce(local_num, op=MPI.SUM)
    sum_ = comm.allreduce(local_sum, op=MPI.SUM)
    sum2 = comm.allreduce(local_sum2, op=MPI.SUM)
    min_ = comm.allreduce(local_min, op=MPI.MIN)
    max_ = comm.allreduce(local_max, op=MPI.MAX)

    avg = sum_/num
    std = np.sqrt(sum2/num - avg**2)

    log = 'Distance: {}x{}, {:.4f}+-{:.4f}, min {:.4f}, max{:.4f}'.format(qry_num, local_dist.shape[1], avg, std, min_, max_)
    if comm_rank==0:
        print log
    if fid!=None:
        print>>fid, log




def CalClassificationError_MPI(local_qry_feat, local_qry_label, ref_feat, ref_label, threshold_l, dist=None, fid=None):

    if dist==None:
        dist = LEculidean(local_qry_feat, ref_feat)
    else:
        assert dist.shape == (local_qry_feat.shape[0], ref_feat.shape[0])

    local_qry_num = local_qry_feat.shape[0]
    ref_num = ref_feat.shape[0]
    local_pos_err_num_arr = np.zeros(len(threshold_l))
    local_neg_err_num_arr = np.zeros(len(threshold_l))
    local_pos_num = local_neg_num = 0
    for lqi in range(local_qry_num):
        pos_indices = np.where(ref_label==local_qry_label[lqi])[0]
        neg_indices = np.where(ref_label!=local_qry_label[lqi])[0]
        pos_dist = dist[lqi][pos_indices]
        neg_dist = dist[lqi][neg_indices]
        local_pos_num += pos_indices.size
        local_neg_num += neg_indices.size
        for idx, thres in enumerate(threshold_l):
            local_pos_err_num_arr[idx] += (pos_dist>=thres).sum()
            local_neg_err_num_arr[idx] += (neg_dist<thres).sum()
    pos_err_num_arr = comm.allreduce(local_pos_err_num_arr, op=MPI.SUM)
    neg_err_num_arr = comm.allreduce(local_neg_err_num_arr, op=MPI.SUM)
    pos_num = comm.allreduce(local_pos_num, op=MPI.SUM)
    neg_num = comm.allreduce(local_neg_num, op=MPI.SUM)

    pos_err_rate_arr = pos_err_num_arr.astype('float')/pos_num
    neg_err_rate_arr = neg_err_num_arr.astype('float')/neg_num
    log = 'pos pair num {}, neg pair num {}\n'.format(pos_num, neg_num)
    for idx, thres in enumerate(threshold_l):
        log += 'Classification error with threshold {:.4f}: false_pos {:.4f}, false_neg {:.4f}, false_avg {:.4f}\n'.format(thres, pos_err_rate_arr[idx], neg_err_rate_arr[idx], 0.5*(pos_err_rate_arr[idx]+neg_err_rate_arr[idx]))

    if comm_rank==0:
        print log

    if fid!=None:
        print>>fid, log

    return pos_err_rate_arr, neg_err_rate_arr



def InterpolateROC(ROC, false_positive=None, false_negative=None):
    if false_positive!=None:
        assert false_negative==None
        right = (ROC[:,1]<=false_positive).argmax()
        left = right - 1
        assert false_positive>=ROC[right,1] and false_positive<=ROC[left,1]
        if false_positive==ROC[left,1]:
            return ROC[left,0]
        w1 = ROC[left,1]-false_positive
        w2 = false_positive-ROC[right,1]
        return (w2*ROC[left,0] + w1*ROC[right,0])/(w1+w2)
    else:
        assert false_negative!=None
        right = (ROC[:,0]>false_negative).argmax()
        left = right - 1
        assert false_negative<=ROC[right,0] and false_negative>=ROC[left,0]
        if false_negative==ROC[left,0]:
            return ROC[left,1]
        w1 = false_negative - ROC[left,0]
        w2 = ROC[right,0] - false_negative
        return (w2*ROC[left,1] + w1*ROC[right,1])/(w1+w2)


def CalROC_MPI(local_pos_conf, local_neg_conf, interesting_false_positive_l=[], interesting_false_negative_l=[], bins=10000, delta_thres=0.01):

    local_min_conf = min(local_pos_conf.min(), local_neg_conf.min())
    local_max_conf = max(local_pos_conf.max(), local_neg_conf.max())
    min_conf = comm.allreduce(local_min_conf, op=MPI.MIN)
    max_conf = comm.allreduce(local_max_conf, op=MPI.MAX)

    local_pos_conf_sum = local_pos_conf.sum()
    local_pos_conf_sum2 = (local_pos_conf**2).sum()
    local_pos_num = local_pos_conf.size
    local_neg_conf_sum = local_neg_conf.sum()
    local_neg_conf_sum2 = (local_neg_conf**2).sum()
    local_neg_num = local_neg_conf.size

    local_pos_hist, thres_arr = np.histogram(local_pos_conf, bins=bins, range=(min_conf, max_conf))
    local_neg_hist, thres_arr = np.histogram(local_neg_conf, bins=bins, range=(min_conf, max_conf))
    pos_hist = comm.reduce(local_pos_hist, root=0, op=MPI.SUM)
    neg_hist = comm.reduce(local_neg_hist, root=0, op=MPI.SUM)
    items = comm.reduce(np.array([local_pos_num, local_pos_conf_sum, local_pos_conf_sum2, local_neg_num, local_neg_conf_sum, local_neg_conf_sum2], dtype='float'), root=0, op=MPI.SUM)
    info = ''
    if comm_rank==0:
        pos_num, pos_conf_sum, pos_conf_sum2, neg_num, neg_conf_sum, neg_conf_sum2 = items
        pos_mean = pos_conf_sum/pos_num
        pos_std = pos_conf_sum2/pos_num - pos_mean**2
        neg_mean = neg_conf_sum/neg_num
        neg_std = neg_conf_sum2/neg_num - neg_mean**2
        info += 'conf statistics: pos {:.0f}, {:.4f}+-{:.4f}; neg {:.0f}, {:.4f}+-{:.4f}\n'.format(pos_num, pos_mean, pos_std, neg_num, neg_mean, neg_std)

        auc = 0
        norm_pos_hist = pos_hist.astype('float')/pos_hist.sum()
        norm_neg_hist = neg_hist.astype('float')/neg_hist.sum()
        pos_err_rate = np.hstack([0, norm_pos_hist.cumsum()])
        neg_err_rate = 1.0 - np.hstack([0, norm_neg_hist.cumsum()])
        prev_pe = 0
        prev_ne = 1
        prev_auc_pe = 0
        prev_auc_ne = 1
        min_err = 1
        min_pne_diff = 1
        ROC_l = [(prev_pe, prev_ne)]
        buf = ''
        for i in range(bins+1):
            thres = thres_arr[i]
            pe = pos_err_rate[i]
            ne = neg_err_rate[i]
            delta = abs(pe-prev_pe) + abs(ne-prev_ne)
            if delta>delta_thres:
                buf += '{:.4f}:{:.4f}:{:.4f}  '.format(pe, ne, thres)
                prev_pe = pe
                prev_ne = ne
            err = 0.5*(pe+ne)
            if err<min_err:
                min_err = err
            pne_diff = abs(pe-ne)
            if pne_diff<min_pne_diff:
                min_pne_diff = pne_diff
                eer = err
            #cal auc
            auc += (pe-prev_auc_pe)*(ne+prev_auc_ne)*0.5

            prev_auc_pe = pe
            prev_auc_ne = ne
            ROC_l.append((pe, ne))
        if len(buf)>0:
            info += buf+'\n'

        ROC = np.array(ROC_l)
        for ifp in interesting_false_positive_l:
            fn = InterpolateROC(ROC, false_positive=ifp)
            info += '{:.4f}:{:.4f} '.format(fn, ifp)
        if len(interesting_false_negative_l)>0:
            info += '\n'
        for ifn in interesting_false_negative_l:
            fp = InterpolateROC(ROC, false_negative=ifn)
            info += '{:.4f}:{:.4f} '.format(ifn, fp)
        if len(interesting_false_negative_l)>0:
            info += '\n'

        auc = 1.0-auc
        info += 'min err {:.4f}, err {:.4f}, auc {:.4f}\n'.format(min_err, eer, auc)

    return info



def CalROC_GivenCoord_MPI(local_tst_coord, local_tst_label, interesting_false_positive_l=[], interesting_false_negative_l=[], bins=10000, delta_thres=0.01):
    from Common import MPI_LAllGather, LEculidean
    tst_coord = MPI_LAllGather(local_tst_coord)
    tst_label = MPI_LAllGather(local_tst_label)
    local_dist_t2t = LEculidean(local_tst_coord, tst_coord)

    local_pos_conf_l = []
    local_neg_conf_l = []
    label_d = {}
    for i, label in enumerate(local_tst_label):
        if label in label_d:
            label_d[label].append(i)
        else:
            label_d[label] = [i]
    for label in sorted(label_d.keys()):
        qry_indices = np.array(label_d[label])
        pos_ref_indices = np.where(tst_label==label)[0]
        neg_ref_indices = np.where(tst_label!=label)[0]
        local_pos_conf_l.append(local_dist_t2t[qry_indices][:,pos_ref_indices].flatten())
        local_neg_conf_l.append(local_dist_t2t[qry_indices][:,neg_ref_indices].flatten())
    local_pos_conf = - np.hstack(local_pos_conf_l)
    local_neg_conf = - np.hstack(local_neg_conf_l)
    return CalROC_MPI(local_pos_conf, local_neg_conf, interesting_false_positive_l, interesting_false_negative_l, bins, delta_thres)



def GetClusterCentroids_MPI(local_coord, local_cate_offset, cluster_num_per_cate, iter_num=1, show_info=True, gather=True):
    from scipy.cluster.vq import kmeans
    local_centroids_l = []
    local_labels_l = []
    for label in sorted(local_cate_offset.keys()):
        si, ei = local_cate_offset[label]
        if cluster_num_per_cate>0:
            centroids, distortion = kmeans(local_coord[si:ei], cluster_num_per_cate, iter=iter_num)
        else:
            n = - cluster_num_per_cate
            centroids, distortion = kmeans(local_coord[si:ei], ((ei-si) + n - 1) / n, iter=iter_num)

        local_centroids_l.append(centroids)
        local_labels_l.append(np.ones(centroids.shape[0]).astype('int')*label)
        if show_info:
            print 'Rank {}/{}: cate {}/{} is done'.format(comm_rank, comm_size, len(local_centroids_l), len(local_cate_offset))
    local_centroids = np.vstack(local_centroids_l)
    local_labels = np.hstack(local_labels_l)
    if gather:
        centroids = np.vstack(comm.allgather(local_centroids))
        labels = np.hstack(comm.allgather(local_labels))
    else:
        centroids = local_centroids
        labels = local_labels
    return centroids, labels

# added by degang.yang
def findMetricThreshold(local_qry_feat, local_qry_label, ref_feat, ref_label, dist=None):
    if dist==None:                                                              
        dist = LEculidean(local_qry_feat, ref_feat)                             
    else:                                                                       
        assert dist.shape == (local_qry_feat.shape[0], ref_feat.shape[0])       
    dist[dist <= 0] = 0.0                                                          
    dist = np.sqrt(dist)
                                                                        
    local_qry_num = local_qry_feat.shape[0]                                     
    ref_num = ref_feat.shape[0]                                                 
                                                                                
    intra_num = intra_sum = intra_sum2 = 0                                      
    intra_min = 1E20                                                            
    intra_max = 0                                                               
                                                                                
    inter_num = inter_sum = inter_sum2 = 0                                      
    inter_min = 1E20                                                            
    inter_max = 0 

    for lqi in range(local_qry_num):                                            
        intra_indices = np.where(ref_label==local_qry_label[lqi])[0]            
        intra_dist = dist[lqi][intra_indices]                                   
        intra_num += intra_indices.size                                         
        intra_sum += intra_dist.sum()                                           
        intra_sum2 += (intra_dist**2).sum()            
        intra_min = min(intra_dist.min(), intra_min)                         
        intra_max = max(intra_dist.max(), intra_max)                         
                                                                                
        inter_indices = np.where(ref_label!=local_qry_label[lqi])[0]            
        inter_dist = dist[lqi][inter_indices]                                   
        inter_num += inter_indices.size                                         
        inter_sum += inter_dist.sum()                                           
        inter_sum2 += (inter_dist**2).sum()                                     
        inter_min = min(inter_dist.min(), inter_min)                         
        inter_max = max(inter_dist.max(), inter_max)                         
                                                                                
    intra_avg = intra_sum / intra_num                                           
    intra_std = np.sqrt(intra_sum2 / intra_num - intra_avg**2)                  
                                                                                
    inter_avg = inter_sum / inter_num                                           
    inter_std = np.sqrt(inter_sum2 / inter_num - inter_avg**2)                  
                                                                                
    log = 'Intra Distance: {}, {:.4f}+-{:.4f}, min {:.4f}, max {:.4f}'.format(intra_num, intra_avg, intra_std, intra_min, intra_max)
    print log                                                                   
                                                                                
    log = 'Inter Distance: {}, {:.4f}+-{:.4f}, min {:.4f}, max {:.4f}'.format(inter_num, inter_avg, inter_std, inter_min, inter_max)
    print log                                                                   
                                                                                
    if (intra_avg >= inter_avg):                                                
        print 'The Metric Feature Is Too Bad!'                                  
    else:                                                                       
        thres = np.linspace(intra_avg - 1.0, inter_avg + 1.0, 400)                           
        CalClassificationError_MPI(                                             
            local_qry_feat = local_qry_feat,                                    
            local_qry_label = local_qry_label,                                  
            ref_feat = ref_feat,                                                
            ref_label = ref_label,                                              
            threshold_l = thres,                                                
            dist = dist)
def CalROC(feat,label):
  dist=LEculidean(feat, feat)
  DiffScore=[]
  SameScore=[]
  for i in range(0,feat.shape[0]):
    for j in range(0,feat.shape[0]):
      if i==j:
        continue
      if label[i]!=label[j]:
        DiffScore.append(dist[i,j])
      else:
        SameScore.append(dist[i,j])
  
  DiffScore.sort()
  Diff_Num = len(DiffScore)
  Same_Num = len(SameScore)
  FAR=[0.01,0.001,0.0001,0.00001]
  for k in range(0,len(FAR)):
    num = int(FAR[k]*Diff_Num)
    thr = DiffScore[num]
    cnt=0
    for i in range(0,Same_Num):
      if SameScore[i]<thr:
        cnt+=1
    GAR = float(cnt)/Same_Num
    print " thr:",thr," FAR:",FAR[k]," GAR:",GAR


def metricEvaluate(feat, label):                                                         
    st = time.time()                                                               
    CalKNNError_MPI(                                                               
        local_qry_feat=feat,                                                       
        local_qry_label=label,                                                     
        ref_feat=feat,                                                             
        ref_label=label,                                                           
        K=10,                                                                      
        leave_one_out=True)                                                        
    print 'time spent {} sec'.format(time.time()-st)                               
    st = time.time()                                                               
    CalAccuracyTopN_MPI(                                                           
        local_qry_coord=feat,                                                      
        local_qry_label=label,                                                     
        ref_coord=feat,                                                            
        ref_label=label,                                                           
        topN_l=[1,2,4,6,10,100],                                                   
        leave_one_out=True)                                                        
    print 'time spent {} sec'.format(time.time()-st)                               
    st = time.time()                                                               
    mean_avg_prec, mean_nDCG = CalMeanAvgPrecNDCG_MPI(                             
        local_qry_feat=feat,                                                       
        local_qry_label=label,                                                     
        ref_feat=feat,                                                             
        ref_label=label)                                                           
    print 'time spent {} sec'.format(time.time()-st)                               
    st = time.time()                                                               
    findMetricThreshold(                                                           
        local_qry_feat = feat,                                                     
        local_qry_label = label,                                                   
        ref_feat = feat,                                                           
        ref_label = label)                                                         
    print 'time spent {} sec'.format(time.time()-st)

if __name__=='__main__':
    from time import time
    QN = 1000
    RN = 10000
    dim = 256
    topK_l = [2,4,6,100]
    gamma_l = None #[0, 0.5, 1, 2]
    Q = np.random.rand(QN, dim)
    R = np.random.rand(RN, dim)
    qry_labels = (np.random.rand(QN)*10).astype('int')
    R_labels = (np.random.rand(RN)*10).astype('int')

    st = time()
    CalAccuracyTopN_MPI(Q, qry_labels, R, R_labels, topK_l, gamma_l, mAP_NDCG=True, leave_one_out=True)
    print 'time spent {} sec'.format(time()-st)
    st = time()
    CalMeanAvgPrecNDCG_MPI(Q, qry_labels, R, R_labels)
    print 'time spent {} sec'.format(time()-st)
