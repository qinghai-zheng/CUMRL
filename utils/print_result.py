from utils.cluster import cluster
import warnings

warnings.filterwarnings('ignore')


def print_result(n_clusters, H, gt, count=1):
    acc_avg, acc_std, nmi_avg, nmi_std, ar_avg, ar_std, f1_avg, f1_std = cluster(n_clusters, H, gt, count=count)
    return  acc_avg,nmi_avg,ar_avg,f1_avg