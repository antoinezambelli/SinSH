# SinSH
Simple Python SSH wrapper for distributed clusters

## Typical usage of Cluster class
```
ip_list = ['192.168.1.155', '192.168.1.156']
my_clust = Cluster([Node('pi', ip) for ip in ip_list])
my_clust.distribute(
    '/home/pi/Desktop/cluster_data',
    '/home/pi/Desktop/cluster_code',
    '/home/pi/Desktop/cluster_res',
    ['python3', '/home/pi/Desktop/cluster_code/fundly_v1.py'],
    res_wc='*_res.p',
    blocking=True
)
```
This will copy `cluster_code` to each node and run the `fundly_v1.py` file on distributed data from `cluster_data`. The data can be files or folders that will be sent to the nodes (for `N` files and `M` nodes we send roughly `N/M` files to each node). Results are then gathered back in `cluster_res`.
