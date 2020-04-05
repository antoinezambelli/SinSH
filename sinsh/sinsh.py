import glob
import os
import pickle
import shutil
import subprocess
import time
from datetime import timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm


def print_progress(iteration, total, delta_t, prefix='', length=50, fill='█', end_str=None):
    '''
    print_progress(): tqdm-style progress bar with ETA.
    Inputs:
        iteration - int - required: current iteration of the progress bar (ie, loop counter).
        total - int - required: total number of iterations for this process (ie, len of iterable).
        delta_t - float - required: seconds since the first call to this method, used for ETA.
        prefix - str - optional: prefix for the progress bar (ie, 'Data transfer').
        length - int - optional: number of characters for the progress bar portion.
        fill - str - optional: the character to print to indicate progress (ie, █#|-).
        end_str - any - optional: flag to determine use of carriage return, use if overriding bar.
    Outputs:
        Print a progress bar to terminal:
        Data transfer: 25%|#   | 1/4 [00:04:00<00:12:00, 240s/it]
    Usage:
        Above example gotten from
        print_progress(1, 4, time.time() - t_0, 'Data transfer: ', 4, '#')
    '''

    if iteration != 0:
        time_per_iter = '{:.2f}s/it'.format(delta_t / float(iteration))
        time_eta = '{}'.format(
            str(timedelta(seconds=round((total - iteration) * (delta_t / float(iteration)))))
        )
        time_elapsed = '{}'.format(str(timedelta(seconds=round(delta_t))))
    else:
        time_per_iter = '0s/it'
        time_eta = '00:00:00'
        time_elapsed = '00:00:00'
    percent = '{}'.format(int(100 * iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + ' ' * (length - filled_length)
    
    my_str = '\r{} {}%|{}| {}/{} [{}<{}, {}]'.format(
        prefix, percent, bar, str(iteration), str(total), time_elapsed, time_eta, time_per_iter
    )
    if end_str is None:
        print(my_str)
    else:
        print(my_str, end='\r')


class Node():
    '''
    Cluster node - the client to talk to via SSH.
    '''

    def __init__(self, username, ip_addr):
        self.username = username
        self.ip_addr = ip_addr

    def cmd(self, cmd_str, *args):
        '''
        cmd(): generic function wrapping subprocess.run(), good for single commands and called by
               appropriate convenience methods (mkdir(), rm(), scp()). Intended for internal use.
        Inputs:
            cmd_str - str or list - required: the command to execute if we do not want default
                    behavior. Default is to execute 'ssh user@ip' + args.
                    If cmd_str is a string, make sure it is .split()-safe.
            args - str - required: additional arguments to pass to subprocess.run(), this will be
                 the bulk of the command.
        Outputs:
            Executes a call as subprocess.run(cmd_str + list(args)) - blocking call.
        Usage:
            To execute subprocess.run(['rm', '-rf', '/path/to/dir']) on the node, we would call
            my_args = ['rm', '-rf', '/path/to/dir']
            my_node.cmd([], *my_args)
        '''

        # Check if called from a str or list format.
        if isinstance(cmd_str, str):
            cmd_str = cmd_str.split()

        # If we have have no cmd_str, use default.
        if not cmd_str:
            cmd_str = [
                'ssh',
                '{}@{}'.format(self.username, self.ip_addr)
            ]

        cmd_str = cmd_str + list(args)  # Add any args that were passed in.

        subprocess.run(
            ' '.join(cmd_str) if any('*' in a for a in cmd_str) else cmd_str,
            shell=any('*' in a for a in cmd_str)
        )  # If passing a wildcard character, set Shell=True, run as string argument.

    def rm(self, *args):
        '''
        rm(): convenience method for calling 'rm' on a Node.
        Inputs:
            args - str - required: additional args to pass with 'rm' to build the full command.
        Outputs:
            Executes a call to Node.cmd() with appropriate args.
        Usage:
            my_node.rm('-r', 'path/to/dir') will remove the ddirectory at /path/to/dir on the Node.
        '''

        args = ['rm'] + list(args)
        self.cmd([], *args)

    def mkdir(self, *args):
        '''
        mkdir(): convenience method for calling 'mkdir' on a Node.
        Inputs:
            args - str - required: additional args to pass with 'mkdir' to build the full command.
        Outputs:
            Executes a call to Node.cmd() with appropriate args.
        Usage:
            my_node.mkdir('/path/to/dir') will crate the ddirectory at /path/to/dir on the Node.
        '''

        args = ['mkdir'] + list(args)
        self.cmd([], *args)

    def scp(self, *args):
        '''
        scp(): legacy method that makes the appropriate call to copy_to() or copy_from().
        Inputs:
            args - str - required: additional args to pass with 'scp' to build the full command.
                 Will look for arg with ':' to determine which path is SSH and which is local.
        Outputs:
            Executes a call to Node.copy_from() or Node.copy_to() with appropriate args.
        Usage:
            my_node.scp('-rpq', '/path/to/local/dir', ':/path/to/remote') will call copy_to() to
            copy /path/to/local/dir to user@ip:/path/to/remote.
        '''

        args = list(args)
        for i in range(len(args)):
            if ':' in args[i]:
                idx = i

        if idx == len(args) - 1:
            self.copy_to(*args)
        else:
            self.copy_from(*args)

    def copy_to(self, *args):
        '''
        copy_to(): convenience method for calling Node.cmd() to run an scp command.
        Inputs:
            args - str - required: additional args to pass with 'scp' to build the full command.
                 Last arg assumed to be remote path.
        Outputs:
            Executes a call to Node.cmd() with appropriate args.
        Usage:
            node.copy_to('-rpq', '/path/local/dir', '/path/remote/dir') will copy the local dir at
            /path/local/dir to user@ip:/path/remote/dir.
        '''

        args = list(args)
        args[-1] = '{}@{}'.format(
            self.username,
            self.ip_addr
        ) + (':' not in args[-1]) * ':' + args[-1]  # Build to-path, add ':' if not there.

        cmd_str = ['scp'] + args
        self.cmd(cmd_str)

    def copy_from(self, *args):
        '''
        copy_from(): convenience method for calling Node.cmd() to run an scp command.
        Inputs:
            args - str - required: additional args to pass with 'scp' to build the full command.
                 Second-to-last arg assumed to be remote path.
        Outputs:
            Executes a call to Node.cmd() with appropriate args.
        Usage:
            node.copy_from('-rpq', '/path/remote/dir', '/path/local/dir') will copy the remote dir
            at user@ip:/path/remote/dir to /path/local/dir.
        '''

        args = list(args)
        args[-2] = '{}@{}'.format(
            self.username,
            self.ip_addr
        ) + (':' not in args[-2]) * ':' + args[-2]  # Build from-path, add ':' if not there.

        cmd_str = ['scp'] + args
        self.cmd(cmd_str)

    def exec(self, *args):
        '''
        exec(): non-blocking alternative to Node.cmd(), wraps subprocess.Popen(). Intended for long
                runtime executions (ie, python3).
        Inputs:
            args - str - required: additional arguments to pass to subprocess.Popen(), this will be
                 the bulk of the command.
        Outputs:
            Executes a call as subprocess.Popen(['ssh', 'user@ip'] + list(args)) - non-blocking.
        Usage:
            my_node.exec(['python3', '/path/to/file.py']) will run file.py on remote Node. Note it
            will not copy the file beforehand.
        '''

        exec_str = [
            'ssh',
            '{}@{}'.format(self.username, self.ip_addr)
        ] + list(args)

        subprocess.Popen(exec_str)

    def shell(self, *args):
        '''
        shell(): blocking call to subprocess.Popen() with piped output.
        Inputs:
            args - str - required: additional arguments to pass to subprocess.Popen(), this will be
                 the bulk of the command.
        Outputs:
            out - str: as returned by subprocess.Popen().communicate().
            err - str: as returned by subprocess.Popen().communicate().
        Usage:
            my_node.shell('find', '/path/to/results/dir', '-name', '*_res.p')[0].split() will give
            a list of all files in /path/to/results/dir that end in _res.p.
        '''

        exec_str = [
            'ssh',
            '-t',
            '{}@{}'.format(self.username, self.ip_addr)
        ] + list(args)

        p_lst = subprocess.Popen(
            exec_str,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )  # Launch process.
        out, err = p_lst.communicate()  # Grab output.

        return out, err


class Cluster():
    '''
    Collection of Nodes with batch processing and auto-distribute functionality. The distribute()
    convenience method should cover basic use-cases without the need for the other methods.
    Usage:
        This code expects (but may work otherwise) a Master device that will not do computations,
        which will call on N Nodes. The following illustrates how data will be copied for a 2-node
        example:
    [Code on Master]
            path_to_code
               /      \
          code_1.py  code_2.py

        The code will be copied as-is to the Nodes.
    [Data on Master]
        Case - single file.
            path_to_data
                 |
            data_1.csv

        Case - multiple files.
            path_to_data
               /      \
          data_1.ext  data_2.ext

        Case - nested dirs (files can have same names).
            path_to_data
               /        \
             abc         def
            /  |         /   \
        data_1 data_2  data_3 data_4

        The data will be copied to Node 0 as, respectively:
            path_to_data
                 |
               0.csv

            path_to_data
                 |
             data_1.ext

            path_to_data
                 |
                abc
               /   \
            data_1 data_2
    [Results on Node 0]
        Case - flat directory *preferred:
            path_to_res
               /      \
        abc_res.ext  def_res.ext

        Case - nested directory (will be flattened so names should be different):
            path_to_res
               /        \
            abc         def
            /  |          \
    abc_1.ext abc_2.ext  def_1.ext

        The results will be copied to Master as, respectively:
            path_to_res
               /      \
    abc_res.ext  def_res.ext

            path_to_res
            /    |      \
      abc_1.ext abc_2.ext def_1.ext

    '''

    def __init__(self, nodes):
        '''
        List of Node instances.
        '''
        self.nodes = nodes

    def clean_dirs(self):
        '''
        clean_dirs(): method to delete old directories on the Nodes and create new ones. Intended
        for internal use.
        Inputs:
            None.
        Outputs:
            Deletes data, code and results directories from the Node and creates clean data and
            results directories. Note that it does not create code directories.
        Usage:
            my_cluster.clean_dirs() will clean the dirs.
        '''

        for node in tqdm(self.nodes, desc='Prep nodes', ncols=100):
            node.rm('-r', self.path_to_data)
            node.rm('-r', self.path_to_code)
            node.rm('-r', self.path_to_res)

            node.mkdir(self.path_to_data)
            node.mkdir(self.path_to_code)
            node.mkdir(self.path_to_res)

        return self

    def partition_data(self):
        '''
        partition_data(): method to chunk the data and copy it to Nodes. Can handle a single large
        .csv file or several files or directories - sending num_files / N files to each Node.
        Inputs:
            None.
        Outputs:
            Copies the contents of Cluster.path_to_data to Nodes. If one .csv file then the file is
            split into smaller fragments. If several files/diretories then they are sent in .tar
            balls and un-tar'd on the Nodes.
        Usage:
            my_cluster.partition_data() will split and copy the data to the Nodes.
        '''

        data_contents = glob.glob(self.path_to_master_data + '/*')

        if len(data_contents) == 1:
            # Get total lines in file.
            with open(data_contents[0]) as f:
                num_lines = sum(1 for line in f)

            # Get chunksize needed.
            chunk_size = -(-num_lines // len(self.nodes))  # Ceil division.

            # Read the .csv in chunks and save as separate files.
            for i, chunk in enumerate(pd.read_csv(data_contents[0], chunksize=chunk_size)):
                chunk.to_csv('{}.csv'.format(i), index=False)

            # Copy the files to Nodes.
            for idx, node in enumerate(self.nodes):
                node.data = ['{}.csv'.format(idx)]

                node.copy_to(
                    '-rpq',
                    self.path_to_master_data + '/{}.csv'.format(idx),
                    self.path_to_data + '/{}.csv'.format(idx)
                )
        else:
            # NOTE: this will error if fewer data chunks than there are nodes.

            # Many files/dirs, send equal number to each Node.
            data_chunked = [list(x) for x in np.array_split(data_contents, len(self.nodes))]

            # tar the data for faster transfer.
            for idx, node in enumerate(tqdm(self.nodes, desc='Distribute data', ncols=100)):
                node.data = data_chunked[idx]  # For future reference.

                # Write node's data to a text file - for tar'ing.
                with open('{}/{}.txt'.format(self.path_to_master_data, idx), 'w') as f:
                    for dat in data_chunked[idx]:
                        f.write('{}\n'.format(os.path.split(dat)[1]))

                # tar files.
                p_p = subprocess.Popen(
                    [
                        'tar', '-cf', '{}.tar'.format(idx), '-C', self.path_to_master_data + '/',
                        '-T', '{}.txt'.format(idx)
                    ],
                    cwd=self.path_to_master_data
                ).wait()

                # Copy tar file to the Node.
                node.copy_to(
                    '-rpq',
                    self.path_to_master_data + '/{}.tar'.format(idx),
                    self.path_to_data + '/{}.tar'.format(idx)
                )

                # Un-tar file on the Node.
                node.cmd(
                    [], 'tar', 'xf', self.path_to_data + '/{}.tar'.format(idx),
                    '-C', self.path_to_data
                )
                node.rm('-r', self.path_to_data + '/{}.tar'.format(idx))  # Cleanup tar on node.

                # Cleanup .txt and .tar files on Master.
                os.remove(self.path_to_master_data + '/{}.tar'.format(idx))
                os.remove(self.path_to_master_data + '/{}.txt'.format(idx))

        return self

    def copy_code(self):
        '''
        copy_code(): method to copy the code directory over to the Nodes.
        Inputs:
            None.
        Outputs:
            Copies the Cluster.path_to_code directory to the Nodes.
        Usage:
            my_cluster.copy_code() will copy the code.
        '''

        # Copy the code over to the Nodes - does not support nested directories.
        for node in tqdm(self.nodes, desc='Distribute code', ncols=100):
            node.copy_to(
                '-rpq',
                self.path_to_master_code + '/*',
                self.path_to_code
            )
        return self

    def launch(self):
        '''
        launch(): method to run a Node.exec() command on the Nodes.
        Inputs:
            None.
        Outputs:
            Launches the Cluster.exec_args process via Node.exec().
        Usage:
            my_cluster.launch() will start the process on the Nodes.
        '''
        for node in self.nodes:
            node.exec(*self.exec_args)

        return self

    def check_status(self, res_wc):
        '''
        check_status(): method to check the completion status of the launched process. It must be
                        able to find files using Cluster.res_wc in the results directory. It assumes
                        1 file for each partitioned data piece that Node received.
        Inputs:
            res_wc - str - required: string to identify a finished compute file, can include
                                     wildcard character. It is passed to Node.shell() in a find
                                     command.
        Outputs:
            out - dict: dictionary key'd by Nodes which has lists of files requested and files done.
        Usage:
            my_cluster.check_status('*_res.p') go through each Node and find files ending in _res.p
            in Cluster.path_to_res and return a dict of the form
            {my_node_1: {'todo': ['data_1', 'data_2'], 'done': ['1_res.p']}, my_node_2: ...}.
        '''

        out = {node: {'done': [], 'todo': node.data} for node in self.nodes}
        for node in self.nodes:
            out[node]['done'] = node.shell('find', self.path_to_res, '-name', res_wc)[0].split()

        pickle.dump(out, open(self.path_to_master_res + '/progress.p', 'wb'))  # For external query.

        return out

    def copy_res(self):
        '''
        copy_res(): method to copy results from Nodes to Master.
        Inputs:
            None.
        Outputs:
            This copies each Node's results directory content to the Master results directory.
            This will flatten any results directory structure.
        Usage:
            my_cluster.copy_res() will read Cluster.path_to_res on all the Nodes and copy contents
            to Master at path_to_res. Any nested directories are flattened.
        '''

        os.remove(self.path_to_master_res + '/progress.p')  # Cleanup progress file.

        for idx, node in enumerate(tqdm(self.nodes, desc='Collect results', ncols=100)):
            # Copy the /res directory over into /res/idx.
            node.copy_from(
                '-rpq',
                self.path_to_res,
                self.path_to_master_res + '/{}'.format(idx)
            )

            # Move files from inner directory to /res - note this flattens results.
            file_list = glob.glob(self.path_to_master_res + '{}/**/*'.format(idx))
            for fp in file_list:
                shutil.move(fp, self.path_to_master_res + '/{}'.format(os.path.split(fp)[1]))

            os.rmdir(self.path_to_master_res + '/{}'.format(idx))

        return self

    def distribute(self, path_to_master_data, path_to_master_res, path_to_master_code,
            path_to_data, path_to_code, path_to_res, exec_args, res_wc='*_res.p', blocking=False):
        '''
        distribute(): convenience method to handle typical use-cases - makes all necessary internal
                      calls to other methods for a full batch-processing job.
        Inputs:
            path_to_master_data - str - required: path to the master data directory.
            path_to_master_res - str - required: path to the collected results directory on master.
            path_to_master_code - str - required: path to code to distribute to nodes.
            path_to_data - str - required: path to the partitioned data directory on the node.
            path_to_code - str - required: path to the code directory.
            path_to_res - str - required: path to the results directory on the node.
            exec_args - list - required: list of arguments for the Cluster.launch() method.
            res_wc - str - optional: filename pattern for the Cluster.check_status() method.
            blocking - bool - optional: setting to True will make the call blocking and display
                                        compute progress via Cluster.check_status().
        Outputs:
            This will clean Node directories, partition and copy the data and launch a process. If
            blocking then it will display a progress bar for all Nodes until *all* nodes complete,
            then it will copy the results over. If non-blocking, it will not wait nor copy results.
        Usage:
            Typical usage for a 2-node cluster:
            ip_list = ['192.168.1.155', '192.168.1.156']
            my_clust = Cluster([Node('pi', ip) for ip in ip_list])
            my_clust.distribute(
                '/home/pi/Desktop/master_data',
                '/home/pi/Desktop/master_res',
                '/home/pi/Desktop/master_code',
                '/home/pi/Desktop/cluster_data',
                '/home/pi/Desktop/cluster_code',
                '/home/pi/Desktop/cluster_res',
                ['python3', '/home/pi/Desktop/cluster_code/fundly_v1.py'],
                res_wc='*_res.p',
                blocking=True
            )
        '''

        self.path_to_master_data = path_to_master_data
        self.path_to_master_res = path_to_master_res
        self.path_to_master_code = path_to_master_code
        self.path_to_data = path_to_data
        self.path_to_code = path_to_code
        self.path_to_res = path_to_res
        self.exec_args = exec_args
        self.res_wc = res_wc

        self.clean_dirs()  # Remove old dirs from Node, re-initialize.
        self.partition_data()  # Copy distributed data to Nodes.
        self.copy_code()  # Copy the code files.
        self.launch()  # Launch the command - ie, 'python3 my_script.py'.

        if blocking:
            t_0 = time.time()

            # Loop and check_status() until complete.
            while True:
                res = self.check_status(self.res_wc)

                # Print current progress with ETA.
                p_cls = subprocess.call('clear', shell=True)
                for node in self.nodes:
                    print_progress(
                        len(res[node]['done']),
                        len(res[node]['todo']),
                        delta_t=time.time() - t_0,
                        prefix=node.ip_addr.ljust(max([len(n.ip_addr) for n in self.nodes])) + ':',
                        length=50,
                        fill='█'
                    )

                # Check if we have all the result files we want for all the nodes.
                if all(len(v['done']) == len(v['todo']) for k, v in res.items()):
                    break

                time.sleep(30)

            self.copy_res()
