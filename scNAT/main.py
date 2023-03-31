import sys

sys.path.append('../')
from functions.Layers import *
from functions.data_processing import *
from functions.utils_s import *
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import glob
import seaborn as sns
import umap.umap_ as umap
import warnings
import shutil
from multiprocessing.pool import Pool 
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class DeepTCR_base(object):

    def __init__(self,Name,max_length=40,device=0,tf_verbosity=3):
        """
        # Initialize Training Object.
        Initializes object and sets initial parameters.
        All DeepTCR algorithms begin with initializing a training object. This object will contain all methods, data, and results during the training process. One can extract learned features, per-sequence predictions, among other outputs from DeepTCR and use those in their own analyses as well.
        Args:
            Name (str): Name of the object. This name will be used to create folders with results as well as a folder with parameters and specifications for any models built/trained.
            max_length (int): maximum length of CDR3 sequence.
            device (int): In the case user is using tensorflow-gpu, one can specify the particular device to build the graphs on. This selects which GPU the user wants to put the graph and train on.
            tf_verbosity (str): determines how much tensorflow log output to display while training.
            0 = all messages are logged (default behavior)
            1 = INFO messages are not printed
            2 = INFO and WARNING messages are not printed
            3 = INFO, WARNING, and ERROR messages are not printed
        """

        #Assign parameters
        self.Name = Name
        self.max_length = max_length
        self.use_beta = False
        self.use_alpha = False
        self.device = '/device:GPU:'+str(device)
        self.use_v_beta = False
        self.use_d_beta = False
        self.use_j_beta = False
        self.use_v_alpha = False
        self.use_j_alpha = False
        self.use_rna = False
        self.regression = False
        self.unknown_str = '__unknown__'

        #Create dataframes for assigning AA to ints
        aa_idx, aa_mat = make_aa_df()
        aa_idx_inv = {v: k for k, v in aa_idx.items()}
        self.aa_idx = aa_idx
        self.aa_idx_inv = aa_idx_inv

        #Create directory for results of analysis
        directory = os.path.join(self.Name,'results')
        self.directory_results = directory
        if not os.path.exists(directory):
            os.makedirs(directory)

        #Create directory for any temporary files
        directory = self.Name
        if not os.path.exists(directory):
            os.makedirs(directory)

        tf.compat.v1.disable_eager_execution()
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf_verbosity)

    def Get_Data(self,directory,file_rna,classes=None, n_jobs=40,
                    aa_column_alpha = None,aa_column_beta = None, count_column = None,sep='\t',aggregate_by_aa=True,
                    v_alpha_column=None,j_alpha_column=None,
                    v_beta_column=None,j_beta_column=None,d_beta_column=None, p=None):
        """
        # Get Data for DeepTCR
        Parse Data into appropriate inputs for neural network from directories where data is stored.
        This method can be used when your data is stored in directories and you want to load it from directoreis into DeepTCR. This method takes care of all pre-processing of the data including:
         - Combining all CDR3 sequences with the same nucleotide sequence (optional).
         - Removing any sequences with non-IUPAC characters.
         - Removing any sequences that are longer than the max_length set when initializing the training object.
         - Determining how much of the data per file to use (type_of_data_cut)
        Args:
            directory (str): Path to directory with folders with tsv/csv files are present for analysis. Folders names become labels for files within them. If the directory contains the TCRSeq files not organized into classes/labels, DeepTCR will load all files within that directory.
            file_rna (str): Path to scRNA-seq file. Rows are cells, and columns are genes.
            classes (list): Optional selection of input of which sub-directories to use for analysis.
            type_of_data_cut (str): Method by which one wants to sample from the TCRSeq File.
                ###
                Options are:
                - Fraction_Response: A fraction (0 - 1) that samples the top fraction of the file by reads. For example, if one wants to sample the top 25% of reads, one would use this threshold with a data_cut = 0.25. The idea of this sampling is akin to sampling a fraction of cells from the file.
                - Frequency_Cut: If one wants to select clones above a given frequency threshold, one would use this threshold. For example, if one wanted to only use clones about 1%, one would enter a data_cut value of 0.01.
                - Num_Seq: If one wants to take the top N number of clones, one would use this threshold. For example, if one wanted to select the top 10 amino acid clones from each file, they would enter a data_cut value of 10.
                - Read_Cut: If one wants to take amino acid clones with at least a certain number of reads, one would use this threshold. For example, if one wanted to only use clones with at least 10 reads,they would enter a data_cut value of 10.
                - Read_Sum: IF one wants to take a given number of reads from each file, one would use this threshold. For example, if one wants to use the sequences comprising the top 100 reads of hte file, they would enter a data_cut value of 100.
            data_cut (float or int): Value  associated with type_of_data_cut parameter.
            n_jobs (int): Number of processes to use for parallelized operations.
            aa_column_alpha (int): Column where alpha chain amino acid data is stored. (0-indexed).
            aa_column_beta (int): Column where beta chain amino acid data is stored.(0-indexed)
            count_column (int): Column where counts are stored.
            sep (str): Type of delimiter used in file with TCRSeq data.
            aggregate_by_aa (bool): Choose to aggregate sequences by unique amino-acid. Defaults to True. If set to False, will allow duplicates of the same amino acid sequence given it comes from different nucleotide clones.
            v_alpha_column (int): Column where v_alpha gene information is stored.
            j_alpha_column (int): Column where j_alpha gene information is stored.
            v_beta_column (int): Column where v_beta gene information is stored.
            d_beta_column (int): Column where d_beta gene information is stored.
            j_beta_column (int): Column where j_beta gene information is stored.
            p (multiprocessing pool object): For parellelized operations, one can pass a multiprocessing pool object to this method.
        Returns:
            variables into training object
            - self.alpha_sequences (ndarray): array with alpha sequences (if provided)
            - self.beta_sequences (ndarray): array with beta sequences (if provided)
            - self.class_id (ndarray): array with sequence class labels
            - self.sample_id (ndarray): array with sequence file labels
            - self.freq (ndarray): array with sequence frequencies from samples
            - self.counts (ndarray): array with sequence counts from samples
            - self.(v/d/j)_(alpha/beta) (ndarray): array with sequence (v/d/j)-(alpha/beta) usage
        """

        if aa_column_alpha is not None:
            self.use_alpha = True

        if aa_column_beta is not None:
            self.use_beta = True

        if v_alpha_column is not None:
            self.use_v_alpha = True

        if j_alpha_column is not None:
            self.use_j_alpha = True

        if v_beta_column is not None:
            self.use_v_beta = True

        if d_beta_column is not None:
            self.use_d_beta = True

        if j_beta_column is not None:
            self.use_j_beta = True


        #Determine classes based on directory names
        data_in_dirs = True
        if classes is None:
            classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory,d))]
            classes = [f for f in classes if not f.startswith('.')]
            if not classes:
                classes = ['None']
                data_in_dirs = False

        self.lb = LabelEncoder()
        self.lb.fit(classes)
        self.classes = self.lb.classes_

        if p is None:
            p_ = Pool(n_jobs)
        else:
            p_ = p
        
        if sep == '\t':
            ext = '*.tsv'
        elif sep == ',':
            ext = '*.csv'
        else:
            print('Not Valid Delimiter')
            return

        #Get data from tcr-seq files
        alpha_sequences = []
        beta_sequences = []
        v_beta = []
        d_beta = []
        j_beta = []
        v_alpha = []
        j_alpha = []
        label_id = []
        file_id = []
        freq = []
        counts=[]
        file_list = []
        seq_index = []
        print('Loading Data...')
        for type in self.classes:
            if data_in_dirs:
                files_read = glob.glob(os.path.join(directory, type, ext))
            else:
                files_read = glob.glob(os.path.join(directory,ext))
            num_ins = len(files_read)
            args = list(zip(files_read,
                            [aa_column_alpha] * num_ins,
                            [aa_column_beta] * num_ins,
                            [count_column] * num_ins,
                            [sep] * num_ins,
                            [self.max_length]*num_ins,
                            [aggregate_by_aa]*num_ins,
                            [v_beta_column]*num_ins,
                            [d_beta_column]*num_ins,
                            [j_beta_column]*num_ins,
                            [v_alpha_column]*num_ins,
                            [j_alpha_column]*num_ins))

            DF = p_.starmap(Get_DF_Data, args)

            DF_temp = []
            files_read_temp = []
            for df,file in zip(DF,files_read):
                if df.empty is False:
                    DF_temp.append(df)
                    files_read_temp.append(file)

            DF = DF_temp
            files_read = files_read_temp

            for df, file in zip(DF, files_read):
                if aa_column_alpha is not None:
                    alpha_sequences += df['alpha'].tolist()
                if aa_column_beta is not None:
                    beta_sequences += df['beta'].tolist()

                if v_alpha_column is not None:
                    v_alpha += df['v_alpha'].tolist()

                if j_alpha_column is not None:
                    j_alpha += df['j_alpha'].tolist()

                if v_beta_column is not None:
                    v_beta += df['v_beta'].tolist()

                if d_beta_column is not None:
                    d_beta += df['d_beta'].tolist()

                if j_beta_column is not None:
                    j_beta += df['j_beta'].tolist()

                label_id += [type] * len(df)
                file_id += [file.split('/')[-1]] * len(df)
                file_list.append(file.split('/')[-1])
                freq += df['Frequency'].tolist()
                counts += df['counts'].tolist()
                seq_index += df.index.tolist()

        alpha_sequences = np.asarray(alpha_sequences)
        beta_sequences = np.asarray(beta_sequences)
        v_beta = np.asarray(v_beta)
        d_beta = np.asarray(d_beta)
        j_beta = np.asarray(j_beta)
        v_alpha = np.asarray(v_alpha)
        j_alpha = np.asarray(j_alpha)
        label_id = np.asarray(label_id)
        file_id = np.asarray(file_id)
        freq = np.asarray(freq)
        counts = np.asarray(counts)
        seq_index = np.asarray(seq_index)

        Y = self.lb.transform(label_id)
        OH = OneHotEncoder(sparse=False,categories='auto')
        Y = OH.fit_transform(Y.reshape(-1,1))

        print('Embedding Sequences...')
        #transform sequences into numerical space
        if aa_column_alpha is not None:
            args = list(zip(alpha_sequences, [self.aa_idx] * len(alpha_sequences), [self.max_length] * len(alpha_sequences)))
            result = p_.starmap(Embed_Seq_Num, args)
            sequences_num = np.vstack(result)
            X_Seq_alpha = np.expand_dims(sequences_num, 1)

        if aa_column_beta is not None:
            args = list(zip(beta_sequences, [self.aa_idx] * len(beta_sequences),  [self.max_length] * len(beta_sequences)))
            result = p_.starmap(Embed_Seq_Num, args)
            sequences_num = np.vstack(result)
            X_Seq_beta = np.expand_dims(sequences_num, 1)

        if self.use_alpha is False:
            X_Seq_alpha = np.zeros(shape=[len(label_id)])
            alpha_sequences = np.asarray([None]*len(label_id))

        if self.use_beta is False:
            X_Seq_beta = np.zeros(shape=[len(label_id)])
            beta_sequences = np.asarray([None]*len(label_id))

        if p is None:
            p_.close()
            p_.join()

        #transform v/d/j genes into categorical space
        num_seq = X_Seq_alpha.shape[0]
        if self.use_v_beta is True:
            self.lb_v_beta = LabelEncoder()
            self.lb_v_beta.classes_ = np.insert(np.unique(v_beta), 0, self.unknown_str)
            v_beta_num = self.lb_v_beta.transform(v_beta)
        else:
            self.lb_v_beta = LabelEncoder()
            v_beta_num = np.zeros(shape=[num_seq])
            v_beta = np.asarray([None]*len(label_id))

        if self.use_d_beta is True:
            self.lb_d_beta = LabelEncoder()
            self.lb_d_beta.classes_ = np.insert(np.unique(d_beta), 0, self.unknown_str)
            d_beta_num = self.lb_d_beta.transform(d_beta)
        else:
            self.lb_d_beta = LabelEncoder()
            d_beta_num = np.zeros(shape=[num_seq])
            d_beta = np.asarray([None]*len(label_id))

        if self.use_j_beta is True:
            self.lb_j_beta = LabelEncoder()
            self.lb_j_beta.classes_ = np.insert(np.unique(j_beta), 0, self.unknown_str)
            j_beta_num = self.lb_j_beta.transform(j_beta)
        else:
            self.lb_j_beta = LabelEncoder()
            j_beta_num = np.zeros(shape=[num_seq])
            j_beta = np.asarray([None]*len(label_id))

        if self.use_v_alpha is True:
            self.lb_v_alpha = LabelEncoder()
            self.lb_v_alpha.classes_ = np.insert(np.unique(v_alpha), 0, self.unknown_str)
            v_alpha_num = self.lb_v_alpha.transform(v_alpha)
        else:
            self.lb_v_alpha = LabelEncoder()
            v_alpha_num = np.zeros(shape=[num_seq])
            v_alpha = np.asarray([None]*len(label_id))

        if self.use_j_alpha is True:
            self.lb_j_alpha = LabelEncoder()
            self.lb_j_alpha.classes_ = np.insert(np.unique(j_alpha), 0, self.unknown_str)
            j_alpha_num = self.lb_j_alpha.transform(j_alpha)
        else:
            self.lb_j_alpha = LabelEncoder()
            j_alpha_num = np.zeros(shape=[num_seq])
            j_alpha = np.asarray([None]*len(label_id))

        print('Read in scRNA-seq data...')
        sc = pd.read_csv(file_rna, sep=sep, index_col=0)
        # mat = np.expand_dims(np.asarray(sc),1)
        mat = np.asarray(sc)
        barcode = np.asarray(sc.index)
        gene = np.asarray(sc.columns)

#         with open(os.path.join(self.Name,'Data.pkl'), 'wb') as f:
#                 pickle.dump([X_Seq_alpha,X_Seq_beta,Y, alpha_sequences,beta_sequences, label_id, file_id, freq,counts,seq_index,
#                             self.lb,file_list,self.use_alpha,self.use_beta,
#                             self.lb_v_beta, self.lb_d_beta, self.lb_j_beta,self.lb_v_alpha,self.lb_j_alpha,
#                             v_beta, d_beta,j_beta,v_alpha,j_alpha,
#                             v_beta_num, d_beta_num, j_beta_num,v_alpha_num,j_alpha_num, mat, barcode, gene,
#                             self.use_v_beta,self.use_d_beta,self.use_j_beta,self.use_v_alpha,self.use_j_alpha],f,protocol=4)


        self.X_Seq_alpha = X_Seq_alpha
        self.X_Seq_beta = X_Seq_beta
        self.Y = Y
        self.alpha_sequences = alpha_sequences
        self.beta_sequences = beta_sequences
        self.class_id = label_id
        self.sample_id = file_id
        self.freq = freq
        self.counts = counts
        self.sample_list = file_list
        self.v_beta = v_beta
        self.v_beta_num = v_beta_num
        self.d_beta = d_beta
        self.d_beta_num = d_beta_num
        self.j_beta = j_beta
        self.j_beta_num = j_beta_num
        self.v_alpha = v_alpha
        self.v_alpha_num = v_alpha_num
        self.j_alpha = j_alpha
        self.j_alpha_num = j_alpha_num
        self.mat = mat
        self.barcode = barcode
        self.gene = gene
        self.seq_index = np.asarray(list(range(len(self.Y))))
        self.predicted = np.zeros((len(self.Y),len(self.lb.classes_)))
        print('Data Loaded')

class vis_class(object):

    def UMAP_Plot(self, set='all', class_id=None, by_class=False, by_cluster=False,
                    by_sample=False, freq_weight=False, show_legend=True,
                    scale=100,alpha=1.0,sample=None,sample_per_class=None,filename=None,
                    prob_plot=None,plot_by_class=False):

        """
        # UMAP visualization of TCR Sequences
        This method displays the sequences in a 2-dimensional UMAP where the user can color code points by class label, sample label, or prior computing clustering solution. Size of points can also be made to be proportional to frequency of sequence within sample.
        Args:
            set (str): To choose which set of sequences to analye, enter either 'all','train', 'valid',or 'test'. Since the sequences in the train set may be overfit, it preferable to generally examine the test set on its own.
            by_class (bool): To color the points by their class label, set to True.
            by_sample (bool): To color the points by their sample lebel, set to True.
            by_cluster (bool): To color the points by the prior computed clustering solution, set to True.
            freq_weight (bool): To scale size of points proportionally to their frequency, set to True.
            show_legend (bool): To display legend, set to True.
            scale (float): To change size of points, change scale parameter. Is particularly useful when finding good display size when points are scaled by frequency.
            alpha (float): Value between 0-1 that controls transparency of points.
            sample (int): Number of events to sub-sample for visualization.
            sample_per_class (int): Number of events to randomly sample per class for UMAP.
            filename (str): To save umap plot to results folder, enter a name for the file and the umap will be saved to the results directory. i.e. umap.png
            prob_plot (str): To plot the predicted probabilities for the sequences as an additional heatmap, specify the class probability one wants to visualize (i.e. if the class of interest is class A, input 'A' as a string). Of note, only probabilities determined from the sequences in the test set are displayed as a means of not showing over-fit probabilities. Therefore, it is best to use this parameter when the set parameter is turned to 'test'.
        """
        idx = None
        features = self.features
        class_id = self.class_id
        sample_id = self.sample_id
        freq = self.freq
        predicted = self.predicted
        if hasattr(self, 'Cluster_Assignments'):
            IDX = self.Cluster_Assignments
        else:
            IDX = None

        if sample_per_class is not None and sample is not None:
            print("sample_per_class and sample cannot be assigned simultaneously")
            return

        if sample is not None:
            idx = np.random.choice(range(len(features)), sample, replace=False)
            features = features[idx]
            class_id = class_id[idx]
            sample_id = sample_id[idx]
            freq = freq[idx]
            predicted = predicted[idx]
            if hasattr(self, 'Cluster_Assignments'):
                IDX = IDX[idx]
            else:
                IDX = None

        if sample_per_class is not None:
            features_temp = []
            class_temp = []
            sample_temp = []
            freq_temp = []
            predicted_temp = []
            cluster_temp = []

            for i in self.lb.classes_:
                sel = np.where(class_id == i)[0]
                sel = np.random.choice(sel, sample_per_class, replace=False)
                features_temp.append(features[sel])
                class_temp.append(class_id[sel])
                sample_temp.append(sample_id[sel])
                freq_temp.append(freq[sel])
                predicted_temp.append(predicted[sel])
                if hasattr(self, 'Cluster_Assignments'):
                    cluster_temp.append(IDX[sel])

            features = np.vstack(features_temp)
            class_id = np.hstack(class_temp)
            sample_id = np.hstack(sample_temp)
            freq = np.hstack(freq_temp)
            predicted = np.hstack(predicted_temp)
            if hasattr(self, 'Cluster_Assignments'):
                IDX = np.hstack(cluster_temp)

        pca = PCA(n_components=20)
        umap_obj = umap.UMAP()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            X_2 = umap_obj.fit_transform(pca.fit_transform(features))
        with open(os.path.join(self.Name, 'umap.pkl'), 'wb') as f:
            pickle.dump([X_2,features,class_id,sample_id,freq,IDX,idx], f, protocol=4)

        df_plot = pd.DataFrame()
        df_plot['x'] = X_2[:, 0]
        df_plot['y'] = X_2[:, 1]
        df_plot['Class'] = class_id
        df_plot['Sample'] = sample_id

        if prob_plot is not None:
            df_plot['Predicted'] = predicted[:,self.lb.transform([prob_plot])[0]]

        if set != 'all':
            df_plot['Set'] = None
            with pd.option_context('mode.chained_assignment',None):
                df_plot['Set'].iloc[np.where(self.train_idx)[0]] = 'train'
                df_plot['Set'].iloc[np.where(self.valid_idx)[0]] = 'valid'
                df_plot['Set'].iloc[np.where(self.test_idx)[0]] = 'test'

        if IDX is not None:
            IDX[IDX == -1] = np.max(IDX) + 1
            IDX = ['Cluster_' + str(I) for I in IDX]
            df_plot['Cluster'] = IDX

        if freq_weight is True:
            s = freq * scale
        else:
            s = scale

        df_plot['s']=s

        if show_legend is True:
            legend = 'full'
        else:
            legend = False

        if by_class is True:
            hue = 'Class'
        elif by_cluster is True:
            hue = 'Cluster'
        elif by_sample is True:
            hue = 'Sample'
        else:
            hue = None

        if set == 'all':
            df_plot_sel = df_plot
        elif set == 'train':
            df_plot_sel = df_plot[df_plot['Set']=='train']
        elif set == 'valid':
            df_plot_sel = df_plot[df_plot['Set']=='valid']
        elif set == 'test':
            df_plot_sel = df_plot[df_plot['Set']=='test']

        df_plot_sel = df_plot_sel.sample(frac=1)
        print(df_plot_sel['Class'])
        plt.figure()
        sns.scatterplot(data=df_plot_sel, x='x', y='y', s=df_plot_sel['s'], hue=hue, legend=legend, alpha=alpha, linewidth=0.0)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('')
        plt.ylabel('')

        if filename is not None:
            plt.savefig(os.path.join(self.directory_results, filename))

        if prob_plot is not None:
            plt.figure()
            plt.scatter(df_plot_sel['x'],df_plot_sel['y'],c=df_plot_sel['Predicted'],s=df_plot_sel['s'],
                    alpha=alpha,cmap='bwr')
            plt.xticks([])
            plt.yticks([])
            plt.xlabel('')
            plt.ylabel('')

            if filename is not None:
                plt.savefig(os.path.join(self.directory_results, 'prob_'+filename))

        if plot_by_class is True:
            for i in self.lb.classes_:
                sel = df_plot_sel['Class']==i
                plt.figure()
                sns.scatterplot(data=df_plot_sel[sel], x='x', y='y', s=df_plot_sel['s'][sel], hue=hue, legend=legend, alpha=alpha,
                                linewidth=0.0)
                plt.xticks([])
                plt.yticks([])
                plt.xlabel('')
                plt.ylabel('')
                
        self.umap = X_2

class DeepTCR_U(DeepTCR_base,vis_class):

    def _reset_models(self):
        self.models_dir = os.path.join(self.Name,'models')
        if os.path.exists(self.models_dir):
            shutil.rmtree(self.models_dir)
        os.makedirs(self.models_dir)

    # latent_dim=64
    def Train_VAE(self,latent_dim=256, kernel = 5, trainable_embedding=True, embedding_dim_aa = 64, embedding_dim_genes = 48, embedding_dim_rna = 1024,
                  use_only_seq=False,use_only_gene=False,size_of_net='medium',latent_alpha=1e-3,rna_alpha=1, gene_alpha=1, seq_alpha=1, var_explained=None,graph_seed=2022,
                  batch_size=64, epochs_min=0,stop_criterion=0.01,stop_criterion_window=30, accuracy_min=None,
                  suppress_output = False,learning_rate=0.001,split_seed=2022, include_RNA = True):

        """
        # Train Variational Autoencoder (VAE)
        This method trains the network and saves features values for sequences for a variety of downstream analyses that can either be done within the DeepTCR framework or by the user by simplying extracting out the learned representations.
        Args:
            latent_dim (int): Number of latent dimensions for VAE.
            kernel (int): The motif k-mer of the first convolutional layer of the graph.
            trainable_embedding (bool): Toggle to control whether a trainable embedding layer is used or native one-hot representation for convolutional layers.
            embedding_dim_aa (int): Learned latent dimensionality of amino-acids.
            embedding_dim_genes (int): Learned latent dimensionality of VDJ genes
            embedding_dim_rna (int): Learned latent dimensionality of RNA
            use_only_seq (bool): To only use sequence feaures, set to True.
            use_only_gene (bool): To only use gene-usage features, set to True.
            size_of_net (list or str): The convolutional layers of this network have 3 layers for which the use can modify the number of neurons per layer. The user can either specify the size of the network with the following options:
                - small == [12,32,64] neurons for the 3 respective layers
                - medium == [32,64,128] neurons for the 3 respective layers
                - large == [64,128,256] neurons for the 3 respective layers
                - custom, where the user supplies a list with the number of nuerons for the respective layers
                    i.e. [3,3,3] would have 3 neurons for all 3 layers.
                    One can also adjust the number of layers for the convolutional stack by changing the length of
                    this list. [3,3,3] = 3 layers, [3,3,3,3] = 4 layers.
            latent_alpha (float): Penalty coefficient for latent loss. This value changes the degree of latent regularization on the VAE.
            var_explained (float (0-1.0)): Following training, one can select the number of latent features that explain N% of the variance in the data. The output of the method will be the features in order of the explained variance.
            graph_seed (int): For deterministic initialization of weights of the graph, set this to value of choice.
            batch_size (int): Size of batch to be used for each training iteration of the net.
            epochs_min (int): The minimum number of epochs to train the autoencoder.
            stop_criterion (float): Minimum percent decrease in determined interval (below) to continue training. Used as early stopping criterion.
            stop_criterion_window (int): The window of data to apply the stopping criterion.
            accuracy_min (float): Minimum reconstruction accuracy before terminating training.
            suppress_output (bool): To suppress command line output with training statisitcs, set to True.
            split_seed (int): For deterministic batching of data during training, one can set this parameter to value of choice.
        Returns:
            VAE Features
            - self.features (array):
            An array that contains n x latent_dim containing features for all sequences
            - self.explained_variance_ (array):
            The explained variance for the N number of latent features in order of descending value.
            - self.explained_variance_ratio_ (array):
            The explained variance ratio for the N number of latent features in order of descending value.
        ---------------------------------------
        """

        GO = graph_object()
        GO.size_of_net = size_of_net
        GO.embedding_dim_genes = embedding_dim_genes
        GO.embedding_dim_aa = embedding_dim_aa
        GO.embedding_dim_rna = embedding_dim_rna
        GO.l2_reg = 0.0

        graph_model_AE = tf.Graph()
        with graph_model_AE.device(self.device):
            with graph_model_AE.as_default():
                if graph_seed is not None:
                    tf.compat.v1.set_random_seed(graph_seed)

                GO.net = 'ae'
                GO.Features = Conv_Model(GO, self, trainable_embedding, kernel, use_only_seq, use_only_gene, include_RNA)
                # fc = tf.compat.v1.layers.dense(GO.Features, 256)
                fc = tf.compat.v1.layers.dense(GO.Features, 512, tf.nn.relu)
                fc = tf.compat.v1.layers.dense(fc, latent_dim, tf.nn.relu)
                # z_mean = tf.compat.v1.layers.dense(GO.Features, latent_dim, tf.nn.relu)
                # z_log_var = tf.compat.v1.layers.dense(GO.Features, latent_dim, tf.nn.relu)
                z_w = tf.compat.v1.get_variable(name='z_w',shape=[latent_dim,latent_dim])
                z_mean = tf.matmul(fc,z_w)
                z_log_var = tf.compat.v1.layers.dense(fc, latent_dim, activation=tf.nn.softplus, name='z_log_var')
                latent_cost = Latent_Loss(z_log_var,z_mean,alpha=latent_alpha)

                z = z_mean + tf.exp(z_log_var / 2) * tf.random.normal(tf.shape(input=z_mean), 0.0, 1.0, dtype=tf.float32)
                z = tf.identity(z, name='z')

                # fc_up = tf.compat.v1.layers.dense(z, 128)
                # fc_up = tf.compat.v1.layers.denGet_RNA_Lossse(fc_up, 256)
                fc_up = tf.compat.v1.layers.dense(z, 512, tf.nn.relu)
                # fc_up = tf.compat.v1.layers.dense(z, embedding_dim_rna, tf.nn.relu)
                fc_up_flat = fc_up
                # fc_up = tf.reshape(fc_up, shape=[-1, 1, 4, 64])
                fc_up = tf.reshape(fc_up, shape=[-1, 1, 4, 128])
                # fc_up = tf.reshape(fc_up, shape=[-1, 1, 4, 100])

                ## RNA
                rna_loss = [Get_RNA_Loss(fc_up_flat, GO.embedding_dim_rna, GO.mat, alpha=rna_alpha)]

                ## CDR3
                seq_losses = []
                seq_accuracies = []
                if size_of_net == 'small':
                    units = [12, 32, 64]
                elif size_of_net == 'medium':
                    units = [32, 64, 128]
                elif size_of_net == 'large':
                    units = [64, 128, 256]
                else:
                    units = size_of_net

                if self.use_beta:
                    upsample_beta = fc_up
                    for _ in range(len(units)-1):
                        upsample_beta = tf.compat.v1.layers.conv2d_transpose(upsample_beta, units[-1-_], (1, 3), (1, 2),activation=tf.nn.relu)

                    kr, str = determine_kr_str(upsample_beta, GO, self)

                    if trainable_embedding is True:
                        upsample3_beta = tf.compat.v1.layers.conv2d_transpose(upsample_beta, GO.embedding_dim_aa, (1, kr),(1, str), activation=tf.nn.relu)
                        upsample3_beta = upsample3_beta[:,:,0:self.max_length,:]

                        embedding_layer_seq_back = tf.transpose(a=GO.embedding_layer_seq, perm=(0, 1, 3, 2))
                        logits_AE_beta = tf.squeeze(tf.tensordot(upsample3_beta, embedding_layer_seq_back, axes=(3, 2)),axis=(3, 4), name='logits')
                    else:
                        logits_AE_beta = tf.compat.v1.layers.conv2d_transpose(upsample_beta, 23, (1, kr),(1, str), activation=tf.nn.relu)
                        logits_AE_beta = logits_AE_beta[:,:,0:self.max_length,:]

                    recon_cost_beta = Recon_Loss(GO.X_Seq_beta, logits_AE_beta,alpha=seq_alpha)
                    seq_losses.append(recon_cost_beta)

                    predicted_beta = tf.squeeze(tf.argmax(input=logits_AE_beta, axis=3), axis=1)
                    actual_ae_beta = tf.squeeze(GO.X_Seq_beta, axis=1)
                    w = tf.cast(tf.squeeze(tf.greater(GO.X_Seq_beta, 0), 1), tf.float32)
                    correct_ae_beta = tf.reduce_sum(input_tensor=w * tf.cast(tf.equal(predicted_beta, actual_ae_beta), tf.float32),axis=1) / tf.reduce_sum(input_tensor=w, axis=1)

                    accuracy_beta = tf.reduce_mean(input_tensor=correct_ae_beta, axis=0)
                    seq_accuracies.append(accuracy_beta)

                if self.use_alpha:
                    upsample_alpha = fc_up
                    for _ in range(len(units)-1):
                        upsample_alpha = tf.compat.v1.layers.conv2d_transpose(upsample_alpha, units[-1-_], (1, 3), (1, 2),activation=tf.nn.relu)

                    kr, str = determine_kr_str(upsample_alpha, GO, self)

                    if trainable_embedding is True:
                        upsample3_alpha = tf.compat.v1.layers.conv2d_transpose(upsample_alpha, GO.embedding_dim_aa, (1, kr), (1, str),activation=tf.nn.relu)
                        upsample3_alpha = upsample3_alpha[:,:,0:self.max_length,:]

                        embedding_layer_seq_back = tf.transpose(a=GO.embedding_layer_seq, perm=(0, 1, 3, 2))
                        logits_AE_alpha = tf.squeeze(tf.tensordot(upsample3_alpha, embedding_layer_seq_back, axes=(3, 2)),axis=(3, 4), name='logits')
                    else:
                        logits_AE_alpha = tf.compat.v1.layers.conv2d_transpose(upsample_alpha, 23, (1, kr), (1, str),activation=tf.nn.relu)
                        logits_AE_alpha = logits_AE_alpha[:,:,0:self.max_length,:]

                    recon_cost_alpha = Recon_Loss(GO.X_Seq_alpha, logits_AE_alpha,alpha=seq_alpha)
                    seq_losses.append(recon_cost_alpha)

                    predicted_alpha = tf.squeeze(tf.argmax(input=logits_AE_alpha, axis=3), axis=1)
                    actual_ae_alpha = tf.squeeze(GO.X_Seq_alpha, axis=1)
                    w = tf.cast(tf.squeeze(tf.greater(GO.X_Seq_alpha, 0), 1), tf.float32)
                    correct_ae_alpha = tf.reduce_sum(input_tensor=w * tf.cast(tf.equal(predicted_alpha, actual_ae_alpha), tf.float32), axis=1) / tf.reduce_sum(input_tensor=w, axis=1)
                    accuracy_alpha = tf.reduce_mean(input_tensor=correct_ae_alpha, axis=0)
                    seq_accuracies.append(accuracy_alpha)
                
                ## Gene
                gene_loss = []
                gene_accuracies = []
                if self.use_v_beta is True:
                    v_beta_loss,v_beta_acc = Get_Gene_Loss(fc_up_flat,GO.embedding_layer_v_beta,GO.X_v_beta_OH,alpha=gene_alpha)
                    gene_loss.append(v_beta_loss)
                    gene_accuracies.append(v_beta_acc)

                if self.use_d_beta is True:
                    d_beta_loss, d_beta_acc = Get_Gene_Loss(fc_up_flat,GO.embedding_layer_d_beta,GO.X_d_beta_OH,alpha=gene_alpha)
                    gene_loss.append(d_beta_loss)
                    gene_accuracies.append(d_beta_acc)

                if self.use_j_beta is True:
                    j_beta_loss,j_beta_acc = Get_Gene_Loss(fc_up_flat,GO.embedding_layer_j_beta,GO.X_j_beta_OH,alpha=gene_alpha)
                    gene_loss.append(j_beta_loss)
                    gene_accuracies.append(j_beta_acc)

                if self.use_v_alpha is True:
                    v_alpha_loss,v_alpha_acc = Get_Gene_Loss(fc_up_flat,GO.embedding_layer_v_alpha,GO.X_v_alpha_OH,alpha=gene_alpha)
                    gene_loss.append(v_alpha_loss)
                    gene_accuracies.append(v_alpha_acc)

                if self.use_j_alpha is True:
                    j_alpha_loss,j_alpha_acc = Get_Gene_Loss(fc_up_flat,GO.embedding_layer_j_alpha,GO.X_j_alpha_OH,alpha=gene_alpha)
                    gene_loss.append(j_alpha_loss)
                    gene_accuracies.append(j_alpha_acc)

                recon_losses = seq_losses + gene_loss + rna_loss

                accuracies = seq_accuracies + gene_accuracies

                if use_only_gene:
                    recon_losses = gene_loss
                    accuracies = gene_accuracies
                if use_only_seq:
                    recon_losses = seq_losses
                    accuracies = seq_accuracies

                temp = []
                temp_seq = []
                temp_gene = []
                temp_rna = []

                for l in recon_losses:
                    l = l[:,tf.newaxis]
                    temp.append(l)
                for l_seq in seq_losses:
                    l_seq = l_seq[:,tf.newaxis]
                    temp_seq.append(l_seq)
                for l_gene in gene_loss:
                    l_gene = l_gene[:,tf.newaxis]
                    temp_gene.append(l_gene)
                for l_rna in rna_loss:
                    l_rna = l_rna[:,tf.newaxis]
                    temp_rna.append(l_rna)
                    
                recon_losses = temp
                recon_losses = tf.concat(recon_losses,1)
                recon_cost = tf.reduce_sum(input_tensor=recon_losses, axis=1)
                recon_cost = tf.reduce_mean(input_tensor=recon_cost)

                temp_seq = tf.concat(temp_seq,1)
                seq_cost = tf.reduce_sum(input_tensor=temp_seq,axis=1)
                seq_cost = tf.reduce_mean(input_tensor=seq_cost)
                temp_gene = tf.concat(temp_gene,1)
                gene_cost = tf.reduce_sum(input_tensor=temp_gene,axis=1)
                gene_cost = tf.reduce_mean(input_tensor=gene_cost)
                temp_rna = tf.concat(temp_rna,1)
                rna_cost = tf.reduce_sum(input_tensor=temp_rna,axis=1)
                rna_cost = tf.reduce_mean(input_tensor=rna_cost)

                total_cost = [recon_losses,latent_cost[:,tf.newaxis]]
                total_cost = tf.concat(total_cost,1)
                total_cost = tf.reduce_sum(input_tensor=total_cost,axis=1)
                total_cost = tf.reduce_mean(input_tensor=total_cost)

                num_acc = len(accuracies)
                accuracy = 0
                for a in accuracies:
                    accuracy += a
                accuracy = accuracy/num_acc
                latent_cost = tf.reduce_mean(input_tensor=latent_cost)

                opt_ae = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_cost)

                GO.saver = tf.compat.v1.train.Saver(max_to_keep=None)

        self._reset_models()
        tf.compat.v1.reset_default_graph()
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.compat.v1.Session(graph=graph_model_AE,config=config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            stop_check_list = []
            accuracy_list = []
            recon_loss = []
            train_loss = []
            latent_loss = []
            seq_loss_lst = []
            gene_loss_lst = []
            rna_loss_lst = []
            training = True
            e = 0
            while training:
                iteration = 0
                Vars = [self.X_Seq_alpha,self.X_Seq_beta,self.v_beta_num,self.d_beta_num,self.j_beta_num,
                        self.v_alpha_num,self.j_alpha_num, self.mat]

                if split_seed is not None:
                    np.random.seed(split_seed)

                for vars in get_batches(Vars, batch_size=batch_size,random=True):
                    feed_dict = {}
                    if self.use_alpha is True:
                        feed_dict[GO.X_Seq_alpha] = vars[0]
                        
                    if self.use_beta is True:
                        feed_dict[GO.X_Seq_beta] = vars[1]

                    if self.use_v_beta is True:
                        feed_dict[GO.X_v_beta] = vars[2]

                    if self.use_d_beta is True:
                        feed_dict[GO.X_d_beta] = vars[3]

                    if self.use_j_beta is True:
                        feed_dict[GO.X_j_beta] = vars[4]

                    if self.use_v_alpha is True:
                        feed_dict[GO.X_v_alpha] = vars[5]

                    if self.use_j_alpha is True:
                        feed_dict[GO.X_j_alpha] = vars[6]
                    
                    feed_dict[GO.mat] = vars[7]
                    
                    train_loss_i, recon_loss_i, seq_losses_i, gene_loss_i, rna_loss_i, latent_loss_i, accuracy_i, _ = sess.run([total_cost, recon_cost, seq_cost, gene_cost, rna_cost, latent_cost, accuracy, opt_ae], feed_dict=feed_dict)
                    accuracy_list.append(accuracy_i)
                    recon_loss.append(recon_loss_i)
                    latent_loss.append(latent_loss_i)
                    train_loss.append(train_loss_i)
                    seq_loss_lst.append(seq_losses_i)
                    gene_loss_lst.append(gene_loss_i)
                    rna_loss_lst.append(rna_loss_i)
   
                    if suppress_output is False:
                        print("Epoch = {}, Iteration = {}".format(e,iteration),
                                "Total Loss: {:.5f}:".format(train_loss_i),
                                "Recon Loss: {:.5f}:".format(recon_loss_i),
                                "CDR3 Loss: {:.5f}:".format(seq_losses_i),
                                "Gene Loss: {:.5f}:".format(gene_loss_i),
                                "RNA Loss: {:.5f}:".format(rna_loss_i),
                                "Latent Loss: {:.5f}:".format(latent_loss_i),
                                "Recon Accuracy: {:.5f}".format(accuracy_i))

                    if e >= epochs_min:
                        if accuracy_min is not None:
                            if np.mean(accuracy_list[-10:]) > accuracy_min:
                                training = False
                                break
                        else:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                stop_check_list.append(stop_check(train_loss,stop_criterion,stop_criterion_window))
                                if np.sum(stop_check_list[-3:]) >= 3:
                                    training = False
                                    break
                    iteration += 1
                e += 1

            features_list = []
            accuracy_list = []
            alpha_features_list = []
            alpha_indices_list = []
            beta_features_list = []
            beta_indices_list = []
            RNA_list = []
            Vars = [self.X_Seq_alpha, self.X_Seq_beta, self.v_beta_num, self.d_beta_num, self.j_beta_num,
                    self.v_alpha_num, self.j_alpha_num, self.mat]

            for vars in get_batches(Vars, batch_size=batch_size, random=False):
                feed_dict = {}
                if self.use_alpha is True:
                    feed_dict[GO.X_Seq_alpha] = vars[0]
                if self.use_beta is True:
                    feed_dict[GO.X_Seq_beta] = vars[1]

                if self.use_v_beta is True:
                    feed_dict[GO.X_v_beta] = vars[2]

                if self.use_d_beta is True:
                    feed_dict[GO.X_d_beta] = vars[3]

                if self.use_j_beta is True:
                    feed_dict[GO.X_j_beta] = vars[4]

                if self.use_v_alpha is True:
                    feed_dict[GO.X_v_alpha] = vars[5]

                if self.use_j_alpha is True:
                    feed_dict[GO.X_j_alpha] = vars[6]

                feed_dict[GO.mat] = vars[7]

                get = z_mean
                features_ind, accuracy_check = sess.run([get, accuracy], feed_dict=feed_dict)
                features_list.append(features_ind)
                accuracy_list.append(accuracy_check)

                if self.use_alpha is True:
                    alpha_ft, alpha_i = sess.run([GO.alpha_out,GO.indices_alpha],feed_dict=feed_dict)
                    alpha_features_list.append(alpha_ft)
                    alpha_indices_list.append(alpha_i)

                if self.use_beta is True:
                    beta_ft, beta_i = sess.run([GO.beta_out,GO.indices_beta],feed_dict=feed_dict)
                    beta_features_list.append(beta_ft)
                    beta_indices_list.append(beta_i)

                rna_lowdim = sess.run([GO.rna_features],feed_dict=feed_dict)
                RNA_list.append(rna_lowdim)

            features = np.vstack(features_list)
            accuracy_list = np.hstack(accuracy_list)
            if self.use_alpha is True:
                self.alpha_features = np.vstack(alpha_features_list)
                self.alpha_indices = np.vstack(alpha_indices_list)

            if self.use_beta is True:
                self.beta_features = np.vstack(beta_features_list)
                self.beta_indices = np.vstack(beta_indices_list)

            self.rna_lowdim = np.vstack(RNA_list[0])    

            self.kernel = kernel
            #
#             if self.use_alpha is True:
#                 var_save = [self.alpha_features, self.alpha_indices, self.alpha_sequences]
#                 with open(os.path.join(self.Name, 'alpha_features.pkl'), 'wb') as f:
#                     pickle.dump(var_save, f)

#             if self.use_beta is True:
#                 var_save = [self.beta_features, self.beta_indices, self.beta_sequences]
#                 with open(os.path.join(self.Name, 'beta_features.pkl'), 'wb') as f:
#                     pickle.dump(var_save, f)
            
#             with open(os.path.join(self.Name, 'rna_features.pkl'), 'wb') as f:
#                     pickle.dump(self.rna_lowdim, f)

#             with open(os.path.join(self.Name, 'kernel.pkl'), 'wb') as f:
#                 pickle.dump(self.kernel, f)


            print('Reconstruction Accuracy: {:.5f}'.format(np.nanmean(accuracy_list)))

            embedding_layers = [GO.embedding_layer_v_alpha,GO.embedding_layer_j_alpha,
                                GO.embedding_layer_v_beta,GO.embedding_layer_d_beta,
                                GO.embedding_layer_j_beta]
            embedding_names = ['v_alpha','j_alpha','v_beta','d_beta','j_beta']
            name_keep = []
            embedding_keep = []
            for n,layer in zip(embedding_names,embedding_layers):
                if layer is not None:
                    embedding_keep.append(layer.eval())
                    name_keep.append(n)

            embed_dict = dict(zip(name_keep,embedding_keep))

            # sort features by variance explained
            cov = np.cov(features.T)
            explained_variance = np.diag(cov)
            ind = np.flip(np.argsort(explained_variance))
            explained_variance = explained_variance[ind]
            explained_variance_ratio = explained_variance / np.sum(explained_variance)
            features = features[:, ind]

            if var_explained is not None:
                features = features[:, 0:np.where(np.cumsum(explained_variance_ratio) > var_explained)[0][0] + 1]

            self.ind = ind[:features.shape[1]]
            #save model data and information for inference engine
#             save_model_data(self,GO.saver,sess,name='VAE',get=z_mean)

        with open(os.path.join(self.Name,'VAE_features.pkl'), 'wb') as f:
            pickle.dump([features,embed_dict,explained_variance,explained_variance_ratio], f,protocol=4)

        self.features = features
        self.embed_dict = embed_dict
        self.explained_variance_ = explained_variance
        self.explained_variance_ratio_ = explained_variance_ratio
        self.loss = train_loss
        self.latent_loss = latent_loss
        self.recon_loss = recon_loss
        
        with open(os.path.join(self.Name,'loss_accuracy.pkl'), 'wb') as f:
            pickle.dump([train_loss,latent_loss,recon_loss, accuracy_list], f,protocol=4)
            
        print('Training Done')
    