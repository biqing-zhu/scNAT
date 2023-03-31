from sklearn.metrics import pairwise_distances
import numpy as np
import scipy.sparse as sps
from scipy import stats
from collections import Counter
from scipy.stats import mannwhitneyu
import pandas as pd
import matplotlib.pyplot as plt

SUBJECT_ID_OBS_KEY = 'subject_id'
mannwhitneyu_kwargs = {'method':'asymptotic'}
HELP_SUFFIX = '_help'
tcr_keys =  ['v_gene_A', 'j_gene_A', 'cdr3_A', 'v_gene_B', 'j_gene_B', 'cdr3_B']

def setup_uns_dicts(adata):
    if 'conga_results' not in adata.uns_keys():
        adata.uns['conga_results'] = {}

    if 'conga_stats' not in adata.uns_keys():
        adata.uns['conga_stats'] = {}

def _get_split_mean_var_tcr( X, X_sq, mask, mean, mean_sq ):
    ''' Helper function to quickly compute mean and variance
    '''
    # use: var = (mean_sq - mean**2)
    #
    N = mask.shape[0]
    assert X.shape[0] == N
    wt_fg = np.sum(mask) / N
    wt_bg = 1. - wt_fg
    mean_fg = X[mask].mean(axis=0)
    mean_sq_fg = X_sq[mask].mean(axis=0)
    if sps.issparse(X):
        mean_fg = mean_fg.A1
        mean_sq_fg = mean_sq_fg.A1
    mean_bg = ((mean - wt_fg*mean_fg)/wt_bg)
    mean_sq_bg = ((mean_sq - wt_fg*mean_sq_fg)/wt_bg)
    var_fg = (mean_sq_fg - mean_fg**2)
    var_bg = (mean_sq_bg - mean_bg**2)
    return mean_fg, var_fg, mean_bg, var_bg

def _get_split_mean_var_rna( X, X_sq, mask, mean, mean_sq ):
    ''' Helper function to quickly compute mean and variance
    '''
    # use: var = (mean_sq - mean**2)
    #
    N = mask.shape[0]
    assert X.shape[0] == N
    wt_fg = np.sum(mask) / N
    wt_bg = 1. - wt_fg
    mean_fg = X[mask].mean(axis=0)
    mean_sq_fg = X_sq[mask].mean(axis=0)
    if sps.issparse(X):
        mean_fg = mean_fg.A1
        mean_sq_fg = mean_sq_fg.A1
    mean_bg = ((mean - wt_fg*mean_fg)/wt_bg).A1
    mean_sq_bg = ((mean_sq - wt_fg*mean_sq_fg)/wt_bg).A1
    var_fg = (mean_sq_fg - mean_fg**2)
    var_bg = (mean_sq_bg - mean_bg**2)
    return mean_fg, var_fg, mean_bg, var_bg

def nbrhood_rank_genes_fast(
        adata,
        nbrs,
        pval_threshold,
        top_n=50,
        min_num_fg=3,
        ttest_pval_threshold_for_mwu_calc=None,
):
    ''' Run graph-vs-features analysis comparing the TCR neighbor graph
    to GEX features (ie, expression levels of the different genes)
    Modeled on scanpy rank_genes_groups
    All pvals are crude bonferroni corrected for:
    * number of non-empty nbrhoods in nbrs_tcr and number of genes in adata.raw.X with at least 3 nonzero cells
      (see pval_rescale below)
    returns a pandas dataframe with the results
    -OR- if also_return_help_message is True
    return results_df, help_message
    '''

    if ttest_pval_threshold_for_mwu_calc is None:
        ttest_pval_threshold_for_mwu_calc = pval_threshold * 10

    rankby_abs = False # following scanpy: this means that we only look at enriched/upregulated/higher score values

    num_clones = adata.shape[0]

    genes = list(adata.raw.var_names)
    X = adata.layers['counts_norm'].copy()
    if not sps.issparse(X):
        X = sps.csr_matrix(X)
    assert X.shape[1] == len(genes)

    X_csc = X.tocsc()

    X_sq = X.multiply(X)

    mean = X.mean(axis=0)
    mean_sq = X_sq.mean(axis=0)

    num_nonempty_nbrhoods = sum(1 for x in nbrs if len(x)>0)

    # len(genes) is probably too hard since lots of the genes are all zeros
    min_nonzero_cells = 3
    gene_nonzero_counts = Counter( X.nonzero()[1] )
    bad_gene_mask = np.array([gene_nonzero_counts[x] < min_nonzero_cells
                              for x in range(len(genes))])
    n_genes_eff = np.sum(~bad_gene_mask)
    pval_rescale = num_nonempty_nbrhoods * n_genes_eff

    results = []

    reference_indices = np.arange(len(genes), dtype=int)

    for ii in range(num_clones):
        if len(nbrs[ii])==0:
            continue
        nbrhood_mask = np.full( (num_clones,), False)
        nbrhood_mask[ nbrs[ii] ] = True
        nbrhood_mask[ ii ] = True

        mean_fg, var_fg, mean_bg, var_bg = _get_split_mean_var_rna(
            X, X_sq, nbrhood_mask, mean, mean_sq)

        num_fg = np.sum(nbrhood_mask)
        if num_fg < min_num_fg:
            continue

        scores, pvals = stats.ttest_ind_from_stats(
            mean1=mean_fg,
            std1=np.sqrt(np.maximum(var_fg, 1e-12)),
            nobs1=num_fg,
            mean2=mean_bg,
            std2=np.sqrt(np.maximum(var_bg, 1e-12)),
            nobs2=num_clones-num_fg,
            equal_var=False  # Welch's
        )

        # scanpy code:
        scores[np.isnan(scores)] = 0.
        pvals [np.isnan(pvals)] = 1.
        pvals [bad_gene_mask] = 1.
        logfoldchanges = np.log2((np.expm1(mean_fg) + 1e-9) /
                                 (np.expm1(mean_bg) + 1e-9))

        pvals_adj = pvals * pval_rescale

        scores_sort = np.abs(scores) if rankby_abs else scores
        partition = np.argpartition(scores_sort, -top_n)[-top_n:]
        partial_indices = np.argsort(scores_sort[partition])[::-1]
        global_indices = reference_indices[partition][partial_indices]

        for ind in global_indices:
            gene = genes[ind]
            pval_adj = pvals_adj[ind]
            log2fold= logfoldchanges[ind]

            if pval_adj > ttest_pval_threshold_for_mwu_calc:
                continue

            # here we are looking for genes (or clone_sizes/inverted nndists)
            #  that are LARGER in the forground (fg)
            col = X_csc[:,ind][nbrhood_mask]
            noncol = X_csc[:,ind][~nbrhood_mask]
            _, mwu_pval = mannwhitneyu( col.toarray()[:,0], noncol.toarray()[:,0],
                                            alternative='greater',
                                            **mannwhitneyu_kwargs )

            mwu_pval_adj = mwu_pval * pval_rescale

            # 2021-06-28 make this more stringent: it used to be either/or
            #if min(mwu_pval_adj, pval_adj) < pval_threshold:
            #
            # sometimes MWU seems a little wonky, so allow good ttests also
            # if MWU is not terrible
            if ((mwu_pval_adj < pval_threshold) or
                (pval_adj < pval_threshold and
                 mwu_pval_adj < 10*pval_threshold)):
                
                results.append(dict(ttest_pvalue_adj=pval_adj,
                                    mwu_pvalue_adj=mwu_pval_adj,
                                    log2enr=log2fold,
                                    feature=gene,
                                    mean_fg=mean_fg[ind],
                                    mean_bg=mean_bg[ind],
                                    num_fg=num_fg,
                                    clone_index=ii))

    results_df = pd.DataFrame(results)

    return results_df

def save_table_and_helpfile(
        table_tag,
        adata,
        outfile_prefix,
        data_type
):
    results = adata.uns['conga_results'][data_type]
    tsvfile = f'{outfile_prefix}_{table_tag}_{data_type}.tsv'
    results.to_csv(tsvfile, sep='\t', index=False)

def retrieve_tcrs_from_adata(tcr):

    global tcr_keys
    tcrs = []
    arrays = [ tcr[x] for x in tcr_keys ]
    for v_gene_A, j_gene_A, cdr3_A, v_gene_B, j_gene_B, cdr3_B in zip(*arrays):
            tcrs.append(((v_gene_A, j_gene_A, cdr3_A),
                         (v_gene_B, j_gene_B, cdr3_B) ) )

    return tcrs

def cdr3len_score_tcr(tcr):
    ''' double-weight the beta chain
    '''
    return len(tcr[0][2]) + 2*len(tcr[1][2])
    
def make_tcr_score_table(adata, tcr_genes, scorenames, vdj_dic):
    ''' Returns a numpy array of the tcr scores with shape:
           (adata.shape[0], len(scorenames))
    '''

    cols = []
    for name in scorenames:
        for i_ab,ab in enumerate('AB'):
            for i_vj,vj in enumerate('VJ'):
                ii_count_reps = set(vdj_dic[vj+ab])
                if name in ii_count_reps:
                    cols.append(
                            [float(x[i_ab][i_vj]==name) for x in tcr_genes])
    # assert matched

    table = np.array(cols).transpose()#[:,np.newaxis]

    assert table.shape == (adata.shape[0], len(scorenames))

    return table

def gex_nbrhood_rank_tcr_scores(
        adata,
        nbrs,
        tcr_genes,
        genes_keep,
        vdj_dic,
        pval_threshold,
        min_num_fg=3,
        ttest_pval_threshold_for_mwu_calc=None
):
    ''' Run graph-vs-features analysis comparing the GEX neighbor graph
    to TCR features.
    We also use this to find differential TCR features for conga clusters,
    by passing in a fake GEX neighbor graph.
    pvalues are bonferroni corrected (actually just multiplied by numtests)
    returns a pandas dataframe with the results,
    OR if also_return_help_message,
    returns results, help_message
    '''
    num_clones = adata.shape[0]

    assert len(genes_keep) == len(set(genes_keep)) # no dups

    if ttest_pval_threshold_for_mwu_calc is None:
        ttest_pval_threshold_for_mwu_calc = pval_threshold*10

    print('making tcr score table, #features=', len(genes_keep))
    score_table = make_tcr_score_table(adata, tcr_genes, genes_keep, vdj_dic)
    score_table_sq = np.multiply(score_table, score_table)
    mean = score_table.mean(axis=0)
    mean_sq = score_table_sq.mean(axis=0)

    pval_rescale = adata.shape[0] * len(genes_keep)

    results = []
    nbrhood_mask = np.full( (num_clones,), False)

    for ii in range(num_clones):
        if len(nbrs[ii])==0:
            continue
        nbrhood_mask.fill(False)
        nbrhood_mask[ nbrs[ii] ] = True
        nbrhood_mask[ ii ] = True

        mean_fg, var_fg, mean_bg, var_bg = _get_split_mean_var_tcr(
            score_table, score_table_sq, nbrhood_mask, mean, mean_sq)
        num_fg = np.sum(nbrhood_mask)
        if num_fg < min_num_fg:
            continue
        scores, pvals = stats.ttest_ind_from_stats(
            mean1=mean_fg,
            std1=np.sqrt(np.maximum(var_fg, 1e-12)),
            nobs1=num_fg,
            mean2=mean_bg,
            std2=np.sqrt(np.maximum(var_bg, 1e-12)),
            nobs2=num_clones-num_fg,
            equal_var=False,  # Welch's
        )

        scores[np.isnan(scores)] = 0
        pvals[np.isnan(pvals)] = 1

        # crude bonferroni
        pvals *= pval_rescale

        for ind in np.argsort(pvals):
            pval = pvals[ind]

            if pval>ttest_pval_threshold_for_mwu_calc:
                continue

            _,mwu_pval = mannwhitneyu(score_table[:,ind][nbrhood_mask],
                                      score_table[:,ind][~nbrhood_mask],
                                      alternative='two-sided',
                                      **mannwhitneyu_kwargs)
            mwu_pval_adj = mwu_pval * pval_rescale

            # make more stringent
            #if min(pval, mwu_pval_adj) <= pval_threshold:

            if ((mwu_pval_adj <= pval_threshold) or
                (pval <= pval_threshold and
                 mwu_pval_adj <= 10*pval_threshold)):

                # get info about the clones most contributing to this skewed
                #  score
                score_name = genes_keep[ind]
                score = scores[ind] # ie the t-statistic

                results.append(dict(ttest_pvalue_adj=pval,
                                    ttest_stat=score,
                                    mwu_pvalue_adj=mwu_pval_adj,
                                    num_fg=num_fg,
                                    mean_fg=mean_fg[ind],
                                    mean_bg=mean_bg[ind],
                                    feature=score_name,
                                    clone_index=ii) )

    results_df = pd.DataFrame(results)
    return results_df

def make_feature_panel_plots(
        adata,
        all_nbrs,
        results_df,
        pngfile,
        feature_type,
        tcr_genes=None,
        vdj_dic=None,
        max_panels_per_bicluster=3,
        max_pvalue=0.05,
        nrows=8, #6,
        ncols=5, #4,
        panel_width_inches=2.5,
        use_nbr_frac=None,
        title=None,
        cmap='viridis',
        sort_order=True, # plot points sorted by feature value
        point_size=10,
):
    ''' Assumes results_df has column mwu_pvalue_adj for sorting
    returns a help message string.
    '''
    assert feature_type in ['gex','tcr']

    if results_df.empty:
        return

    xy = adata.obsm['UMAP_2D']

    # first let's figure out how many panels we could have
    # sort the results by pvalue
    df = results_df.sort_values('mwu_pvalue_adj')# makes a copy

    nbr_frac_for_cluster_results = (max(all_nbrs.keys()) if use_nbr_frac is None
                                    else use_nbr_frac)

    seen = set()
    inds=[]
    for row in df.itertuples():
        if row.mwu_pvalue_adj > max_pvalue:
            break
        feature = row.feature
        if feature in seen:
            continue
        seen.add(feature)
        inds.append(row.Index)
    if not inds:
        print('no results to plot!')
        return

    if len(inds) < nrows*ncols:
        # make fewer panels
        nrows = max(1, int(np.sqrt(len(inds))))
        ncols = (len(inds)-1)//nrows + 1

    df = df.loc[inds[:nrows*ncols],:]
    assert df.shape[0] <= nrows*ncols

    # try to get the raw values
    feature_to_raw_values = {}

    var_names = list(adata.raw.var_names)
    features = set(df['feature'])

    for f in features:
        if feature_type=='gex' and f in var_names:
            print('feature_type is gex')
            # careful since TR/IG gene names are present in var_names but its
            # better to use the VDJ information
            feature_to_raw_values[f] = adata.layers['counts_norm'][:, var_names.index(f)].toarray()[:,0]
        else:
            assert feature_type=='tcr'
            feature_score_table = make_tcr_score_table(adata, tcr_genes, [f], vdj_dic)
            feature_to_raw_values[f] = feature_score_table[:,0]

    figsize= (ncols*panel_width_inches, nrows*panel_width_inches)
    plt.figure(figsize=figsize)
    plotno=0
    for row in df.itertuples():
        plotno+=1
        plt.subplot(nrows, ncols, plotno)

        feature = row.feature

        scores = np.array(feature_to_raw_values[feature])

        if sort_order:
            reorder = np.argsort(scores)
        else:
            reorder = np.arange(len(scores))

        row_nbr_frac = use_nbr_frac if use_nbr_frac is not None else \
                       nbr_frac_for_cluster_results if row.nbr_frac==0.0 else \
                       row.nbr_frac
        nbrs = all_nbrs[row_nbr_frac]
        assert nbrs.shape[0] == adata.shape[0]
         # this will not work for ragged nbr arrays right now
        num_neighbors = nbrs.shape[1]

        nbr_averaged_scores = (
            scores + scores[nbrs].sum(axis=1))/(num_neighbors+1)

        plt.scatter(xy[reorder,0], xy[reorder,1],
                    c=nbr_averaged_scores[reorder], cmap=cmap,
                    s=point_size)
        # plt.scatter(xy[:,0], xy[:,1], c=nbr_averaged_scores, cmap='coolwarm',
        #             s=15)
        plt.title('{} {:.1e}'.format(feature, row.mwu_pvalue_adj))
        plt.xticks([],[])
        plt.yticks([],[])
        plt.text(0.99, 0.01, f'K={num_neighbors} nbr-avged',
                 va='bottom', ha='right', fontsize=9,
                 transform=plt.gca().transAxes)
        plt.text(0.01, 0.99, f'{np.min(nbr_averaged_scores):.2f}',
                 va='top', ha='left', fontsize=8,
                 transform=plt.gca().transAxes)
        plt.text(0.99, 0.99, f'{np.max(nbr_averaged_scores):.2f}',
                 va='top', ha='right', fontsize=8,
                 transform=plt.gca().transAxes)
        if (plotno-1)//ncols == nrows-1:
            plt.xlabel('UMAP1')
        if plotno%ncols == 1:
            plt.ylabel('UMAP2')

    tl_rect = [0,0,1,1]
    if title is not None:
        title_height_inches = 0.25
        plt.suptitle(title)
        tl_rect = [0, 0, 1, 1.0-title_height_inches/figsize[1]]
    plt.tight_layout(rect=tl_rect)
    print('making:', pngfile)
    plt.savefig(pngfile)

    help_message = f"""Graph-versus-feature analysis was used to identify
        a set of {feature_type.upper()} features that showed biased distributions
        in the neighborhoods. This plot shows the distribution of the
        top-scoring {feature_type.upper()} features on the UMAP 2D landscape. The 
        features are ranked by 'mwu_pvalue_adj' ie Mann-Whitney-Wilcoxon adjusted 
        P value (raw P value * number of comparisons). At most 
        {max_panels_per_bicluster} features from clonotype neighbhorhoods
        in each (GEX,TCR) cluster pair are shown. The raw scores for each feature
        are averaged over the K nearest neighbors (K is indicated in the lower
        right corner of each panel) for each clonotype. The min and max
        nbr-averaged scores are shown in the upper corners of each panel.
        """
    if sort_order:
        help_message += "Points are plotted in order of increasing feature score.\n"

    return help_message

def make_figure_helpfile(
        figure_tag,
        adata,
):
    pngfile = adata.uns['conga_results'][figure_tag]
    help_message = adata.uns['conga_results'].get(
        figure_tag+HELP_SUFFIX, '')
    if help_message:
        helpfile = pngfile+'_README.txt'
        out = open(helpfile, 'w')
        print('writing help message to file:', helpfile)
        out.write(help_message+'\n')
        out.close()
    else:
        print('WARNING: no help message for figure_tag:', figure_tag)