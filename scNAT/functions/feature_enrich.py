from functions.utils_f import *
import numpy as np
from collections import Counter

GRAPH_VS_GEX_FEATURES = 'graph_vs_gex_features'
GRAPH_VS_GEX_FEATURES_PANELS = 'graph_vs_gex_features_panels'
GRAPH_VS_TCR_FEATURES_PLOT = 'graph_vs_tcr_features_plot'
HELP_SUFFIX = '_help'
GRAPH_VS_TCR_FEATURES_PANELS = 'graph_vs_tcr_features_panels'

def calc_nbrs(
        adata,
        nbr_fracs,
        obsm_tag_gex = 'z_vae',
        sort_nbrs = False,
):
    ''' returns dict mapping from nbr_frac to [nbrs_gex, nbrs_tcr]
    nbrs exclude self and any clones in same atcr group or btcr group
    '''
    all_nbrs = {}
    for nbr_frac in nbr_fracs:
        all_nbrs[nbr_frac] = [ None, None ]

    tag = 'gex'
    obsm_tag = obsm_tag_gex 

    print('compute D', tag, adata.shape[0])
    D = pairwise_distances( adata.obsm[obsm_tag], metric='euclidean' )
    # for ii,(a,b) in enumerate(zip(agroups, bgroups)):
    #         D[ii, (agroups==a) ] = 1e3
    #         D[ii, (bgroups==b) ] = 1e3

    for nbr_frac in nbr_fracs:
        num_neighbors = max(1, int(nbr_frac*adata.shape[0]))
        print('argpartitions:', nbr_frac, adata.shape[0], tag)
        nbrs = np.argpartition( D, num_neighbors-1 )[:,:num_neighbors] # will NOT include self in there

        if sort_nbrs:
            ar = np.arange(adata.shape[0])[:,None]
            inds = np.argsort(D[ar, nbrs])
            nbrs = nbrs[ar, inds]

        assert nbrs.shape == (adata.shape[0], num_neighbors)
        all_nbrs[nbr_frac] = nbrs

    return all_nbrs

def run_graph_vs_features(
        adata, tcr,
        all_nbrs,
        pval_threshold= 1., # 'pvalues' are raw_pvalue * num_tests
        outfile_prefix= None,
):
    ''' This runs graph-vs-features analysis comparing the TCR graph to
    GEX features and the GEX graph to TCR features. It also looks for genes
    that are associated with particular TCR V or J segments
    results are stored in adata.uns['conga_results'] under the tags
    GEX_GRAPH_VS_TCR_FEATURES
    TCR_GRAPH_VS_GEX_FEATURES
    TCR_GENES_VS_GEX_FEATURES
    and written to tsvfiles if outfile_prefix is not None
    '''

    setup_uns_dicts(adata) # make life easier

    nbr_fracs = sorted(all_nbrs.keys())

    min_gene_count = 5
    tcr_genes = retrieve_tcrs_from_adata(tcr)
    genes_va = []
    genes_ja = []
    genes_vb = []
    genes_jb = []

    for i_cell in range(tcr.shape[0]):
        genes_va = genes_va + [tcr_genes[i_cell][0][0]]
        genes_ja = genes_ja + [tcr_genes[i_cell][0][1]]
        genes_vb = genes_vb + [tcr_genes[i_cell][1][0]]
        genes_jb = genes_jb + [tcr_genes[i_cell][1][1]]

    vdj_dic = {'VA': genes_va, 'JA': genes_ja, 'VB': genes_vb, 'JB': genes_jb}
    genes_all = genes_va + genes_ja + genes_vb + genes_jb
    genes_keep = [x for x, counts in Counter(np.array(genes_all)).items() if counts >= min_gene_count]
    
    tcr_feature_results = []
    gex_feature_results = []
    for nbr_frac in nbr_fracs:
        nbrs = all_nbrs[nbr_frac]

        #### GRAPHS VS TCR FEATURES ###################
        results_df_tcr = gex_nbrhood_rank_tcr_scores(
            adata, nbrs, tcr_genes, genes_keep, vdj_dic, pval_threshold)
        results_df_tcr['nbr_frac'] = nbr_frac
        tcr_feature_results.append(results_df_tcr)
        
        #### GRAPHS VS GEX FEATURES ###################
        results_df_rna = nbrhood_rank_genes_fast(
            adata, nbrs, pval_threshold)
        results_df_rna['nbr_frac'] = nbr_frac
        gex_feature_results.append(results_df_rna)
        
    tcr_feature_results = pd.concat(tcr_feature_results, ignore_index=True)
    gex_feature_results = pd.concat(gex_feature_results, ignore_index=True)
    tcr_feature_results.sort_values('mwu_pvalue_adj', inplace=True)
    gex_feature_results.sort_values('mwu_pvalue_adj', inplace=True)
    
    adata.uns['conga_results']['TCR features'] = tcr_feature_results
    adata.uns['conga_results']['RNA features'] = gex_feature_results

    # write the tables to files:
    data_type = ['TCR features', 'RNA features']
    if outfile_prefix is not None:
        for type in data_type:
            save_table_and_helpfile(
                GRAPH_VS_GEX_FEATURES, adata, outfile_prefix, type)

    return # all done with graph-vs-features analysis

def make_graph_vs_features_plots(
        adata, 
        all_nbrs, 
        outfile_prefix, 
        tcr=None, 
):
    setup_uns_dicts(adata) # should already be done

    tcr_results = adata.uns['conga_results'].get(
        'TCR features', pd.DataFrame([]))
    rna_results = adata.uns['conga_results'].get(
        'RNA features', pd.DataFrame([]))

    ## graph vs gex features #####################3

    if not rna_results.empty:
        figure_tag = GRAPH_VS_GEX_FEATURES_PANELS
        pngfile = f'{outfile_prefix}_{figure_tag}.png'
        feature_type = 'gex'
        help_message = make_feature_panel_plots(
            adata, all_nbrs, rna_results, pngfile, feature_type,
            title=figure_tag)
        adata.uns['conga_results'][figure_tag] = pngfile
        adata.uns['conga_results'][figure_tag+HELP_SUFFIX] = help_message
        make_figure_helpfile(figure_tag, adata)

    else:
        print('conga.plotting.make_graph_vs_features_plots:: missing results',
              'dont forget to call conga.correlations.run_graph_vs_features')

    ## graph vs tcr features #####################
    if not tcr_results.empty:
        figure_tag = GRAPH_VS_TCR_FEATURES_PANELS
        pngfile = f'{outfile_prefix}_{figure_tag}.png'
        feature_type = 'tcr'
        tcr_genes = retrieve_tcrs_from_adata(tcr)
        genes_va = []
        genes_ja = []
        genes_vb = []
        genes_jb = []

        for i_cell in range(tcr.shape[0]):
            genes_va = genes_va + [tcr_genes[i_cell][0][0]]
            genes_ja = genes_ja + [tcr_genes[i_cell][0][1]]
            genes_vb = genes_vb + [tcr_genes[i_cell][1][0]]
            genes_jb = genes_jb + [tcr_genes[i_cell][1][1]]
        vdj_dic = {'VA': genes_va, 'JA': genes_ja, 'VB': genes_vb, 'JB': genes_jb}
        
        help_message = make_feature_panel_plots(
            adata, all_nbrs, tcr_results, pngfile, feature_type, tcr_genes, vdj_dic,
            title=figure_tag)
        adata.uns['conga_results'][figure_tag] = pngfile
        adata.uns['conga_results'][figure_tag+HELP_SUFFIX] = help_message
        make_figure_helpfile(figure_tag, adata)

    else:
        print('conga.plotting.make_graph_vs_features_plots:: missing results',
              'dont forget to call conga.correlations.run_graph_vs_features')




