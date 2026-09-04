
# As always
from os import path
import numpy as np
import pandas as pd

# Lots of plotting here, so:
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.cm import coolwarm, ScalarMappable
import matplotlib.colors as mcolors

# gene set enrichment and helpers
from .aREA import aREA
from .utils import gene_sets_to_regulon, _prep_ges
from . import plotting as pl
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

"""
This module implements gene set enrichment functionality for one-tailed gene sets. 
"""

PYREA_RC_PARAMS = {'axes.linewidth': 0.5, 'axes.titlelocation': 'left',
                   'axes.titlepad': 4.0, 'axes.labelpad': 2.0,
                   'axes.xmargin': .02, 'axes.ymargin': .02,
                   'xtick.major.size': 0, 'xtick.minor.size': 0,
                    'xtick.major.pad': 2.0, 'xtick.minor.pad': 2.0 ,
                    'ytick.major.size': 0, 'ytick.minor.size': 0,
                    'ytick.major.pad': 2.0, 'ytick.minor.pad': 2.0 ,
                    'legend.framealpha': 1.0, 'legend.edgecolor': '0.5',
                    'legend.handlelength': 1.0, 
                    'legend.handletextpad': 0.4,
                    'legend.columnspacing': 1.0}


class Gsea1T:
    
    """Base class for one-tailed gene set enrichment analysis"""

    def __init__(self, 
                 ges: pd.Series, 
                 gene_set: dict, 
                 weight: float = 1):

        self.weight = weight
        self.gene_set_name, self.gene_set_org = gene_set.popitem()

        if not isinstance(ges, pd.Series):
            raise TypeError('Need an indexed pandas Series, please.')
        else:
            self.ges = _prep_ges(ges, asc_sort=False) # Gsea1T needs to be sorted from high to low for running sum
            self.ns = len(self.ges)
            self.along_scores = [*range(self.ns)]

        
        if not np.in1d(self.gene_set_org, ges.index).any():
            raise ValueError('None of the genes in gene set found in GES index')
        else:
            self.gene_set_final = [g for g in self.gene_set_org if g in self.ges.index]

        self.gs_idx = self._find_hits(self.ges, self.gene_set_final)

        self.rs = self._derive_rs(self.ges, self.gs_idx, self.weight)
        self.es_idx = np.abs(self.rs).argmax()
        self.left_end_closer = self.es_idx <= (self.ns-self.es_idx)

        self.gs_reg = gene_sets_to_regulon({self.gene_set_name:self.gene_set_final}, minsize=len(self.gene_set_final))
        self.aREA_nes = aREA(self.ges,
                            self.gs_reg, 
                            minsize=len(self.gene_set_final)).iloc[0][0]

        self.pval = norm.sf(np.abs(self.aREA_nes))*2
        self.ledge, self.ledge_xinfo = self._get_ledge(self.ges, self.gs_idx, self.es_idx, self.left_end_closer)
        if self.left_end_closer:
            self.ledge_yinfo = 0, self.rs[self.es_idx]
        else:
            self.ledge_yinfo = self.rs[self.es_idx], 0

    def __repr__(self):

        """String representation for the Gsea1T class"""

        return (f"Gsea1T(GES length: {self.ns}, " 
                f"Gene set:{len(self.gene_set_org)}, "
                f"Name: {self.gene_set_name}, "
                f"Overlap: {len(self.gene_set_final)})"
                )

    def _find_hits(self, 
                   ges: pd.Series,
                   gene_set: list)-> list:

        """Finds the positions of a gene set in a given gene expression signature list

        Parameters
        ----------
        ges :
            pd.Series: The gene expression signature (indexed numeric pandas Series).
        gene_set :
            list: The gene set in question.
        ges: pd.Series :
            
        gene_set: list :
            

        Returns
        -------
        list
            A list of positions for each gene set member in the gene expression signature.

        """

        return [idx for idx, gene in enumerate(ges.index.values) if gene in gene_set]

    def _derive_rs(self, 
                   ges: pd.Series, 
                   gene_indices: list,
                   weight: float=1)->np.ndarray:
        
        """Derives the running sum for plotting

        Parameters
        ----------
        ges :
            pd.Series: The gene expression signature (indexed numeric pandas Series).
        gene_indices :
            list: The positions of the gene set members in the signature.
        weight :
            float: Weighting exponent, see Subramanian PNAS 2005, for details.
        ges: pd.Series :
            
        gene_indices: list :
            
        weight: float :
            

        Returns
        -------
        np.array
            The running sum array.

        """

        Nr = np.sum(np.abs(ges[gene_indices])**weight) # normalization factor
        Nh = len(ges) - len(gene_indices) # non-hits
        tmp = np.repeat(-1/Nh, len(ges))
        tmp[gene_indices] = np.abs(ges[gene_indices])**weight/Nr
        rs = np.cumsum(tmp) # running sum
        
        return rs

    def _get_ledge(self, 
                   ges: pd.Series,
                   gene_indices: list,
                   es_index: int,
                   left_end_closer: bool)-> tuple:
        """

        Parameters
        ----------
        ges :
            pd.Series: The gene expression signature (indexed numeric pandas Series).
        gene_indices :
            list:
        es_index :
            int:
        ges: pd.Series :
            
        gene_indices: list :
            
        es_index: int :
            

        Returns
        -------
        tuple
            A DataFrame of leading edge genes and their positions in the signature as well as the
        tuple
            A DataFrame of leading edge genes and their positions in the signature as well as the
            x-axis limits of the leading edge area.

        """
        # TODO: This is still a bit of a hack regarding the positions for the leading edge plot
        
         # if True, we're closer to the left hand side
        
        if left_end_closer:
            # Leading edge indices
            ledge_idx = [idx for idx in gene_indices if idx <= es_index]
            ledge_xinfo = np.min(ledge_idx), es_index
        else:
            ledge_idx = [idx for idx in gene_indices if idx >= es_index]
            ledge_xinfo = es_index, np.max(ledge_idx)
            
            
        ledge_genes = ges.index[ledge_idx].values
        ledge_values = ges[ledge_genes].values
        
        df = pd.DataFrame(zip(ledge_genes, ledge_idx, ledge_values), 
                          columns=['gene', 'index','gene_stat'])
        
        return df, ledge_xinfo

    def _filter_ledge(self, 
                      ledge:pd.DataFrame,
                    left_end_closer: bool, 
                    n_genes:int = None,
                    *, #keyword only from here
                    subset: dict=None):
        
        ledge_sub = ledge.copy()
        
        # Set a default if none is provided
        if not n_genes:
            n_genes = 25
        
        # cut down to available genes if there are fewer than intended
        if len(ledge_sub) < n_genes:
            n_genes = len(ledge_sub)

        if subset:

            if not isinstance(subset, dict):
                raise TypeError('Subset instructions need to be supplied as dictionary')

            if not any([k in subset for k in ['gene_stat', 'genes']]):
                raise ValueError("Valid items are: {gene_stat:float} OR {genes:[str]}")
                
            if stat_filter := subset.get('gene_stat', 0):
                ledge_sub = ledge_sub[ledge_sub['gene_stat'].abs() >= stat_filter]
                
            elif gene_filter := subset.get('genes', 0):
                ledge_sub = ledge_sub[ledge_sub['gene'].isin(gene_filter)]
                
            if not len(ledge_sub)>0:
                raise AssertionError('Filter result has length 0!')
            
            elif len(ledge_sub)>n_genes:
                print(f'Found more than {n_genes} genes, subsetting to best {n_genes}!')
                if left_end_closer:
                    ledge_sub = ledge_sub.nsmallest(n_genes, 'index')
                else:
                    ledge_sub = ledge_sub.nlargest(n_genes, 'index')

        else: 
            print(f'No filter provided: taking the top {n_genes} leading edge genes !')
            if left_end_closer:             
                ledge_sub = ledge_sub.nsmallest(n_genes, 'index')
            else:
                ledge_sub = ledge_sub.nlargest(n_genes, 'index')
                    
        return ledge_sub
    
    
    def _ledge_grid_prep(self, left_end_closer: bool, fig=None):
    
        if fig is None:
            fig = plt.gcf()
            
        if left_end_closer:
            height_ratios = [0.8, 0.2, 2]
            gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=height_ratios, hspace=0.05)
            grid_dict = {'running_sum':gs[2,0], 'events':gs[1,0], 'labels':gs[0,0]}
        else:
            height_ratios = [2, 0.2, 0.8]
            gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=height_ratios, hspace=0.05)
            grid_dict = {'running_sum':gs[0,0], 'events':gs[1,0], 'labels':gs[2,0]}
            
        return grid_dict
      
    def plot(self,
             conditions: tuple = ('High', 'Low'),
             ges_symlog: bool = True,
             ges_stat_fmt: str = '1.0f',
             figsize: tuple = None,
             ges_type: str = None,
             ges_kw: dict = None,
             evt_kw: dict = None,
             rs_kw: dict = None,
             leg_kw: dict = None
             )->mpl.figure.Figure:

        """This will return a figure object containing 3 axes displaying the gene expression signature,
        the gene set indices and the running sum line

        Parameters
        ----------
        figsize :
            tuple: Tuple of floats to specify the Figure size in inches (Default value = (2.5, 2.5):
        bar_alpha :
            float: A float between 0 and 1 to specify the alpha for the bars in the eventplot (Default value = 0.7).
        phenotypes :
            tuple:  (Default value = ('A', 'B'): A tuple of strings to specify the two phenotypes being compared.
        colors :
            tuple:  (Default value = ('.75', '#439D75'): A tuple of strings for a) the gene expression signature in the first axes and b) the
        colors :
            tuple:  (Default value = ('.75', '#439D75'): A tuple of strings for a) the gene expression signature in the first axes and b) the
            gene set and running sum in the other two axes.
        ges_type :
            str:  A string to specify the type of gene-level statistic (Default value = None)
        figsize: tuple :
             (Default value = (3)
        3) :
            
        conditions: tuple :
             (Default value = ('A')
        'B') :
            
        ges_symlog: bool :
             (Default value = True)
        ges_stat_fmt: str :
             (Default value = '1.0f')
        ges_type: str :
             (Default value = None)
        ges_kw: dict :
             (Default value = None)
        evt_kw: dict :
             (Default value = None)
        rs_kw: dict :
             (Default value = None)

        Returns
        -------
        
            A Figure Object.

        """
        
        # Some defaults
        ges_prop = {'color':'.5', 'alpha':0.25, 'linewidth':0.1}
        evt_prop = {'color': '#747C92', 'alpha':0.7, 'linewidths':0.5} # Slate gray
        
        if figsize:
            width, height = figsize
        else:
            height = 2.2
            width = 2.5
        
        height_evt = 0.2
        height_ges = 0.5
        height_rest = height - (height_evt + height_ges)
        
        
        with plt.rc_context(PYREA_RC_PARAMS):
            
            fig = plt.figure(figsize=(width, height), 
                             tight_layout=True)
            
            gs = fig.add_gridspec(nrows=3, ncols=1, 
                                height_ratios=[height_ges, height_evt, height_rest], 
                                hspace=0)

            # first graph: gene expression signature
            ax_ges = fig.add_subplot(gs[0,0])
            if ges_kw:
                ges_prop.update(ges_kw)
            pl._plot_ges(self.along_scores, 
                        self.ges.values, 
                        conditions=conditions,
                        is_high_to_low=True,
                        ges_type=ges_type, 
                        symlog=ges_symlog,
                        stat_fmt=ges_stat_fmt,
                        ax=ax_ges, 
                        **ges_prop)
            
            # second graph: bars to indicate positions of individual genes
            ax_evt= fig.add_subplot(gs[1,0])
            if evt_kw:
                evt_prop.update(evt_kw)
            ax_evt.eventplot(self.gs_idx, **evt_prop)
            ax_evt.axis('off')

            # Third graph: Running sum
            rs_prop = {"color": evt_prop.get('color')} # cannot be missing, see above
            leg_prop = {'title':self.gene_set_name, "loc":0}
        
            ax_rs = fig.add_subplot(gs[2,0])
                    
            if rs_kw is not None:
                rs_prop.update(rs_kw)
            ax_rs.tick_params(labelsize='x-small')
            pl._plot_run_sum(self.rs, self.es_idx, ax=ax_rs, **rs_prop)
            if leg_kw:
                leg_prop.update(leg_kw)
            leg = pl._stats_legend(self.aREA_nes, self.pval, ax=ax_rs, leg_kw=leg_prop)
            ax_rs.add_artist(leg)
            pl._format_xaxis_ges(self.ns, ax=ax_rs)
            
        return fig

    def plot_ledge(self,
                   figsize: tuple = None,
                   n_genes: int = None,
                   subset: dict = None,
                   highlight: list = None,
                   rs_kw: dict = None,
                   leg_kw: dict = None,
                   evt_kw: dict = None,
                   text_kw: dict = None,
                   rect_kw: dict = None,
                   conn_patch_kw: dict = None):
        """

        Parameters
        ----------
        figsize: tuple :
             (Default value = (3.5)
        3) :
            
        highlight: tuple :
             (Default value = None)
        rs_kw: dict :
             (Default value = None)
        lbl_kw: dict :
             (Default value = None)
        patch_kw:dict :
             (Default value = None)

        Returns
        -------

        """        
        # Default values
        rs_prop = {'color':'#747C92'} #Slate gray
        leg_prop = {'title':self.gene_set_name}
        evt_prop = {'color': rs_prop.get('color'), 'alpha':0.7, 'linewidths':0.5, 'lineoffsets':0.5}
        text_prop = {'fontsize':'xx-small', 'rotation':90, 'ha':'center', 'va':'bottom'}
        rect_prop = {'color':rs_prop.get('color'), 'alpha':0.25, 'lw':0}
        conn_patch_prop = {'color':'.15', 'lw':0.25}

        # Prepare gene labels 
        df = self.ledge.copy()
        df_sub = self._filter_ledge(df, self.left_end_closer, n_genes, subset=subset)  
        xmin, xmax = self.ledge_xinfo
        
        if rs_kw:
            rs_prop.update(rs_kw)
        if text_kw:
            text_prop.update(text_kw)
        if leg_kw:
            leg_prop.update(leg_kw)
        if rect_kw:
            rect_prop.update(rect_kw)
        if conn_patch_kw:
            conn_patch_prop.update(conn_patch_kw)
        
        if figsize:
            width, height = figsize
        else:
            width = 2.5
            height = 3
                
        with plt.rc_context(PYREA_RC_PARAMS):
            
            fig = plt.figure(figsize=(width, height))
            # set up the grid
            grid_dict = self._ledge_grid_prep(self.left_end_closer, fig=fig)
            
            ax_rs = fig.add_subplot(grid_dict.get('running_sum'))
            pl._plot_run_sum(self.rs, self.es_idx, ax=ax_rs, **rs_prop)
            leg = pl._stats_legend(self.aREA_nes, self.pval, ax=ax_rs, leg_kw=leg_prop)
            ax_rs.add_artist(leg)
            pl._format_xaxis_ges(self.ns, ax=ax_rs)
            if self.left_end_closer:
                ax_rs.spines['top'].set_visible(False)
            else:
                ax_rs.spines['bottom'].set_visible(False)
                ax_rs.xaxis.set_label_position('top')
                ax_rs.xaxis.tick_top()
                
            # events
            ax_evt = fig.add_subplot(grid_dict.get('events'))
            if evt_kw:
                evt_prop.update(evt_kw)
            ax_evt.eventplot(self.ledge['index'].values, **evt_prop)
            ax_evt.set_xticks([])
            ax_evt.set_yticks([])
            ax_evt.set_xlim(xmin, xmax)
            ax_evt.set_ylim(0, 1)
            for axis in ['top','bottom','left','right']:
                ax_evt.spines[axis].set_linewidth(0.25)

            # labels
            ax_lbls = fig.add_subplot(grid_dict.get('labels'), sharex=ax_evt)
            ax_lbls.axis('off')
            ax_lbls.set_ylim(0, 1)

            # get patches
            patch_dict = pl._ledge_patch_prep(self.left_end_closer, 
                                           self.ledge_xinfo, 
                                           self.ledge_yinfo, 
                                           rect_prop, 
                                           conn_patch_prop,
                                           ax_rs=ax_rs, 
                                           ax_evt=ax_evt)
            # Draw connection lines
            fig.add_artist(patch_dict.get('conn_left'))     
            fig.add_artist(patch_dict.get('conn_right'))
            # Add shaded rectangles    
            ax_evt.add_artist(patch_dict.get('evt_rect'))
            ax_rs.add_artist(patch_dict.get('rs_rect'))
            # Add text
            pl._plot_ledge_labels(df_sub, 
                                self.left_end_closer, 
                                self.ledge_xinfo, 
                                text_prop,
                                highlight=highlight, 
                                line_kw=conn_patch_prop,
                                ax=ax_lbls)
        
        return fig    

            
class Gsea1TMultSigs:
    
    """To implement one-tailed gene set enrichment analysis on multiple different signatures and visualize them"""
     
    def __init__(self, 
                 dset:pd.DataFrame, 
                 gene_set:list,  
                 ordered: bool=True):
                
        if not isinstance(dset, pd.DataFrame):
            raise TypeError('Need an indexed pandas DataFrame, please.')
            
        else: 
            self.dset = _prep_ges(dset)
            self.ns = len(self.dset)
            self.samples = self.dset.columns.values
        
        if not np.in1d(gene_set, dset.index).any():
            raise ValueError('None of the genes in gene set found in dset DataFrame index')
        else:
            self.gs_org = gene_set
            self.gs_final = [g for g in gene_set if g in self.dset.index]
        
        self.gs_idx = self._find_hits(self.dset, self.gs_final)
        
        self.gs_reg = gene_sets_to_regulon({'GS':self.gs_org}, minsize=len(self.gs_final))
        
        self.stats = self._get_stats(dset=self.dset, 
                                     regulon=self.gs_reg, 
                                     minsize=len(self.gs_final),
                                     samples=self.samples)
        
        if ordered:
            idx = self.stats['NES'].argsort().values
            self.stats = self.stats.take(idx, axis=0)
            self.stats.reset_index(inplace=True, drop=True)
            self.gs_idx = self.gs_idx.take(idx, axis=0)
            
    
    def __repr__(self):
        
        """String representation for the Gsea1TMultSig class"""

        return (
                f"Gsea1TMult(Number of genes: {self.ns}\n"
                f"Number of signatures: {len(self.samples)}\n"
                f"Gene set: n={len(self.gs_org)}\n"
                f"Overlap: n={len(self.gs_final)}\n"
                )    
       
    def _find_hits(self, dset, *gene_sets):
        
        """Finds the positions of a gene set in a given gene expression signaure matrix

        Parameters
        ----------
        dset :
            
        *gene_sets :
            

        Returns
        -------

        """
        
        # rank order all signatures 
        hits = dset.apply(lambda x: x.sort_values().argsort(), axis=0)
        
        # subset to those in the gene set
        hitlist = [hits.loc[gene_set].values.T for gene_set in gene_sets]
        
        col_names = [f'position_set_{i}' for i in range(len(gene_sets))]
        
        hitdf = pd.DataFrame(zip(*hitlist), index=self.samples, columns=col_names)
        # Return the indices for the gene set in each signature as a 
        # numpy array of arrays, transverse for plotting
        return hitdf
    
    def _get_stats(self, 
                   dset: pd.DataFrame, 
                   regulon: pd.DataFrame, 
                   minsize: int,
                   samples:np.ndarray):
        
        """Computes normalized enrichment scores and some tools for later visualization

        Parameters
        ----------
        dset: pd.DataFrame :
            
        regulon: pd.DataFrame :
            
        minsize: int :
            
        samples:np.ndarray :
            

        Returns
        -------

        """
        
        nes = aREA(dset, regulon, minsize).iloc[0,:].values # get a flattened one-dimensional array
        pvals = norm.sf(np.abs(nes))*2 # retrieve two-sided p-value from normal distribution
        fdrs = multipletests(pvals, method = 'fdr_bh')[1] #FDR
        
        stats = pd.DataFrame(data=zip(samples, nes, pvals, fdrs),
                            columns=['signature_name', 'NES', 'pval', 'FDR'])

        return stats
        
        
    def plot(self, 
             figsize: tuple = None,
             add_title: str = None,
             evt_kw: dict = None,
             norm_kw: dict = None,
             pcm_kw: dict= None):
         
        """This will return a figure object containing 4 axes displaying the gene expression signature,
        the gene set indices and the running sum line

        Parameters
        ----------
        bar_alpha :
            float:  (Default value = 0.5)
        figsize :
            tuple:  (Default value = (3.5)
        3) :
            
        norm_kws :
            dict:  (Default value = None)
        figsize: tuple :
             (Default value = (3)
        add_title: str :
             (Default value = None)
        evt_kw: dict :
             (Default value = None)
        norm_kw: dict :
             (Default value = None)
        pcm_kw: dict :
             (Default value = None)

        Returns
        -------

        """
        
        df = self.stats.copy()
        
        # needs to be unnested
        evt_data = [arr for tup in self.gs_idx.values for arr in tup]
        
        evt_prop = {'linelengths':3/4, 'color':'.15', 'alpha':0.7, 'linewidths':0.5}
             
        norm, pcm_prop = pl._prepare_nes_colors(df, norm_kw, pcm_kw)
        
        # Prettify FDR format
        df['FDR'] = df['FDR'].apply(pl._fdr_formatter)
                
        # In this first bit, we determine the NES color scheme
        
        if figsize:
            width, height = figsize
        else:
            height = len(df) * 0.3
            width = 3
        
        height_cbar = 0.15 
        height_rest = height - height_cbar
        width_nes = 0.25
        width_rest = width - width_nes
        
        with plt.rc_context(PYREA_RC_PARAMS):
            
            fig = plt.figure(figsize=(width, height), dpi=150) 
                    
            gs = fig.add_gridspec(nrows=2, 
                        ncols=2, 
                        hspace=0.025, 
                        wspace=0.025,
                        width_ratios=[width_rest, width_nes],
                        height_ratios=[height_cbar, height_rest])
            
            # Plot 1: Illustrate the ordered signature as a colorbar
            ax_cbar = fig.add_subplot(gs[0,0])
            cb = fig.colorbar(ScalarMappable(cmap=plt.cm.RdBu_r), 
                orientation='horizontal', ticks=[], cax=ax_cbar)
            #cb.outline.set_visible(False)
            ax_cbar.text(0, 0.5, 'Low', color='w', ha='left', va='center', fontsize='x-small')
            ax_cbar.text(1, 0.5, 'High', color='w', ha='right', va='center', fontsize='x-small')
            if add_title:
                ax_cbar.set_title(add_title, fontsize='x-small')
            
            # Plot 2: Illustrate targets
            ax_evt = fig.add_subplot(gs[1,0])
            
            if evt_kw:
                evt_prop.update(evt_kw)
                
            ax_evt.eventplot(evt_data, **evt_prop)
            ax_evt.set_yticks(range(len(df)))
            ax_evt.set_ylim([-0.5, len(df)-0.5])
            ax_evt.set_yticklabels(df['signature_name'].values, fontsize='x-small')
            ax_evt.set_xlim([-.5, self.ns+.5])
            # x-axis formating
            pl._format_xaxis_ges(self.ns, ax=ax_evt)    
            
            # remove spines
            for spine in ['right', 'left', 'top']:
                ax_evt.spines[spine].set_visible(False)

            # Plot 3: NES heatmap
            ax_pcm = fig.add_subplot(gs[1,1])
        
            ax_pcm.pcolormesh(df['NES'].values[:,np.newaxis], 
                norm=norm, **pcm_prop)

            for row in df.itertuples():
                ax_pcm.text(0.5, row.Index + 0.5, f'{row.NES:1.1f}',
                    fontsize='xx-small', c=row.color,
                    ha='center', va='center')
                ax_pcm.text(1, row.Index + 0.5, row.FDR,
                    fontsize='xx-small', c='k',
                    ha='left', va='center')
                ax_pcm.axis('off')

            # Plot 4: Not sure if necessary to open new plot, but use grid fully. Just titles for annotation
            ax_titles = fig.add_subplot(gs[0,1])
            ax_titles.text(0.5, 0, 'NES', fontsize='x-small', c='k', ha='center', va='bottom')
            ax_titles.text(1.1, 0, 'FDR', fontsize='x-small', c='k', ha='left', va='bottom')
            ax_titles.axis('off')
            
        return fig



class Gsea1TMultSets:
    
    """To implement one-tailed gene set enrichment of multiple gene sets on one gene expression signature
    and visualize the results

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(self, 
                 ges: pd.Series, 
                 gene_sets: dict, 
                 ordered: bool = True,
                 minsize: int = 20):

        self.minsize = minsize

        if not isinstance(ges, pd.Series):
            raise TypeError('Need an indexed pandas Series, please.')
        if not isinstance(gene_sets, dict):
            raise TypeError('Need a dictionary of gene sets, please')
        else:
            self.ges = _prep_ges(ges)
            self.ns = len(self.ges)
            self.along_scores = [*range(self.ns)]

        self.n_org_gene_sets = len(gene_sets)
        self.gene_sets = self._prep_gene_sets(gene_sets, self.ges.index, self.minsize)
        
        if len(self.gene_sets) == 0:
            raise ValueError('None of the gene sets have enough genes to be found in GES index')
       
        self.gs_names = list(self.gene_sets.keys())
        
        self.gs_reg = gene_sets_to_regulon(self.gene_sets, minsize=self.minsize)
              
        self.gs_idx = self._find_hits(self.ges, self.gene_sets)
        
        self.stats = self._get_stats(self.ges, self.gs_reg, self.minsize, self.gs_idx)
                
        if ordered:
            self.stats.sort_values('NES', inplace=True)
            self.stats.reset_index(inplace=True, drop=True)
            

    def __repr__(self):

        """String representation for the Gsea1TMultSet class"""

        return (
                f"Gsea1TMultSet(GES length: {self.ns}\n"
                f"Gene sets in: {self.n_org_gene_sets}\n"
                f"Gene sets used: {len(self.gene_sets)}"
                )
        
    def _prep_gene_sets(self, 
                        gene_sets: dict,
                        genes: pd.Index,
                        minsize: int) -> dict:
        
        """Go through the provided dict of gene sets and check whether at least minsize
        genes of said gene set occur in the gene expression signature

        Parameters
        ----------
        gene_sets :
            dict:
        genes :
            pd.Index:
        minsize :
            int:
        gene_sets: dict :
            
        genes: pd.Index :
            
        minsize: int :
            

        Returns
        -------

        """
        return {key:val for key, val in gene_sets.items() if np.in1d(val, genes).sum() >= minsize}

    def _find_hits(self, 
                   ges: pd.Series, 
                   gene_sets: dict):
        """

        Parameters
        ----------
        ges :
            pd.Series:
        gene_sets :
            dict:
        ges: pd.Series :
            
        gene_sets: dict :
            

        Returns
        -------

        """

        # This will have to be sorted first, which is the case in this Class
        ges_idx = ges.copy().argsort()
        
        ges_pos = [ges_idx[ges_idx.index.intersection(val)].values for val in gene_sets.values()]
        
        return pd.DataFrame(zip(gene_sets.keys(), ges_pos), columns=['gene_set', 'positions'])
            
        
    def _get_stats(self, 
                   ges: pd.Series,
                   regulon: pd.DataFrame,
                   minsize: int,
                   add_positions: pd.DataFrame=None):
        
        """Computes normalized enrichment scores and some tools for later visualization

        Parameters
        ----------
        ges :
            pd.Series:
        regulon :
            dict:
        ges: pd.Series :
            
        regulon: pd.DataFrame :
            
        minsize: int :
            
        add_positions: pd.DataFrame :
             (Default value = None)

        Returns
        -------

        """
        
        nes = aREA(ges, regulon, minsize).iloc[:,0]
        pvals = norm.sf(np.abs(nes.values))*2 # retrieve two-sided p-value from normal distribution
        fdrs = multipletests(pvals, method = 'fdr_bh')[1] #FDR

        df = pd.DataFrame(data=zip(nes.index.values, nes.values, pvals, fdrs),
                            columns=['gene_set', 'NES', 'pval', 'FDR']) 
        
        if add_positions is not None:
            df = df.merge(add_positions, on='gene_set')
            
        return df
                    
                    
    def _filter_multset_stats(self, 
                              stats:pd.DataFrame, 
                              subset: dict): 
        
        """Filter stat results if needed for plotting

        Parameters
        ----------
        stats:pd.DataFrame :
            
        subset: dict :
            

        Returns
        -------

        """
        if not isinstance(subset, dict):
            raise TypeError('Subset instructions need to be supplied as dictionary')
        
        if not any([k in subset for k in ['FDR', 'gene_sets', 'NES']]):
            raise ValueError('Can only subset based on FDR, NES or gene set names')
            
        if 'gene_sets' in subset:
            stats = stats[stats['gene_set'].isin(subset.get('gene_sets'))]
  
        elif 'NES' in subset:
            stats = stats[stats['NES'].abs()>=subset.get('NES', 2)]
        
        else: # In the end, we will always filter based on FDR (even if you don't spell right)
            stats = stats[stats['FDR']<=subset.get('FDR', 0.1)]
        
        if not len(stats)>0:
            raise AssertionError('None of the gene sets fullfilled the required cutoff')
        
        stats.reset_index(inplace=True, drop=True)
        
        return stats

    def plot(self,
             conditions: tuple = ('Low', 'High'),
             strip_gs_names: bool = True,
             ges_symlog: bool= True,
             ges_stat_fmt: str = '1.0f',
             figsize:tuple=None,
             subset: dict = None,
             ges_type:str = None,
             ges_kw: dict = None,
             evt_kw: dict = None,
             pcm_kw: dict = None,
             norm_kw: dict = None
             ):
         
        """This will return a figure object containing 4 axes displaying the gene expression signature,
        the gene set indices and the Normalized Enrichment scores and associated FDR

        Parameters
        ----------
        phenotypes :
            tuple:  (Default value = ('A')
        'B') :
            
        bar_alpha :
            float:  (Default value = 0.5)
        figsize :
            tuple:  (Default value = (3.5)
        3) :
            
        norm_kws :
            dict:  (Default value = None)
        ges_type :
            str:  (Default value = None)
        figsize: tuple :
             (Default value = (3)
        conditions: tuple :
             (Default value = ('A')
        strip_gs_names: bool :
             (Default value = True)
        ges_symlog: bool :
             (Default value = True)
        ges_stat_fmt: str :
             (Default value = '1.0f')
        subset: dict :
             (Default value = None)
        ges_type:str :
             (Default value = None)
        ges_kw: dict :
             (Default value = None)
        evt_kw: dict :
             (Default value = None)
        pcm_kw: dict :
             (Default value = None)
        norm_kw: dict :
             (Default value = None)

        Returns
        -------

        """
        
          # Key input data: 
        df = self.stats.copy() 
        
        # Some setup: 
        ges_prop = {'color':'.5', 'alpha':0.25, 'linewidth':0.1}
        evt_prop = {'color': '.15', 'alpha':0.7, 'linelengths':3/4, 'linewidths':0.5}         
        
        norm, pcm_prop = pl._prepare_nes_colors(df, norm_kw, pcm_kw)        
                    
        if subset:
            df = self._filter_multset_stats(df, subset)
            
        if strip_gs_names:
            df['gene_set'] = df['gene_set'].str.split('_', n=1).str[-1]         
                
        df['FDR'] = df['FDR'].apply(pl._fdr_formatter)
        
        # Setup figure and gridspec
        if figsize:
            width, height = figsize
        else:
            height = len(df) * 0.15
            width = 3
        
        height_ges = 0.5
        height_rest = height - height_ges
        width_nes = 0.3
        width_rest = width - width_nes
        
        with plt.rc_context(PYREA_RC_PARAMS):
            
            fig = plt.figure(figsize=(width, height)) 
                       
            gs = fig.add_gridspec(nrows=2, 
                        ncols=2, 
                        hspace=0.025, 
                        wspace=0.025,
                        width_ratios=[width_rest, width_nes],
                        height_ratios=[height_ges, height_rest])
            
            # Plot 1: Illustrate the ordered signature
            ax_ges = fig.add_subplot(gs[0,0])
            if ges_kw:
                ges_prop.update(ges_kw)
            pl._plot_ges(self.along_scores, self.ges.values, ges_type=ges_type, conditions=conditions, 
                        is_high_to_low=False, symlog=ges_symlog, stat_fmt=ges_stat_fmt, ax=ax_ges, **ges_prop)
            
            # Plot 2: Illustrate targets
            ax_evt = fig.add_subplot(gs[1,0])
            if evt_kw:
                evt_prop.update(evt_kw)
            ax_evt.eventplot(df['positions'].values, **evt_prop)
            # y-axis formatting
            ax_evt.set_yticks(range(len(df)))
            ax_evt.set_ylim([-0.5, len(df)-0.5])
            ax_evt.set_yticklabels(df['gene_set'], fontsize='xx-small')
            # x-axis formatting        
            ax_evt.set_xlim([-.5, self.ns+.5])
            pl._format_xaxis_ges(self.ns, ax=ax_evt)
            
            # remove spines
            for spine in ['right', 'left', 'top']:
                ax_evt.spines[spine].set_visible(False)

            # Plot 3: NES heatmap            
            ax_pcm = fig.add_subplot(gs[1,1])

            ax_pcm.pcolormesh(df['NES'].values[:,np.newaxis], 
                        norm=norm, **pcm_prop)
        
            for row in df.itertuples():
                ax_pcm.text(0.5, row.Index + 0.5, f"{row.NES:1.1f}",
                    fontsize='xx-small', c=row.color,
                    ha='center', va='center')
                ax_pcm.text(1, row.Index + 0.5, row.FDR,
                    fontsize='xx-small', c='k',
                    ha='left', va='center')
                ax_pcm.axis('off')

            # Plot 4: Not sure if necessary to open new plot, but use grid fully. Just titles for annotation
            ax_titles = fig.add_subplot(gs[0,1])
            ax_titles.text(0.5, 0, 'NES', fontsize='x-small', c='k', ha='center', va='bottom')
            ax_titles.text(1.1, 0, 'FDR', fontsize='x-small', c='k', ha='left', va='bottom')
            ax_titles.axis('off')
        
        return fig

