"""
This module implements gene set enrichment functionality for two-tailed gene sets. 
"""

# As always
import numpy as np
import pandas as pd

# Lots of plotting here, so:
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable


# gene set enrichment and helpers
from .aREA import aREA
from .utils import gene_sets_to_regulon, _prep_ges
from . import plotting as pl
from .Gsea1T import Gsea1T, Gsea1TMultSigs
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests



PYREA_RC_PARAMS = {'axes.linewidth': 0.5, 'axes.titlelocation': 'left',
                   'axes.titlepad': 4.0, 'axes.labelpad': 2.0,
                   'axes.xmargin': .02, 'axes.ymargin': .02,
                   'xtick.major.size': 0, 'xtick.minor.size': 0,
                    'xtick.major.pad': 2.0, 'xtick.minor.pad': 2.0 ,
                    'ytick.major.size': 0, 'ytick.minor.size': 0,
                    'ytick.major.pad': 2.0, 'ytick.minor.pad': 2.0 ,
                    'legend.framealpha': 1.0, 'legend.handlelength': 1.0, 
                    'legend.handletextpad': 0.4, 
                    'legend.columnspacing': 1.0}



class Gsea2T(Gsea1T):
    """ """

    def __init__(self, 
                 ges: pd.Series, 
                 gene_set_1: dict, 
                 gene_set_2: dict,
                 weight: float = 1):
               
        self.weight = weight
        self.gene_set_name_1, self.gene_set_org_1 = gene_set_1.popitem()
        self.gene_set_name_2, self.gene_set_org_2 = gene_set_2.popitem()

        if not isinstance(ges, pd.Series):
            raise TypeError('Need an indexed pandas Series, please.')
        else:
            self.ges = _prep_ges(ges, asc_sort=False) # Gsea1T needs to be sorted from high to low for running sum
            self.ns = len(self.ges)
            self.along_scores = [*range(self.ns)]

        if not np.in1d(self.gene_set_org_1, ges.index).any():
            raise ValueError('None of the genes in gene set 1 found in GES index')
        elif not np.in1d(self.gene_set_org_2, ges.index).any():
            raise ValueError('None of the genes in gene set 2 found in GES index')
        else:
            self.gene_set_final_1 = [g for g in self.gene_set_org_1 if g in self.ges.index]
            self.gene_set_final_2 = [g for g in self.gene_set_org_2 if g in self.ges.index]
        
        self.gs_idx_1 = self._find_hits(self.ges, self.gene_set_final_1)
        self.gs_idx_2 = self._find_hits(self.ges, self.gene_set_final_2)

        self.rs_1 = self._derive_rs(self.ges, self.gs_idx_1, self.weight)
        self.es_idx_1 = np.abs(self.rs_1).argmax()
        self.left_end_closer_1 = self.es_idx_1 <= (self.ns-self.es_idx_1)
        
        self.rs_2 = self._derive_rs(self.ges, self.gs_idx_2, self.weight)
        self.es_idx_2 = np.abs(self.rs_2).argmax()
        self.left_end_closer_2 = self.es_idx_2 <= (self.ns-self.es_idx_2)
        

        self.gs_reg_1 = gene_sets_to_regulon({self.gene_set_name_1:self.gene_set_final_1}, minsize=len(self.gene_set_final_1))
        self.aREA_nes_1 = aREA(self.ges,
                            self.gs_reg_1).iloc[0][0]
        
        self.gs_reg_2 = gene_sets_to_regulon({self.gene_set_name_2:self.gene_set_final_2}, minsize=len(self.gene_set_final_2))
        self.aREA_nes_2 = aREA(self.ges,
                            self.gs_reg_2).iloc[0][0]

        self.pval_1 = norm.sf(np.abs(self.aREA_nes_1))*2
        self.ledge_1, self.ledge_xinfo_1 = self._get_ledge(self.ges, 
                                                           self.gs_idx_1, 
                                                           self.es_idx_1,
                                                           self.left_end_closer_1)

        
        self.pval_2 = norm.sf(np.abs(self.aREA_nes_2))*2
        self.ledge_2, self.ledge_xinfo_2 = self._get_ledge(self.ges, 
                                                           self.gs_idx_2, 
                                                           self.es_idx_2,
                                                           self.left_end_closer_2)

        if self.left_end_closer_1:
            self.ledge_yinfo_1 = 0, self.rs_1[self.es_idx_1]
        else:
            self.ledge_yinfo_1 = self.rs_1[self.es_idx_1], 0

        if self.left_end_closer_2:
            self.ledge_yinfo_2 = 0, self.rs_2[self.es_idx_2]
        else:
            self.ledge_yinfo_2 = self.rs_2[self.es_idx_2], 0
        
        
    def __repr__(self):
        
        """String representation for the Gsea2T class"""

        return (
            f"Gsea2T(GES length: {self.ns}\n"
            f"Gene set 1: {len(self.gene_set_org_1)}, Overlap 1: {len(self.gene_set_final_1)}\n"
            f"Gene set 2: {len(self.gene_set_org_2)}, Overlap 2: {len(self.gene_set_final_2)})\n"
        )
    
    def plot(self,
             conditions: tuple = ('High', 'Low'), #always sorted from high to low for Gsea2T
             figsize:tuple=None,
             ges_symlog: bool=True,
             ges_type: str = None,
             ges_stat_fmt:str='1.0f',
             ges_kw: dict = None,
             evt_kw_1: dict = None,
             rs_kw_1: dict = None,
             leg_kw_1:dict=None,
             evt_kw_2: dict = None,
             rs_kw_2: dict = None,
             leg_kw_2:dict=None
             ):
        """

        Parameters
        ----------
        figsize: tuple :
             (Default value = (3)
        3) :
            
        conditions: tuple :
             (Default value = ('A')
        'B') :
            
        ges_type: str :
             (Default value = None)
        ges_symlog: bool :
             (Default value = True)
        ges_stat_fmt:str :
             (Default value = '1.0f')
        ges_kw: dict :
             (Default value = None)
        evt_kw_1: dict :
             (Default value = None)
        rs_kw_1: dict :
             (Default value = None)
        leg_kw_1:dict :
             (Default value = None)
        evt_kw_2: dict :
             (Default value = None)
        rs_kw_2: dict :
             (Default value = None)
        leg_kw_2:dict :
             (Default value = None)

        Returns
        -------

        """
        
        # Defaults for matched appearance
        ges_prop = {'color':'.5', 'alpha':0.25, 'linewidth':0.1}
        if ges_kw:
            ges_prop.update(ges_kw)

        rs_prop_1 = {"color":'#AC3220'} # Chinese Red
        if rs_kw_1:
            rs_prop_1.update(rs_kw_1)
        rs_prop_2 = {"color":'#50808E'} # Teal Blue
        if rs_kw_2:
            rs_prop_2.update(rs_kw_2)
      
        evt_prop_1 = {'color': rs_prop_1.get("color"), 'alpha':0.7, 'linewidths':0.5} 
        if evt_kw_1:
            evt_prop_1.update(evt_kw_1)
        evt_prop_2 = {'color': rs_prop_2.get("color"), 'alpha':0.7, 'linewidths':0.5}
        if evt_kw_2:
                evt_prop_2.update(evt_kw_2) 

        leg_prop_1 = {'title':self.gene_set_name_1, "loc":"upper right", 'labelcolor':rs_prop_1.get('color')}
        if leg_kw_1:
                leg_prop_1.update(leg_kw_1)  
        leg_prop_2 = {'title':self.gene_set_name_2, "loc":"lower left", 'labelcolor':rs_prop_2.get('color')}
        if leg_kw_2:
                leg_prop_2.update(leg_kw_2) 


        if figsize:
            width, height = figsize
        else:
            width = 2.5
            height = 2.4
        
        height_ges = 0.5
        height_evt = 0.2
        height_rest = height - (height_evt*2 + height_ges)

        with plt.rc_context(PYREA_RC_PARAMS):
            
            fig = plt.figure(figsize=(width, height))
            
            gs = fig.add_gridspec(nrows=4, ncols=1, 
                                height_ratios=[height_ges, height_evt, height_rest, height_evt], 
                                hspace=0)

            # first graph
            ax_ges = fig.add_subplot(gs[0,0])
            pl._plot_ges(self.along_scores, 
                        self.ges.values, 
                        ges_type=ges_type,
                        conditions=conditions,
                        is_high_to_low=True,
                        symlog=ges_symlog,
                        stat_fmt=ges_stat_fmt,
                        ax=ax_ges, 
                        **ges_prop)
            
            # second graph: bars to indicate positions of FIRST GENE SET genes
            ax_evt1 = fig.add_subplot(gs[1,0])
         
            ax_evt1.eventplot(self.gs_idx_1, **evt_prop_1)
            ax_evt1.axis('off')

            # Third graph: Running sums and legends
            ax_rs = fig.add_subplot(gs[2,0])
            ax_rs.set_xticks([])  
            pl._plot_run_sum(self.rs_1, self.es_idx_1, **rs_prop_1)
            leg1 = pl._stats_legend(self.aREA_nes_1, self.pval_1, ax=ax_rs, leg_kw=leg_prop_1)
            ax_rs.add_artist(leg1)
            pl._plot_run_sum(self.rs_2, self.es_idx_2, ax=ax_rs, add=True, **rs_prop_2)
            leg2 = pl._stats_legend(self.aREA_nes_2, self.pval_2, ax=ax_rs, leg_kw=leg_prop_2)
            ax_rs.add_artist(leg2)
            ax_rs.spines['bottom'].set_visible(False)
            
            # fourth graph: bars to indicate positions of SECOND GENE SET genes
            ax_evt2 = fig.add_subplot(gs[3,0])
            ax_evt2.eventplot(self.gs_idx_2, **evt_prop_2)
            ax_evt2.set_yticks([])
            pl._format_xaxis_ges(self.ns, ax=ax_evt2)
        
            for spine in ['left', 'right']:
                ax_evt2.spines[spine].set_visible(False)
        
        return fig
    
    def plot_ledge(self,
         figsize: tuple= None,
        n_genes: int = None,
        conn_patch_kw: dict = None,
        subset_1: dict = None,
        highlight_1: list = None,
        rs_kw_1: dict = None,
        leg_kw_1: dict = None,
        evt_kw_1: dict = None,
        text_kw_1: dict = None,
        rect_kw_1: dict = None,
        subset_2: dict = None,
        highlight_2: list = None,
        rs_kw_2: dict = None,
        leg_kw_2: dict = None,
        evt_kw_2: dict = None,
        text_kw_2: dict = None,
        rect_kw_2: dict = None):
        """

        Parameters
        ----------
        figsize: tuple :
             (Default value = (3)
        3) :
            
        lbl_kw_1: dict :
             (Default value = None)
        ledge_trim_1:int :
             (Default value = None)
        rs_kw_1: dict :
             (Default value = None)
        leg_kw_1:dict :
             (Default value = None)
        patch_kw_1:dict :
             (Default value = None)
        highlight_set_1: tuple :
             (Default value = None)
        lbl_kw_2:dict :
             (Default value = None)
        ledge_trim_2:int :
             (Default value = None)
        rs_kw_2:dict :
             (Default value = None)
        leg_kw_2:dict :
             (Default value = None)
        patch_kw_2:dict :
             (Default value = None)
        highlight_set_2:tuple :
             (Default value = None)

        Returns
        -------

        """
        
        # Setup 

        # Common parameters
        conn_patch_prop = {'color':'.15', 'lw':0.25} # this will be the same for both
        text_prop_1 = {'fontsize':'xx-small', 'rotation':90, 'ha':'center', 'va':'bottom'}
        text_prop_2 = {'fontsize':'xx-small', 'rotation':90, 'ha':'center', 'va':'top'}

        rs_prop_1 = {'color':'#AC3220'} #Chinese red
        leg_prop_1 = {'title':self.gene_set_name_1, "loc":"upper right", "labelcolor":rs_prop_1.get("color", "0.15")}
        evt_prop_1 = {'color': rs_prop_1.get('color'), 'alpha':0.7, 'linewidths':0.5, 'lineoffsets':0.5}
        rect_prop_1 = {'color': rs_prop_1.get('color'), 'alpha':0.25, 'lw':0}
     
      
        rs_prop_2 = {'color':'#50808E'} #Teal blue
        leg_prop_2 = {'title':self.gene_set_name_2, "loc":"lower left", "labelcolor":rs_prop_2.get("color", "0.15")}
        evt_prop_2 = {'color': rs_prop_2.get('color'), 'alpha':0.7, 'linewidths':0.5, 'lineoffsets':0.5}
        rect_prop_2 = {'color': rs_prop_2.get('color'), 'alpha':0.25, 'lw':0}
     
      
         # Prepare gene labels 
        df_1 = self.ledge_1.copy()
        df_sub_1 = self._filter_ledge(df_1, self.left_end_closer_1, n_genes, subset=subset_1)  
        xmin_1, xmax_1 = self.ledge_xinfo_1
        
        df_2 = self.ledge_2.copy()
        df_sub_2 = self._filter_ledge(df_2, self.left_end_closer_2, n_genes, subset=subset_2)  
        xmin_2, xmax_2 = self.ledge_xinfo_2

        if conn_patch_kw:
            conn_patch_prop.update(conn_patch_kw)
    
        if rs_kw_1:
            rs_prop_1.update(rs_kw_1)
        if text_kw_1:
            text_prop_1.update(text_kw_1)
        if leg_kw_1:
            leg_prop_1.update(leg_kw_1)
        if rect_kw_1:
            rect_prop_1.update(rect_kw_1)

        if rs_kw_2:
            rs_prop_2.update(rs_kw_2)
        if text_kw_2:
            text_prop_2.update(text_kw_2)
        if leg_kw_2:
            leg_prop_2.update(leg_kw_2)
        if rect_kw_2:
            rect_prop_2.update(rect_kw_2)
    
        if figsize:
            width, height = figsize
        else:
            width = 2.5
            height = 3.2
        
        height_text = 0.5
        height_evt = 0.2
        height_rest = height - (height_evt*2 + height_text*2)  
        
        with plt.rc_context(PYREA_RC_PARAMS):
            
            
            fig = plt.figure(figsize=(width, height))

            # grid           
            height_ratios = [height_text, height_evt, height_rest, height_evt, height_text]
            gs = fig.add_gridspec(nrows=5, ncols=1, height_ratios=height_ratios, hspace=0.05)
        
            # Common running sum plot
            ax_rs = fig.add_subplot(gs[2, 0])
            # leading edge from gene set 1
            ax_evt_1 = fig.add_subplot(gs[1, 0])
            ax_lbls_1 = fig.add_subplot(gs[0, 0], sharex=ax_evt_1)
            # leading edge from gene set 2
            ax_evt_2 = fig.add_subplot(gs[3, 0])
            ax_lbls_2 = fig.add_subplot(gs[4, 0], sharex=ax_evt_2)


            pl._plot_run_sum(self.rs_1, self.es_idx_1, ax=ax_rs, **rs_prop_1)
            leg_1 = pl._stats_legend(self.aREA_nes_1, 
                                        self.pval_1, 
                                        ax=ax_rs,
                                        leg_kw=leg_prop_1)
            ax_rs.add_artist(leg_1)
            pl._plot_run_sum(self.rs_2, self.es_idx_2, add=True, ax=ax_rs, **rs_prop_2)
            leg_2 = pl._stats_legend(self.aREA_nes_2, 
                                        self.pval_2, 
                                        ax=ax_rs,
                                        leg_kw=leg_prop_2)
            ax_rs.add_artist(leg_2)
            ax_rs.set_xticks([])
            for spine in ['top', 'bottom']:
                ax_rs.spines[spine].set_visible(False)

            # first gene set
            if evt_kw_1:
                evt_prop_1.update(evt_kw_1)
            ax_evt_1.eventplot(self.ledge_1['index'].values, **evt_prop_1)
            ax_evt_1.set_xticks([])
            ax_evt_1.set_yticks([])
            ax_evt_1.set_xlim(xmin_1, xmax_1)
            ax_evt_1.set_ylim(0, 1)
            for spine in ['top','bottom','left','right']:
                ax_evt_1.spines[spine].set_linewidth(0.25)
            ax_lbls_1.axis('off')
            ax_lbls_1.set_ylim(0, 1)

            # get patches
            patch_dict_1 = pl._ledge_patch_prep(True, 
                                           self.ledge_xinfo_1, 
                                           self.ledge_yinfo_1, 
                                           rect_prop_1, 
                                           conn_patch_prop,
                                           ax_rs=ax_rs, 
                                           ax_evt=ax_evt_1)
            # Draw connection lines
            fig.add_artist(patch_dict_1.get('conn_left'))     
            fig.add_artist(patch_dict_1.get('conn_right'))
            # Add shaded rectangles    
            ax_evt_1.add_artist(patch_dict_1.get('evt_rect'))
            ax_rs.add_artist(patch_dict_1.get('rs_rect'))
            # Add text
            pl._plot_ledge_labels(df_sub_1, 
                                True, 
                                self.ledge_xinfo_1, 
                                text_prop_1,
                                highlight=highlight_1, 
                                line_kw=conn_patch_prop,
                                ax=ax_lbls_1)


         # second gene set
            if evt_kw_2:
                evt_prop_2.update(evt_kw_2)
            ax_evt_2.eventplot(self.ledge_2['index'].values, **evt_prop_2)
            ax_evt_2.set_xticks([])
            ax_evt_2.set_yticks([])
            ax_evt_2.set_xlim(xmin_2, xmax_2)
            ax_evt_2.set_ylim(0, 1)
            for axis in ['top','bottom','left','right']:
                ax_evt_2.spines[axis].set_linewidth(0.25)
            ax_lbls_2.axis('off')
            ax_lbls_2.set_ylim(0, 1)

            # get patches
            patch_dict_2 = pl._ledge_patch_prep(False, 
                                           self.ledge_xinfo_2, 
                                           self.ledge_yinfo_2, 
                                           rect_prop_2, 
                                           conn_patch_prop,
                                           ax_rs=ax_rs, 
                                           ax_evt=ax_evt_2)
            # Draw connection lines
            fig.add_artist(patch_dict_2.get('conn_left'))     
            fig.add_artist(patch_dict_2.get('conn_right'))
            # Add shaded rectangles    
            ax_evt_2.add_artist(patch_dict_2.get('evt_rect'))
            ax_rs.add_artist(patch_dict_2.get('rs_rect'))
            # Add text
            pl._plot_ledge_labels(df_sub_2, 
                                False, 
                                self.ledge_xinfo_2, 
                                text_prop_2,
                                highlight=highlight_2, 
                                line_kw=conn_patch_prop,
                                ax=ax_lbls_2)

        return fig    


class GseaReg(Gsea1T):
    """ """

    def __init__(self, 
                 ges: pd.Series, 
                 regulon: pd.DataFrame):

        if not isinstance(ges, pd.Series):
            raise TypeError('Need an indexed pandas Series, please.')
        else:
            self.ges = _prep_ges(ges, asc_sort=False) # Gsea1T needs to be sorted from high to low for running sum
            self.ns = len(self.ges)
            self.along_scores = [*range(self.ns)]
            
            
        if not isinstance(regulon, pd.DataFrame):
            raise TypeError('Regulon needs to be supplied as a DataFrame, please.')
        else:
            self.regulator = ''.join(set(regulon['source']))
            self.regulon_in = regulon
            
        # if not all(self.regulon_in.columns.values == ['source',' target', 'mor', 'likelihood']):
        #     raise ValueError('Regulon value DataFrame must have columns: source, target, mor, likelihood')
    
        if not np.in1d(self.regulon_in['target'].values, self.ges.index).any():
            raise ValueError("None of the regulator targets were found in the gene expression signature index!")
        else:
            self.reg_df, self.reg_pos, self.reg_neg = self._split_regulon(self.regulon_in, self.ges.index)

        # Gene set 1 are the positive targets
        self.gs_idx_1 = self._find_hits(self.ges, self.reg_pos['target'].to_list())
        self.gs_idx_2 = self._find_hits(self.ges, self.reg_neg['target'].to_list())

        self.rs_1 = self._derive_rs(self.ges, self.gs_idx_1) #Default weight, cannot be modified
        self.es_idx_1 = np.abs(self.rs_1).argmax()
        self.left_end_closer_1 = self.es_idx_1 <= (self.ns-self.es_idx_1)
        
        self.rs_2 = self._derive_rs(self.ges, self.gs_idx_2)
        self.es_idx_2 = np.abs(self.rs_2).argmax()
        self.left_end_closer_2 = self.es_idx_2 <= (self.ns-self.es_idx_2)
       
        self.aREA_nes = aREA(self.ges, self.regulon_in, minsize=len(self.reg_df)).iloc[0][0]
        self.pval = norm.sf(np.abs(self.aREA_nes))*2
        
        self.ledge_1, self.ledge_xinfo_1 = self._get_ledge(self.ges, 
                                                           self.gs_idx_1, 
                                                           self.es_idx_1,
                                                           self.left_end_closer_1)
        
        self.ledge_2, self.ledge_xinfo_2 = self._get_ledge(self.ges, 
                                                           self.gs_idx_2, 
                                                           self.es_idx_2,
                                                           self.left_end_closer_2)

        if self.left_end_closer_1:
            self.ledge_yinfo_1 = 0, self.rs_1[self.es_idx_1]
        else:
            self.ledge_yinfo_1 = self.rs_1[self.es_idx_1], 0

        if self.left_end_closer_2:
            self.ledge_yinfo_2 = 0, self.rs_2[self.es_idx_2]
        else:
            self.ledge_yinfo_2 = self.rs_2[self.es_idx_2], 0
        
        
    def __repr__(self):
        
        """String representation for the GseaReg class"""

        return (
            f"GseaReg(GES length: {self.ns}\n"
            f"Regulator: {self.regulator}\n"
            f"Number of targets: {len(self.regulon_in)}, Overlap with GES: {len(self.reg_df)}\n"
            f"Positive targets: {len(self.reg_pos)}, Negative targets: {len(self.reg_neg)}\n"
        )
    
    def _split_regulon(self, 
                       reg_df:pd.DataFrame, 
                       ges_index: pd.Index):
        """This will divide the regulon into negative and positive targets

        Parameters
        ----------
        reg_df:pd.DataFrame :
            
        ges_index: pd.Index :
            

        Returns
        -------

        """
        # prefiltering
        reg_df_new = reg_df[reg_df['target'].isin(ges_index)] # We test that there must be some before, so safe to subset. 
        
        # In theory, it's not necessary to have both negative and positive targets. 
        # But this is about plotting, so we require both. 
        mask = reg_df_new['mor'].values>=0
        if all(mask) or not any(mask):
            raise ValueError("Could not identify two tails in the gene expression signature, please double check.")
            
        return reg_df_new, reg_df_new[mask], reg_df_new[~mask]
        
    
    def plot(self,
             figsize: tuple=None,
             conditions: tuple = ('High', 'Low'),
             ges_type: str = None,
             ges_symlog: bool=True,
             ges_stat_fmt:str='1.0f',
             ges_kw: dict = None,
             evt_kw_1: dict = None,
             rs_kw_1: dict = None,
             evt_kw_2: dict = None,
             rs_kw_2: dict = None,
             stat_legend_kw:dict =None,
             target_leg_kw:dict=None
             ):
        """

        Parameters
        ----------
        figsize: tuple :
             (Default value = (3)
        3) :
            
        conditions: tuple :
             (Default value = ('A')
        'B') :
            
        ges_type: str :
             (Default value = None)
        ges_symlog: bool :
             (Default value = True)
        ges_stat_fmt:str :
             (Default value = '1.0f')
        ges_kw: dict :
             (Default value = None)
        evt_kw_1: dict :
             (Default value = None)
        rs_kw_1: dict :
             (Default value = None)
        evt_kw_2: dict :
             (Default value = None)
        rs_kw_2: dict :
             (Default value = None)
        stat_kw:dict :
             (Default value = None)
        leg_kw:dict :
             (Default value = None)

        Returns
        -------

        """
        
        # Some defaults
        ges_prop = {'color':'.5', 'alpha':0.25, 'linewidth':0.1}
        if ges_kw:
            ges_prop.update(ges_kw)
        # Line properties for running sum lines
        rs_prop_1 = {'color': '#AC3220'} # Chinese Red
        if rs_kw_1:
                rs_prop_1.update(rs_kw_1)
        rs_prop_2 = {'color': '#50808E'} # Teal Blue
        if rs_kw_2:
            rs_prop_2.update(rs_kw_2)            

        evt_prop_1 = {'color': rs_prop_1.get("color"), 'alpha':0.7, 'linewidths':0.5} 
        if evt_kw_1:
            evt_prop_1.update(evt_kw_1)        
        evt_prop_2 = {'color': rs_prop_2.get("color"), 'alpha':0.7, 'linewidths':0.5}
        if evt_kw_2:
            evt_prop_2.update(evt_kw_2)
            
       
        stat_legend_prop = {'loc':3, "title":self.regulator} # Stats go to bottom left by default
        if stat_legend_kw:
            stat_legend_prop.update(stat_legend_kw)

        if figsize:
            width, height = figsize
        else:
            width = 2.5
            height = 2.4

        height_evt = 0.2
        height_ges = 0.5
        height_rest = height - (height_evt*2 + height_ges)
        

        with plt.rc_context(PYREA_RC_PARAMS):
                
            fig = plt.figure(figsize=(width, height))

            gs = fig.add_gridspec(4, 1, 
                                  height_ratios=[height_ges, height_evt, height_rest, height_evt], 
                                  hspace=0)

            # first graph
            ax_ges = fig.add_subplot(gs[0, 0])
          
            pl._plot_ges(self.along_scores, 
                        self.ges.values, 
                        conditions=conditions,
                        is_high_to_low=True,
                        ges_type=ges_type, 
                        symlog=ges_symlog,
                        stat_fmt=ges_stat_fmt,
                        ax=ax_ges, 
                        **ges_prop)
            
            # second graph: bars to indicate positions of FIRST GENE SET genes
            ax_evt1 = fig.add_subplot(gs[1,0], sharex=ax_ges)
            ax_evt1.eventplot(self.gs_idx_1, **evt_prop_1)
            ax_evt1.axis('off')

            # Third graph: Running sums
            
            ax_rs = fig.add_subplot(gs[2,0])
            pl._plot_run_sum(self.rs_1, self.es_idx_1, ax=ax_rs, **rs_prop_1)
            pl._plot_run_sum(self.rs_2, self.es_idx_2, ax=ax_rs, add=True, **rs_prop_2)
            # First legend illustrates the regulon tails
            leg_targets = pl._add_reg_legend(rs_prop_1.get('color'), rs_prop_2.get('color'), 
                                            ax=ax_rs, target_leg_kw=target_leg_kw)
            ax_rs.add_artist(leg_targets)    
            # Second legend adds stat info and regulator name
            leg_stats = pl._stats_legend(self.aREA_nes, self.pval, ax=ax_rs, leg_kw=stat_legend_prop)
            ax_rs.add_artist(leg_stats)        
            ax_rs.set_xticks([])
            ax_rs.spines['bottom'].set_visible(False)
            
            # fourth graph: bars to indicate positions of SECOND GENE SET genes
            ax_evt2 = fig.add_subplot(gs[3,0], sharex=ax_rs)
            ax_evt2.eventplot(self.gs_idx_2, **evt_prop_2)
            ax_evt2.set_yticks([])
            ax_evt2.tick_params(labelsize='x-small')
            pl._format_xaxis_ges(self.ns, ax=ax_evt2)
            for spine in ['left', 'right']:
                ax_evt2.spines[spine].set_visible(False)
        
        return fig
    
    def plot_ledge(self,
         figsize: tuple=None,
         n_genes: int = None,
         conn_patch_kw: dict = None,
        subset_1: dict = None,
        highlight_1: list = None,
        rs_kw_1: dict = None,
        leg_kw_1: dict = None,
        evt_kw_1: dict = None,
        text_kw_1: dict = None,
        rect_kw_1: dict = None,
        subset_2: dict = None,
        highlight_2: list = None,
        rs_kw_2: dict = None,
        leg_kw_2: dict = None,
        evt_kw_2: dict = None,
        text_kw_2: dict = None,
        rect_kw_2: dict = None,
        stat_leg_kw:dict=None,
        target_leg_kw:dict=None):

        """

        Parameters
        ----------
        figsize: tuple :
             (Default value = (3)
        3) :
            
        stat_kw:dict :
             (Default value = None)
        leg_kw:dict :
             (Default value = None)
        lbl_kw_1: dict :
             (Default value = None)
        ledge_trim_1:int :
             (Default value = None)
        rs_kw_1: dict :
             (Default value = None)
        patch_kw_1:dict :
             (Default value = None)
        highlight_set_1: tuple :
             (Default value = None)
        lbl_kw_2:dict :
             (Default value = None)
        ledge_trim_2:int :
             (Default value = None)
        rs_kw_2:dict :
             (Default value = None)
        patch_kw_2:dict :
             (Default value = None)
        highlight_set_2:tuple :
             (Default value = None)

        Returns
        -------

        """
        # setup
        # Common parameters
        # Legend = statistics
        stat_legend_prop = {'loc':"lower left", "title":self.regulator} # Stats go to bottom left by default
        if stat_leg_kw:
            stat_legend_prop.update(stat_leg_kw)
        
        conn_patch_prop = {'color':'.15', 'lw':0.25} # this will be the same for both

        text_prop_1 = {'fontsize':'xx-small', 'rotation':90, 'ha':'center', 'va':'bottom'}
        text_prop_2 = {'fontsize':'xx-small', 'rotation':90, 'ha':'center', 'va':'top'}

        rs_prop_1 = {'color':'#AC3220'} #Chinese red
        evt_prop_1 = {'color': rs_prop_1.get('color'), 'alpha':0.7, 'linewidths':0.5, 'lineoffsets':0.5}
        rect_prop_1 = {'color': rs_prop_1.get('color'), 'alpha':0.25, 'lw':0}
     
        rs_prop_2 = {'color':'#50808E'} #Teal blue
        evt_prop_2 = {'color': rs_prop_2.get('color'), 'alpha':0.7, 'linewidths':0.5, 'lineoffsets':0.5}
        rect_prop_2 = {'color': rs_prop_2.get('color'), 'alpha':0.25, 'lw':0}
     
         # Prepare gene labels 
        df_1 = self.ledge_1.copy()
        df_sub_1 = self._filter_ledge(df_1, self.left_end_closer_1, n_genes, subset=subset_1)  
        xmin_1, xmax_1 = self.ledge_xinfo_1
        
        df_2 = self.ledge_2.copy()
        df_sub_2 = self._filter_ledge(df_2, self.left_end_closer_2, n_genes, subset=subset_2)  
        xmin_2, xmax_2 = self.ledge_xinfo_2

        if conn_patch_kw:
            conn_patch_prop.update(conn_patch_kw)
    
        if rs_kw_1:
            rs_prop_1.update(rs_kw_1)
        if text_kw_1:
            text_prop_1.update(text_kw_1)
        if rect_kw_1:
            rect_prop_1.update(rect_kw_1)

        if rs_kw_2:
            rs_prop_2.update(rs_kw_2)
        if text_kw_2:
            text_prop_2.update(text_kw_2)
        if rect_kw_2:
            rect_prop_2.update(rect_kw_2)
     
        if figsize:
            width, height = figsize
        else:
            width = 2.5
            height = 3.2
        
        height_text = 0.5
        height_evt = 0.2
        height_rest = height - (height_evt*2 + height_text*2)  
    
        
        with plt.rc_context(PYREA_RC_PARAMS):
            
            fig = plt.figure(figsize=(width, height))      
        
            height_ratios = [height_text, height_evt, height_rest, height_evt, height_text]
            gs = fig.add_gridspec(nrows=5, ncols=1, height_ratios=height_ratios, hspace=0.05)
        
            
            # Common running sum plot
            ax_rs = fig.add_subplot(gs[2, 0])
            # leading edge from gene set 1
            ax_evt_1 = fig.add_subplot(gs[1, 0])
            ax_lbls_1 = fig.add_subplot(gs[0, 0], sharex=ax_evt_1)
            # leading edge from gene set 2
            ax_evt_2 = fig.add_subplot(gs[3, 0])
            ax_lbls_2 = fig.add_subplot(gs[4, 0], sharex=ax_evt_2)

                
            pl._plot_run_sum(self.rs_1, self.es_idx_1, ax=ax_rs, **rs_prop_1)
            leg_stats = pl._stats_legend(self.aREA_nes, 
                                        self.pval, 
                                        ax=ax_rs,
                                        leg_kw=stat_legend_prop)
            ax_rs.add_artist(leg_stats)
            pl._plot_run_sum(self.rs_2, self.es_idx_2, add=True, ax=ax_rs, **rs_prop_2)
            leg_targets = pl._add_reg_legend(rs_prop_1.get('color'), rs_prop_2.get('color'), 
                                            ax=ax_rs, target_leg_kw=target_leg_kw)
            ax_rs.add_artist(leg_targets)
            ax_rs.set_xticks([])
            for spine in ['top', 'bottom']:
                ax_rs.spines[spine].set_visible(False)
            
            # first gene set
            if evt_kw_1:
                evt_prop_1.update(evt_kw_1)
            ax_evt_1.eventplot(self.ledge_1['index'].values, **evt_prop_1)
            ax_evt_1.set_xticks([])
            ax_evt_1.set_yticks([])
            ax_evt_1.set_xlim(xmin_1, xmax_1)
            ax_evt_1.set_ylim(0, 1)
            for spine in ['top','bottom','left','right']:
                ax_evt_1.spines[spine].set_linewidth(0.25)
            ax_lbls_1.axis('off')
            ax_lbls_1.set_ylim(0, 1)

            # Get patches
            patch_dict_1 = pl._ledge_patch_prep(True, 
                                           self.ledge_xinfo_1, 
                                           self.ledge_yinfo_1, 
                                           rect_prop_1, 
                                           conn_patch_prop,
                                           ax_rs=ax_rs, 
                                           ax_evt=ax_evt_1)
            # Draw connection lines
            fig.add_artist(patch_dict_1.get('conn_left'))     
            fig.add_artist(patch_dict_1.get('conn_right'))
            # Add shaded rectangles    
            ax_evt_1.add_artist(patch_dict_1.get('evt_rect'))
            ax_rs.add_artist(patch_dict_1.get('rs_rect'))

              # Add text
            pl._plot_ledge_labels(df_sub_1, 
                                True, 
                                self.ledge_xinfo_1, 
                                text_prop_1,
                                highlight=highlight_1, 
                                line_kw=conn_patch_prop,
                                ax=ax_lbls_1)

            # second gene set
            if evt_kw_2:
                evt_prop_2.update(evt_kw_2)
            ax_evt_2.eventplot(self.ledge_2['index'].values, **evt_prop_2)
            ax_evt_2.set_xticks([])
            ax_evt_2.set_yticks([])
            ax_evt_2.set_xlim(xmin_2, xmax_2)
            ax_evt_2.set_ylim(0, 1)
            for axis in ['top','bottom','left','right']:
                ax_evt_2.spines[axis].set_linewidth(0.25)
            ax_lbls_2.axis('off')
            ax_lbls_2.set_ylim(0, 1)

            # get patches
            patch_dict_2 = pl._ledge_patch_prep(False, 
                                           self.ledge_xinfo_2, 
                                           self.ledge_yinfo_2, 
                                           rect_prop_2, 
                                           conn_patch_prop,
                                           ax_rs=ax_rs, 
                                           ax_evt=ax_evt_2)
            # Draw connection lines
            fig.add_artist(patch_dict_2.get('conn_left'))     
            fig.add_artist(patch_dict_2.get('conn_right'))
            # Add shaded rectangles    
            ax_evt_2.add_artist(patch_dict_2.get('evt_rect'))
            ax_rs.add_artist(patch_dict_2.get('rs_rect'))
            # Add text
            pl._plot_ledge_labels(df_sub_2, 
                                False, 
                                self.ledge_xinfo_2, 
                                text_prop_2,
                                highlight=highlight_2, 
                                line_kw=conn_patch_prop,
                                ax=ax_lbls_2)

        return fig 
    
    
class GseaRegMultSigs(Gsea1TMultSigs, GseaReg):
    """ """
     
    def __init__(self, 
                 dset: pd.DataFrame, 
                 regulon: pd.DataFrame,  
                 ordered: bool=True):
                    
        if not isinstance(dset, pd.DataFrame):
            raise TypeError('Need an indexed pandas DataFrame, please.')
        
        else: 
            self.dset = _prep_ges(dset)
            self.ns = len(self.dset)
            self.samples = self.dset.columns.values
             
        if not isinstance(regulon, pd.DataFrame):
            raise TypeError('Regulon needs to be supplied as a DataFrame, please.')
        else:
            self.regulator = ''.join(set(regulon['source']))
            self.regulon_in = regulon
       
        # if not all(self.reg_df_in.columns.values == np.array(['target', 'mor', 'likelihood'])):
        #     raise ValueError('Regulon value DataFrame must have columns: target, mor, likelihood')

        if not np.in1d(self.regulon_in['target'].values, self.dset.index).any():
            raise ValueError("None of the regulator targets were found in the gene expression signature index!")
        else:
            self.reg_df, self.reg_pos, self.reg_neg = self._split_regulon(self.regulon_in, self.dset.index)
            
        self.stats = self._get_stats(dset=self.dset, 
                                     regulon=regulon, 
                                     minsize=len(self.reg_df),
                                     samples=self.samples)
        
        self.gs_idx = self._find_hits(self.dset, 
                                      self.reg_pos['target'].to_list(), 
                                      self.reg_neg['target'].to_list())

        if ordered:
            idx = self.stats['NES'].argsort().values
            self.stats = self.stats.take(idx, axis=0)
            self.stats.reset_index(inplace=True, drop=True)
            self.gs_idx = self.gs_idx.take(idx, axis=0)
            
    
    def __repr__(self):
        
        """String representation for the Gsea1TMultSig class"""

        return (
                f"GseaRegMultSig(Number of genes: {self.ns}\n"
                f"Number of signatures: {len(self.samples)}\n"
                f"Regulator: {self.regulator}\n"
                f"Number targets: {len(self.regulon_in)}, Overlap: {len(self.reg_df)}\n"
                )        
        
    def plot(self, 
             regulon_colors:tuple = ('#AC3220', '#50808E'),
             figsize: tuple = None,
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
        regulon_colors:tuple :
             (Default value = ('#AC3220')
        '#50808E') :
            
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
        
        evt_data = [arr for tup in self.gs_idx.values for arr in tup]
        
        evt_prop = {'linelengths':0.33, 'alpha':0.7, 'linewidths':0.5}
        
        norm, pcm_prop = pl._prepare_nes_colors(df, norm_kw, pcm_kw)
      
        # Prettify FDR format
        df['FDR'] = df['FDR'].apply(pl._fdr_formatter)
               
        if figsize:
            width, height = figsize
        else:
            height = len(df) * 0.5
            width = 3/4 * height
            
        height_cbar = 0.15
        height_rest = height - height_cbar
        width_nes = 0.3
        width_rest = width - width_nes 
        
        with plt.rc_context(PYREA_RC_PARAMS):
                    
            fig = plt.figure(figsize=(width, height), dpi=200)
            
            gs = fig.add_gridspec(nrows=2, ncols=2, 
                                hspace=0.2, 
                                wspace=0.025,
                                width_ratios=[width_rest, width_nes],
                                height_ratios=[height_cbar, height_rest])
            
            # Plot 1: Illustrate the ordered signature as a colorbar
            ax_cbar = fig.add_subplot(gs[0, 0])
            cb = fig.colorbar(ScalarMappable(cmap=plt.cm.RdBu_r), 
                            orientation='horizontal', ticks=[], cax=ax_cbar)
            #cb.outline.set_visible(False)
            ax_cbar.text(0, 0.5, 'Low', color='w', ha='left', va='center', fontsize='x-small')
            ax_cbar.text(1, 0.5, 'High', color='w', ha='right', va='center', fontsize='x-small')
        
            # Plot 2: Illustrate targets       
            if evt_kw:
                evt_prop.update(evt_kw)
            
            lofs, ytick_pos = pl._prepare_multi_gseareg(linelengths=evt_prop.get('linelengths'), 
                                            number_of_ys=len(self.samples))
            
            cls = np.tile(regulon_colors, len(self.samples))
                
            ax_evt = fig.add_subplot(gs[1,0])                  
            ax_evt.eventplot(evt_data, lineoffsets=lofs, colors=cls, **evt_prop)
            ax_evt.set_yticks(ytick_pos)
            add_y = evt_prop.get('linelengths')
            ax_evt.set_ylim([0-add_y, np.max(lofs)+add_y])
            ax_evt.set_yticklabels(df['signature_name'].values, fontsize='x-small', va='top')
            ax_evt.set_xlim([-.5, self.ns+.5])
            ax_evt.xaxis.set_label_position('top')
            ax_evt.xaxis.tick_top()
            pl._format_xaxis_ges(self.ns, ax=ax_evt)
        # x-axis formating
            for spine in ['right', 'left']:
                ax_evt.spines[spine].set_visible(False)
            ax_evt.annotate(text=f"{self.regulator} regulon", 
                            xy=(0.5, -0.01), xycoords='axes fraction', 
                            ha='center', va='top', fontsize='small')
        
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
            ax_titles.text(0.5, 0, 'NES', fontsize='x-small', c='k', ha='center', va='top')
            ax_titles.text(1.1, 0, 'FDR', fontsize='x-small', c='k', ha='left', va='top')
            ax_titles.axis('off')
        
        return fig




class GseaMultReg:
    
    """To implement one-tailed gene set enrichment of multiple regulons on one gene expression signature
    and visualize the results

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(self, 
                 ges: pd.Series, 
                 regulons: pd.DataFrame,
                 minsize: int = 20,
                 ordered: bool=True):
        
        if not isinstance(ges, pd.Series):
            raise TypeError('Need an indexed pandas Series, please.')
        else:
            self.ges = _prep_ges(ges)
            self.ns = len(self.ges)
            self.along_scores = [*range(self.ns)]
            self.minsize = minsize
            
        if not isinstance(regulons, pd.DataFrame):
            raise TypeError('Regulons need to be supplied as a DataFrame, please.')
        
        self.regulons_in = regulons['source'].nunique()
           
        self.targets = self._split_regulons(regulons=regulons)
        
        self.target_idx = self._find_hits(self.ges, self.targets)
        
        self.stats = self._get_stats(self.ges, regulons, minsize=self.minsize)
                  
        if ordered:
            idx = self.stats['NES'].argsort().values
            self.stats = self.stats.take(idx, axis=0)
            self.stats.reset_index(inplace=True, drop=True)
            
            
    def __repr__(self):

        """String representation for the Gsea1TMultSet class"""

        return (
                f"GseaMultReg(GES length: {self.ns}\n"
                f"Regulators provided: {self.regulons_in}\n"
                f"Regulators evaluated: {len(self.stats)}"
                )
        
    def _split_targets(self, df):
        """

        Parameters
        ----------
        df :
            

        Returns
        -------

        """
        mask = df['mor']>=0
        return df['target'].values[mask], df['target'].values[~mask]
  
    def _split_regulons(self, regulons:pd.DataFrame):
        """

        Parameters
        ----------
        regulons:pd.DataFrame :
            

        Returns
        -------

        """
        targets = regulons.groupby('source').apply(self._split_targets)
        
        target_df = pd.DataFrame(zip(*targets), 
                               columns=targets.index, 
                               index=['positive_targets', 'negative_targets'])
        
        return target_df.T
        
    def _find_hits(self, 
                   ges: pd.Series, 
                   target_df: pd.DataFrame):
        """

        Parameters
        ----------
        ges :
            pd.Series:
        gene_sets :
            dict:
        ges: pd.Series :
            
        target_df: pd.DataFrame :
            

        Returns
        -------

        """

        # This will have to be sorted first, which is the case in this Class
        ges = ges.argsort()
        
        pos = [ges[ges.index.intersection(val)].values for val in target_df['positive_targets'].values]
        neg = [ges[ges.index.intersection(val)].values for val in target_df['negative_targets'].values]
        
        pos_df = pd.DataFrame(zip(pos, neg), 
                            index=target_df.index, 
                            columns=['pos_idx', 'neg_idx'])
        
        return pos_df
    
    def _get_stats(self, 
                   ges: pd.Series,
                   regulons: pd.DataFrame,
                   minsize:int):
        
        """Computes normalized enrichment scores and some tools for later visualization

        Parameters
        ----------
        ges :
            pd.Series:
        regulon :
            dict:
        ges: pd.Series :
            
        regulons: pd.DataFrame :
            
        minsize:int :
            

        Returns
        -------

        """
        
        nes = aREA(ges, regulons, minsize=minsize)
        nes.columns = ['NES']
        nes.index.name = 'Regulator'
        nes['pvals'] = norm.sf(np.abs(nes['NES'].values))*2 # retrieve two-sided p-value from normal distribution
        nes['FDR'] = multipletests(nes['pvals'].values, method = 'fdr_bh')[1] #FDR
        nes.reset_index(inplace=True)

        return nes
                    
    def plot(self, 
             conditions: tuple = ('Low', 'High'),
             regulon_colors: tuple=('#AC3220', '#50808E'),
             ges_symlog: bool=True,
             ges_stat_fmt:str='1.0f',
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
        regulon_colors: tuple :
             (Default value = ('#AC3220')
        '#50808E') :
            
        ges_symlog: bool :
             (Default value = True)
        ges_stat_fmt:str :
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
        evt_prop = {'alpha':0.7, 'linelengths':0.5, 'linewidths':0.5}        
        
        norm, pcm_prop = pl._prepare_nes_colors(df, norm_kw, pcm_kw)
                        
        # prettify FDR 
        df['FDR'] = df['FDR'].apply(pl._fdr_formatter)
                    
        if subset:
            df = self._filter_multset_stats(df, subset)
        elif len(df)>20:
            top_rp = df.take(df['NES'].abs().nlargest(20).index)['Regulator']
            df = df[df['Regulator'].isin(top_rp)]
            df.reset_index(inplace=True, drop=True)
                  
        targets = self.target_idx.loc[df['Regulator']].values
        evt_data = [arr for tup in targets for arr in tup]
        
         # Setup figure and gridspec
        if figsize:
            width, height = figsize
        else:
            height = len(df) * 0.2    
            width = 3/4 * height
        
        height_ges = 0.6
        height_rest = height - height_ges
        width_nes = 0.3
        width_rest = width - width_nes
        
        
        with plt.rc_context(PYREA_RC_PARAMS):
            
            fig = plt.figure(figsize=(width, height)) 
                        
            gs = fig.add_gridspec(nrows=2, ncols=2, 
                                  hspace=0.025, wspace=0.025,
                                  width_ratios=[width_rest, width_nes],
                                  height_ratios=[height_ges, height_rest])
        
            # Plot 1: Illustrate the ordered signature
            ax_ges = fig.add_subplot(gs[0,0])
            if ges_kw:
                ges_prop.update(ges_kw)
                
            pl._plot_ges(self.along_scores, 
                        self.ges.values, 
                        ges_type=ges_type, 
                        conditions=conditions, 
                        is_high_to_low=False, 
                        symlog=ges_symlog,
                        stat_fmt=ges_stat_fmt,
                        ax=ax_ges, **ges_prop)
            
            # Plot 2: Illustrate targets
            ax_evt = fig.add_subplot(gs[1,0])
            
            lofs, ytick_pos = pl._prepare_multi_gseareg(linelengths=evt_prop.get('linelengths'), 
                                            number_of_ys=len(df))
            
            cls = np.tile(regulon_colors, len(df))
            
            if evt_kw:
                evt_prop.update(evt_kw)
                
            ax_evt.eventplot(evt_data, lineoffsets=lofs, colors=cls, **evt_prop)
            ax_evt.set_yticks(ytick_pos)
            add_y = evt_prop.get('linelengths')
            ax_evt.set_ylim([0-add_y, np.max(lofs)+add_y])
            ax_evt.set_yticklabels(df['Regulator'].values, fontsize='x-small', va='top')
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

