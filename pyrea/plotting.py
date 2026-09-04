
"""Outsourced some helper functions for readability"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, ConnectionPatch
from typing import Sequence, Union

def _plot_ges(along_scores: list, 
              ges_values:np.ndarray, 
              ges_type: str, 
              conditions: tuple, 
              symlog:bool,
              stat_fmt:str,
              is_high_to_low: bool = True,
              ax = None,
              **kwargs):
    """

    Parameters
    ----------
    along_scores: list :
        
    ges_values:np.ndarray :
        
    ges_type: str :
        
    conditions: tuple :
        
    symlog:bool :
        
    stat_fmt:str :
        
    is_high_to_low: bool :
         (Default value = True)
    ax :
         (Default value = None)
    **kwargs :
        

    Returns
    -------

    """
    if ax is None:
        ax = plt.gca()
        
    ax.fill_between(along_scores, ges_values, **kwargs) #filler
    ax.set_xticks([])
    if ges_type is None:
        ges_type = 'Gene score'
    ax.set_ylabel(ges_type, fontsize='xx-small')
    ax.axhline(y=0, linestyle='-', lw=0.5, c='.15')
    ax.set_xlim([-.5, len(ges_values)+.5])
    p1, p2 = conditions
    if is_high_to_low:
        ax.annotate(p1, xy=(0.05, 0.05), xycoords='axes fraction', ha='left', va='bottom', fontsize='small')
        ax.annotate(p2, xy=(0.95, 0.95), xycoords='axes fraction', ha='right', va='top', fontsize='small')
    else:
        ax.annotate(p1, xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top', fontsize='small')
        ax.annotate(p2, xy=(0.95, 0.05), xycoords='axes fraction', ha='right', va='bottom', fontsize='small')
    ax.yaxis.set_tick_params(labelsize='xx-small')
    if symlog:
        ax.set_yscale('symlog')
    yticks = [np.min(ges_values), 0, np.max(ges_values)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(list(map(lambda x: f'{x:{stat_fmt}}', yticks)))
    
    return ax

def _plot_run_sum(run_sum:np.ndarray, 
                  es_idx: int, 
                  add: bool=False,
                  ax=None,
                  **kwargs):
    """

    Parameters
    ----------
    run_sum:np.ndarray :
        
    es_idx: int :
        
    add: bool :
         (Default value = False)
    ax :
         (Default value = None)
    **kwargs :
        

    Returns
    -------

    """
    if ax is None:
        ax = plt.gca()
        
    lcolor = kwargs.get('color', 'C0')
    
    if add:
        ax.plot(run_sum, **kwargs)
        ax.vlines(es_idx, 0, run_sum[es_idx], color=lcolor, linestyle='--', lw=.8)
    else:
        ax.plot(run_sum, **kwargs)
        ax.axhline(y=0, linestyle='-', c='.15', lw=0.5)
        ax.vlines(es_idx, 0, run_sum[es_idx], color=lcolor, linestyle='--', lw=.8)
        ax.tick_params(labelsize='x-small')
        ax.set_ylabel('ES', fontsize='small')
    
    return ax
    
def _stats_legend(nes: float, 
                pval: float,
                ax=None,
                leg_kw: dict = None):
    
    """Turn text for normalized enrichment score into a legend artist for easy placement in the axes.

    Parameters
    ----------
    nes: float :
        
    pval: float :
        
    leg_kw: dict :
         (Default value = None)

    Returns
    -------

    """
    
    if ax is None:
        ax = plt.gca()


    leg_prop = {'loc':0, 
                'handlelength':0, 
                'handletextpad':0, 
                'labelspacing':0.2, 
                'edgecolor':'.15', 
                'fontsize':'xx-small', 
                'title_fontsize':'x-small',
                'labelcolor':'0.15',
                }
    
    if leg_kw:
        leg_prop.update(leg_kw)
            
    ptch = mpl.patches.Patch(color='w')
    nes, pval = f"NES: {nes:1.2f}", f"p: {pval:1.1e}"
      
    leg = ax.legend([ptch]*2, [nes, pval], **leg_prop)
    leg.get_frame().set_linewidth(0.5)
    plt.setp(leg.get_title(), color=leg_prop.get('labelcolor'))
        
    return leg

def _add_reg_legend(color_pos:str, 
                    color_neg:str, 
                    ax=None,
                    target_leg_kw:dict=None):
    """

    Parameters
    ----------
    color_pos:str :
        
    color_neg:str :
        
    leg_kw:dict :
         (Default value = None)

    Returns
    -------

    """
    if ax is None:
        ax = plt.gca()

    leg_prop = {'loc':1, #default position is top right
                'edgecolor':'.15', 
                'labelspacing':0.2,
                'handlelength':0.75, 
                'handletextpad':0.4,
                'fontsize':'xx-small', 
                'title':'Targets',
                'title_fontsize':'x-small'}
    
    if target_leg_kw is not None:
        leg_prop.update(target_leg_kw)
        
    handles = [Line2D([0],[0], lw=2, color=color_pos), Line2D([0],[0], lw=2, color=color_neg)]

    leg = ax.legend(handles, ['positive', 'negative'], **leg_prop)
    leg.get_frame().set_linewidth(0.5)
    
    return leg

    
def _plot_ledge_labels(ledge_sub:pd.DataFrame, 
                       left_end_closer: bool, 
                       ledge_xinfo:tuple, 
                       text_kw:dict, 
                       highlight:list=None,
                       line_kw:dict=None, 
                       ax=None):
    
    if ax is None:
        ax =  plt.gca()
    
    line_prop = {'lw':.2, 'color':'.5'}
    
    if line_kw:
        line_prop.update(line_kw)
        
    df = ledge_sub.copy()
    xmin, xmax = ledge_xinfo
    
    if left_end_closer: 
        y_course = [0, 0.05, 0.25]
        gene_name_y = 0.3
        df['gene_name_xposition'] = np.linspace(xmin, xmax, len(df))
    else:
        y_course = [1, 0.95, 0.75]
        gene_name_y = 0.7
        text_kw.update({"va":"top"})
        df['gene_name_xposition'] = np.linspace(xmax, xmin, len(df))
    
    [ax.plot([row.index, row.index, row.gene_name_xposition], y_course, **line_prop) for row in df.itertuples()]
    
        
    for row in df.itertuples():
        if highlight:
            text_prop_highlight = text_kw.copy()            
            text_prop_highlight.update({'fontweight':'bold', 'color':'r'})

            if row.gene in highlight:
                ax.text(row.gene_name_xposition, gene_name_y, row.gene, **text_prop_highlight)
            else: 
                ax.text(row.gene_name_xposition, gene_name_y, row.gene, **text_kw)
        else:
            ax.text(row.gene_name_xposition, gene_name_y, row.gene, **text_kw)
      
    return ax
    

def _ledge_patch_prep(left_end_closer: bool, 
                      ledge_xinfo: tuple, 
                      ledge_yinfo:tuple, 
                      rect_prop:dict,
                      conn_patch_prop:dict,
                      *,  
                      ax_rs, 
                      ax_evt, 
                      ):
      
      xmin, xmax = ledge_xinfo
      ymin, ymax = ledge_yinfo
      rect2 = Rectangle((xmin, 0), xmax, 1, **rect_prop)
      
      if left_end_closer:
            con1 = ConnectionPatch(xyA=(xmin, ymax), coordsA=ax_rs.transData, 
                       xyB=(xmin, 0), coordsB=ax_evt.transData, **conn_patch_prop)
            con2 = ConnectionPatch(xyA=(xmax, ymax), coordsA=ax_rs.transData, 
                       xyB=(xmax, 0), coordsB=ax_evt.transData, **conn_patch_prop)
            rect1 = Rectangle((xmin, ymin), (xmax-xmin), ymax, **rect_prop)
            
            
      else:
            con1 = ConnectionPatch(xyA=(xmin, ymin), coordsA=ax_rs.transData, 
                       xyB=(xmin, 1), coordsB=ax_evt.transData, **conn_patch_prop)
            con2 = ConnectionPatch(xyA=(xmax, ymin), coordsA=ax_rs.transData, 
                       xyB=(xmax, 1), coordsB=ax_evt.transData, **conn_patch_prop)
            rect1 = Rectangle((xmin, ymin), (xmax-xmin), abs(ymin), **rect_prop)
      
      patch_dict = {'rs_rect':rect1, 'evt_rect':rect2, 'conn_left':con1, 'conn_right':con2}
      
      return patch_dict
      
    

 # Fix FDR for plotting
def _fdr_formatter(val: float):
    """

    Parameters
    ----------
    val: float :
        

    Returns
    -------

    """
    if val < 2e-16:
        return r'<2$e^{-16}$'
    else:
        a, b = f'{val:.1e}'.split('e')
        b = int(b)
    return fr'${a}e^{{{b}}}$'

def _prepare_multi_gseareg(linelengths:float, 
                                number_of_ys:int, 
                                space:float=0.2, 
                                start:float=0):
    """

    Parameters
    ----------
    linelengths:float :
        
    number_of_ys:int :
        
    space:float :
         (Default value = 0.2)
    start:float :
         (Default value = 0)

    Returns
    -------

    """
    
    lineoffsets = []
    tmp = start
    for i in range(number_of_ys*2):
        if i == 0:
            lineoffsets.append(start)
        elif i%2!=0:
            tmp += linelengths
            lineoffsets.append(tmp)
        else:
            tmp += space + linelengths
            lineoffsets.append(tmp)
        
    ytick_pos = [x for i, x in enumerate(lineoffsets) if i%2!=0]
    
    return lineoffsets, ytick_pos

def _format_xaxis_ges(ges_length:int, ax=None):
    """

    Parameters
    ----------
    ges_length:int :
        
    stepsize:int :
         (Default value = 1000)
    ax :
         (Default value = None)

    Returns
    -------

    """
    
    if ax is None:
        ax = plt.gca()
    
    if ges_length >= 10_000:
        stepsize = 1000
    elif ges_length >=2_000:
        stepsize = 500
    elif ges_length >= 500:
        stepsize = 100
    else:
        stepsize = 50
        
    xticks = np.percentile(np.arange(0, ges_length, stepsize), [0, 50, 100])
    xtick_aln = ['left', 'center', 'right']
    
    ax.set_xticks(xticks)
    for i, tick in enumerate(ax.xaxis.get_majorticklabels()):
        tick.set_horizontalalignment(xtick_aln[i])  
    ax.xaxis.set_tick_params(labelsize='x-small')
    
    return ax


def _color_light_or_dark(rgba_in:np.ndarray)-> str:
    """[For plotting purposes, we determine whether a color is light or dark and adjust its text color accordingly.
    Also see https://stackoverflow.com/questions/22603510/is-this-possible-to-detect-a-colour-is-a-light-or-dark-colour]

    Args:
        rgba_in ([np.ndarray]): [A numpy array containing RGBA as returned by matplotlib colormaps]

    Returns:
        [str]: [A string: w for white or k for black]
    """
    r,g,b,_ = rgba_in*255
    hsp = np.sqrt(0.299 * (r * r) + 0.587 * (g * g) + 0.114 * (b * b))
    if (hsp>127.5):
        # light color, return black for text
        return 'k'
    else:
        # dark color, return white for text
        return 'w'



def _prepare_nes_colors(stats:pd.DataFrame,
                        norm_kw: dict=None, 
                        pcm_kw: dict=None):
    
    nes = stats['NES'].values
    
    pcm_prop = {'edgecolor':'.25', 'lw':.5}
    
    if all(nes>=0):
        norm_prop = {'vmin':0, 'vmax':np.max(nes)}
        pcm_prop.update({'cmap':'Reds'})
        
    elif all(nes<=0):
        norm_prop = {'vmin':np.min(nes), 'vmax':0}
        pcm_prop.update({'cmap':'Blues_r'})
        
    else:
        norm_prop = {'vmin':np.min(nes), 'vcenter':0, 'vmax':np.max(nes)}
        pcm_prop.update({'cmap':'PuOr_r'})
        
        
    if norm_kw is not None:
        norm_prop.update(norm_kw)
            
    if pcm_kw is not None:
        pcm_prop.update(pcm_kw)        
            
    if all(nes>=0) | all(nes<=0):
        norm = colors.Normalize(**norm_prop)
    else:
        norm = colors.TwoSlopeNorm(**norm_prop)
       
    norm_map = norm(nes)
    cmap = plt.cm.get_cmap(pcm_prop.get('cmap'))
    color_rgbas = cmap(norm_map)
        
        # Add color for NES text
    stats['color'] = [_color_light_or_dark(rgba) for rgba in color_rgbas]
    
    return norm, pcm_prop  