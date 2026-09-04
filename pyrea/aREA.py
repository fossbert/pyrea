import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Union, Tuple


def aREA(dset: Union[pd.DataFrame, pd.Series], 
         regulon: pd.DataFrame, 
         minsize: int = 20,
         dset_filter: bool = False,
         )->pd.DataFrame:
    
    """[This function takes an , respectively, and
    computes analytic rank-based enrichment analysis as described in Alvarez et al., Nat
    Genetics, 2016]

    Parameters
    ----------
    dset: Union[pd.DataFrame, pd.Series] :
        
    regulon: pd.DataFrame :
        
    minsize: int :
         (Default value = 20)

    Returns
    -------

    """
    
    # Pre-processing
    if dset_filter:
      dset = filter_dset(dset, regulon)
    
    regulon_filtered = filter_regulon(dset, regulon, minsize=minsize)
    
    # transform dset to rank matrices
    t1, t2 = get_tmats(dset)

    # get weight matrices from MoR and Likelihood
    mor, wts, wtss = get_mor_wts(regulon_filtered)

    # calculate three-tailed enrichment score
    s1 = calc_2TES(t2, mor, wtss) #2Tailed ES
    s2 = calc_1TES(t1, mor, wtss) #1tailed ES
    s3 = calc_3TES(s1, s2)

    # Normalize
    nes = calc_nes(s3, wts)

    return nes


def calc_nes(s3:pd.DataFrame, 
             wts:pd.DataFrame)->pd.DataFrame:
    """This function normalizes the three-tailed enrichment score using the weights from the interaction confidence

    Parameters
    ----------
    s3:pd.DataFrame :
        
    lik:pd.DataFrame :
        

    Returns
    -------

    """

    # likelihood weights
    lwts = np.sqrt(np.sum(wts.values**2, axis=0))[:,np.newaxis]

    nes = s3.values * lwts
    
    nes_df = pd.DataFrame(nes, index=s3.index, columns=s3.columns)

    return nes_df


def calc_3TES(s1:pd.DataFrame, 
              s2:pd.DataFrame)->pd.DataFrame:

    """This function calculates the three-tailed enrichment score from two enrichment score (ES) DataFrames.
        It will expect the two-tail ES as s1 and one-tail ES as s2.

    Parameters
    ----------
    s1:pd.DataFrame :
        
    s2:pd.DataFrame :
        

    Returns
    -------

    """

    # Extract the signs of the two-tail enrichment scores
    s1_signs = s1.copy()
    s1_signs[s1_signs>=0] = 1
    s1_signs[s1_signs<0] = -1

    s2_signs = s2.copy()
    s2_signs = (s2_signs.values > 0).astype(int)
    
    # Create final ES
    s3 = (np.abs(s1.values) + s2.values * s2_signs) * s1_signs.values
    
    s3_df = pd.DataFrame(s3, index=s1.index, columns=s1.columns)
    return s3_df



def calc_1TES(t1:pd.DataFrame, 
              mor:pd.DataFrame, 
              wtss:pd.DataFrame)->pd.DataFrame:

    """[This function calculates the one-tail enrichment score DataFrame]

    Parameters
    ----------
    t1:pd.DataFrame :
        
    mor:pd.DataFrame :
        
    lik_scaled:pd.DataFrame :
        

    Returns
    -------

    """
    # Weight DataFrame irrespective of direction
    wm1 = (1 - np.abs(mor.values)) * wtss.values
    # Align matrices
    # pos = t1.index.intersection(wm1.index)
    # t1 = t1.loc[pos]
    # wm1 = wm1.loc[pos]

    # Multiply
    s2 = wm1.T @ t1.loc[mor.index].values
    
    s2_df = pd.DataFrame(s2, index=mor.columns, columns=t1.columns)
    return s2_df


def calc_2TES(t2:pd.DataFrame, 
              mor: pd.DataFrame, 
              wtss: pd.DataFrame)->pd.DataFrame:

    
    """This function calculates the two-tail enrichment score DataFrame

    Parameters
    ----------
    t2:pd.DataFrame :
        
    mor: pd.DataFrame :
        
    lik_scaled: pd.DataFrame :
        

    Returns
    -------

    """
    # Create two-tailed weighted MoR DF
    wm2 = mor.values * wtss.values

    # Align DFs
    # pos = t2.index.intersection(wm2.index)
    # t2 = t2.loc[pos]
    # wm2 = wm2.loc[pos]

    # Multiply
    s1 = wm2.T @ t2.loc[mor.index].values
    
    s1_df = pd.DataFrame(s1, index=mor.columns, columns=t2.columns)
    return s1_df


def get_mor_wts(regulon: pd.DataFrame)-> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """

    Parameters
    ----------
    regulon: pd.DataFrame :
        
    minsize: int :
        

    Returns
    -------

    """
 
    if not isinstance(regulon, pd.DataFrame):
      raise ValueError('Need a pandas DataFrame as a regulon')
    if len(regulon.columns)!=4:
      raise ValueError('Regulon DataFrame needs 4 columns: source, target, mor, likelihood')
    
    if not all(regulon.columns.values == np.array(['source', 'target', 'mor', 'likelihood'])):
      regulon.columns = ['source', 'target', 'mor', 'likelihood']

    # Wrangling
    df = regulon.copy()
    df.set_index(['source', 'target'], inplace=True)
    mor = df['mor'].unstack(level=0)
    wts = df['likelihood'].unstack(level=0)
   
    # Implement filter step
    mor.fillna(0, inplace=True)
    
    wts.fillna(0, inplace=True)  
    # Scale likelihood values
    wts = wts / np.max(wts.values, axis=0, keepdims=True)
    
    # Scale likelihood DF for column sums
    wtss = wts / np.sum(wts.values, axis=0, keepdims=True)

    return mor, wts, wtss

def get_tmats(dset: Union[pd.DataFrame, pd.Series])-> Tuple[pd.DataFrame, pd.DataFrame]:

    """From a single gene expression Series or gene expression DataFrame, this function calculates
    T1 and T2 numeric DataFrames, i.e. the scaled expression rank of every gene within the sample. The ranked values are
    transformed to quantiles from the normal distribution.

    Parameters
    ----------
    dset: Union[pd.DataFrame :
        
    pd.Series] :
        

    Returns
    -------

    """
    assert isinstance(dset, pd.Series) or isinstance(dset, pd.DataFrame), 'Wrong data input'

    # rank computation
    t2 = dset.rank(method='first') / (len(dset)+1)
    t1 = np.abs(t2 - 0.5) * 2
    t1 = t1 + (1 - np.max(t1, axis=0))/2

    # transform to quantiles of a normal distribution
    if isinstance(dset, pd.Series):
        t2q = pd.DataFrame(norm.ppf(t2.values),
                  index=t2.index,
                  columns=[t2.name])

        t1q = pd.DataFrame(norm.ppf(t1.values),
                  index=t1.index,
                  columns=[t1.name])

    # No further checking necessary, this is checked during Class instantiation
    else:
        t2q = pd.DataFrame(norm.ppf(t2.values),
                  index=t2.index,
                  columns=t2.columns)

        t1q = pd.DataFrame(norm.ppf(t1.values),
                  index=t1.index,
                  columns=t1.columns)

    return t1q, t2q




def filter_dset(dset: Union[pd.DataFrame, pd.Series], 
                            regulon: pd.DataFrame):
  
  mask = dset.index.isin(regulon['target'])
  
  try:
    dset_filtered = dset.loc[mask].copy()
    return dset_filtered
  
  except:
    raise ValueError("Didn't find any suitable targets in the expression signature(s)!")
          
def filter_regulon(dset: Union[pd.DataFrame, pd.Series], 
                            regulon: pd.DataFrame, 
                            minsize: int):
  
  mask = regulon['target'].isin(dset.index)
   
  try:
    reg_filtered = regulon[mask].copy()
    cts = reg_filtered.groupby('source')['target'].count() 
    keep_rps = cts[cts>=minsize].index
    regulon_filtered_pruned = reg_filtered[reg_filtered['source'].isin(keep_rps)]
    return regulon_filtered_pruned
  except:
    raise ValueError("Fialed at filtering and pruning the regulon")
  
  