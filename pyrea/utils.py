import numpy as np
import pandas as pd
from typing import Union

# file system and import
import os
import glob
import json



"""Helper functions used throughout the package"""

def gene_sets_to_regulon(genesets: dict, 
                         minsize: int = 20)-> pd.DataFrame:

    """This function generates a regulon dictionary suitable for aREA from a 'regulary' dictionary
    of pathway names as keys and a list of the related gene symbols as values.

    Parameters
    ----------
    genesets :
        dict: Dictionary of pathway names (=keys) and lists of pathway member genes (=values).
    minsize :
        int: Integer specifying the minimum number of genes or targets a pathway or regulator should have (Default value = 20).
    maxsize :
        int: Integer specifying the maximum number of genes or targets a pathway or regulator should have (Default value = None).
    genesets: dict :
        
    minsize: int :
         (Default value = 20)

    Returns
    -------
    dictionary
        A dictionary of DataFrames with information on pathway/regulator targets, mode of regulation (always=1) and likelihoods (always=1/N targets).

    """

    reg = {}

    for key, val in genesets.items():
      # duplicate values will mess things up ! 
        set_in = set(val)
        ns = len(val)
        
        if ns>= minsize:
          reg[key] = _gene_set_to_df(set_in, ns)
     
    df = pd.concat(reg)
    df.index.set_names('source', level=0, inplace=True)
    df.reset_index(level=0, inplace=True)
    
    return df
  
def _gene_set_to_df(gene_set: set, ns: int, negative: bool=False):
  
  mor = np.ones(ns)

  if negative:

    mor = mor*-1
   
  return pd.DataFrame(data=zip(gene_set, mor, np.abs(mor/ns)),
                      columns=['target', 'mor', 'likelihood'])


def _prep_ges(ges: Union[pd.Series,pd.DataFrame], 
              asc_sort: bool=True) -> pd.Series:

    """[This function is used to clean up signatures containing NA's in their index or merging
    scores values for duplicated genes by averaging them.]

    Parameters
    ----------
    ges :
        pd.Series or pd.DataFrame:
    asc_sort :
        bool:  (Default value = True)
    ges: Union[pd.Series :
        
    pd.DataFrame] :
        
    asc_sort: bool :
         (Default value = True)

    Returns
    -------
    pd.Series|pd.DataFrame
        cleaned gene expression signature(s)

    """
    
    assert (isinstance(ges, pd.Series) or isinstance(ges, pd.DataFrame)), 'Need pandas Series or DataFrame!'
        
    ges_out = ges.copy()
    if isinstance(ges, pd.Series):
      
      if ges_out.name is None:
        ges_out.name = 'ges'
      if ges_out.isnull().any():
        ges_out.dropna(inplace=True)    
        
      if ges_out.index.duplicated().any():
        ges_out = ges_out.groupby(level=0).mean() # collapse by averaging
      
      ges_out.sort_values(ascending=asc_sort, inplace=True)
      
    else:
      if ges_out.isnull().any().any():
        ges_out.dropna(inplace=True)
          
      if ges_out.index.duplicated().any(): 
        ges_out = ges.groupby(level=0).mean()
                 
    return ges_out


def sig_to_reg(ges: Union[pd.Series,pd.DataFrame], 
              direction: str='both',
              nn:int = 50,
              reverse: bool = False):
  

    """[This function expects a numeric signature and will build a regulon from the top nn genes (rows)]

    Parameters
    ----------
    ges :
        pd.Series or pd.DataFrame
    direction :
        str:  (Default value = 'both'). Further options include 'up' and 'down'
    nn :
        int:  (Default value = 50)
    reverse :
        bool:  (Default value = False)

    Returns
    -------
    pd.DataFrame
        A regulon representing the most up|downregulated genes for the signature(s).
    """

    ges_prepped = _prep_ges(ges)

    dir_options = ['both', 'up', 'down']

    if direction not in dir_options:

      raise ValueError(f"{direction} is not a valid option, please choose from: {', '.join(dir_options)}")
    
    if len(ges_prepped.shape)<2: # For 1-dimensional signatures, convert to 2 dimensions

      ges_prepped = ges_prepped.to_frame()

    reg_dict = {k:_get_ges_tops(v, direction, nn) for k, v in ges_prepped.items()}

    d_reg = pd.concat(reg_dict)
    d_reg.index.set_names('source', level=0, inplace=True)
    d_reg.reset_index(level=0, inplace=True)

    if reverse:

      d_reg.loc[:,'mor'] = d_reg.loc[:,'mor'] * -1

    return d_reg
    

def _get_ges_tops(ges:pd.Series, direction:str, nn:int):
  
  """Helper function, returs top nn index labels based on the numeric results of the signature.
  """

  if direction == 'both':

    genes_down =  ges.nsmallest(nn).index.to_list()
    genes_up = ges.nlargest(nn).index.to_list()

    d_down = _gene_set_to_df(genes_down, nn, negative=True)

    d_up = _gene_set_to_df(genes_up, nn)

    d_out = pd.concat([d_up, d_down], axis=0)

    d_out.loc[:,'likelihood'] =  d_out['likelihood'] * 0.5
    
    return  d_out
  
  elif direction == 'up':

    genes =  ges.nlargest(nn).index.to_list()

    return _gene_set_to_df(genes, nn)
  
  elif direction == 'down':

    genes =  ges.nsmallest(nn).index.to_list()

    return _gene_set_to_df(genes, nn, negative=True)


    

    
  
  
def load_genesets(collection: str = 'h', 
                  species: str = 'human'):
  
  DIR = os.path.dirname(os.path.realpath(__file__))
  
  coptions = ['h', 'pid', 'wikipathways', 'reactome']
    
  if collection.lower() not in coptions:
    raise ValueError(f"{collection} is not a suitable option. Please choose from {', '.join(coptions)}.")
  
  soptions =  ['human', 'mouse']
  
  if species.lower() not in soptions:
    raise ValueError(f"{species} is not a suitable option. Please choose from {', '.join(soptions)}.")
  
  if collection == 'h':
    gpattern = f"{collection}*{species}*"
  else:
    gpattern = f"*{collection}*{species}*"
  
  fps = glob.glob(os.path.join(DIR, 'data', gpattern))
  print(f"Found {len(fps)} gene set collection(s)!")
  
  fp = fps[0]  
  bn = os.path.basename(fp)
    
  try:
    with open(fp) as infile:
      genesets_in = json.load(infile)
    
    print(f"Loaded {bn} into a dictionary for you.")
    return genesets_in
  except:
    raise ImportError(f'Something went wrong when trying to import {fp}')
  


def load_species_converter(keys="human"):
    
    koptions =  ['human', 'mouse']

    if keys.lower() not in koptions:
      raise ValueError(f"{keys} is not a suitable option. Please choose from {', '.join(koptions)}.")
  
  
    DIR = os.path.dirname(os.path.realpath(__file__))

    fp = os.path.join(DIR, 'data', 'Human_Mouse_Homology.json')
    bn = os.path.basename(fp)
      
    try:
      with open(fp) as infile:
        genesets_in = json.load(infile)
        
        if keys == 'mouse':
          genesets_in = {v:k for k, v in genesets_in.items()}
      
      print(f"Loaded {bn} into a dictionary for you.")
      return genesets_in
    except:
      raise ImportError(f'Something went wrong when trying to import {fp}')
        
      

  
  
      