import scanpy as sc
import pandas as pd
import numpy as np
import anndata

def export_files(state):
    adata = anndata.read_h5ad(state+'.h5ad')

    # find cell with lowest E-cad as root
    Ecad, names = adata.X[7], list(adata.obs_names)
    adata.uns['iroot'] = np.argmin(Ecad)

    sc.pp.neighbors(adata, n_neighbors=4, n_pcs=20)
    sc.tl.diffmap(adata)
    sc.tl.dpt(adata)

    pst = np.asarray(adata.obs['dpt_pseudotime'])
    df_pst = pd.DataFrame(data=pst, index=list(adata.obs_names), columns=['PseudoTime'])
    df_pst.to_csv('PseudoTime_'+state+'.csv')

    df_count = pd.DataFrame(data=adata.X.transpose(), index=list(adata.var_names), columns=list(adata.obs_names))
    df_count.to_csv('ExpressionData_'+state+'.csv')


export_files('epi')
export_files('hyb')
export_files('mes')