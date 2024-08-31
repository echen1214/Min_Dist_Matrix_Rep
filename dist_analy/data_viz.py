# visualization module
import warnings
from ipywidgets import interact,fixed,IntSlider
import altair as alt
import py3Dmol
import pandas as pd

def py3Dmol_onerec_multilig_pose(rec_fn:str, rec_chain:str, lig_fn_list: list, \
    lig_chain:str =" ", rec_style:list = None, lig_style:dict =None):
    """ 
    Parameters
    ----------
    rec_fn_list : list of str
        list of filenames corresponding to the receptor protein
    rec_chain_list : list of str
        list of chains that will dictate which chain will be colored by the rec_style
    lig_fn_list : list of str, optional
        list of ligand filenames
    rec_style : list of dict, optional
        list of dictionaries that change the receptor_chain style
    lig_style : dict, optional
        list of dictionaries that change the ligand style
    """    

    def view_setFrame(index):
        view.setFrame(index).zoomTo().update()
        return

    def add_model(models, i, rec_fn, lig_fn=None):
        models += "MODEL " + str(i) + "\n"
        if rec_fn:
            with open(rec_fn) as f1:
                for x in f1:
                    if x[:6]=="ATOM  " or x[:6]=="HETATM":
                        models += x
        if lig_fn:
            with open(lig_fn) as f1:
                for x in f1:
                    if x[:6]=="ATOM  " or x[:6]=="HETATM":
                        models += f"{x[:17]}LIG{x[20:]}"
                    else:
                        models += x
        models += "ENDMDL\n"        
        return models    

    if rec_style is None:
        rec_style = {"cartoon": {"style": "edged", 'color': 'red'}}
    if lig_style is None:
        lig_style = {"stick": {'color': 'blue'}}
        
    view = py3Dmol.view(width=400, height=300)

    models = ""
    for i, lig in enumerate(lig_fn_list):
        models = add_model(models, i, rec_fn, lig)
    # else:
    #     for i, rec in enumerate(rec_fn_list):
    #         models = add_model(models, i, rec)       
    # print(models)
    view.addModelsAsFrames(models)
    for i, _ in enumerate(lig_fn_list):
        view.setStyle({'chain':rec_chain, 'frame':i}, rec_style)

    view.setStyle({'resn':'LIG'}, lig_style)

    view.zoomTo()
    view.show()
    interact(view_setFrame, index=IntSlider(min=0,max=len(lig_fn_list)-1, step=1))

def py3Dmol_lig_pose(rec_fn_list:list, rec_chain_list:list, lig_fn_list: list=[], \
    lig_chain:str =" ", rec_style:list = None, lig_style:dict =None):
    """ Pass in a three lists of receptors, the corresponding chain, and ligand file
    and visualize them in the py3Dmol which embeds a 3Dmol.js view into the jupyter notebook
    this also adds a slider that iterates through the list

    py3Dmol tutorials and examples
    [1](https://nbviewer.org/github/3dmol/3Dmol.js/blob/master/py3Dmol/examples.ipynb)
    [2](https://www.insilicochemistry.io/tutorials/foundations/chemistry-visualization-with-py3dmol)
    [3](https://3dmol.csb.pitt.edu/doc/index.html)

    TODO:
    - passing into many files slows down the view heavily
    - it would also be valuable to provide an option to align the structures before hand
    - is there a clever way we can handle one receptor file and multiple ligand files (virtual screening task)a
        - same with the handling of the receptor/ligand styles
    - worth to make a more universal function/class to handle all types of data inputs

    Parameters
    ----------
    rec_fn_list : list of str
        list of filenames corresponding to the receptor protein
    rec_chain_list : list of str
        list of chains that will dictate which chain will be colored by the rec_style
    lig_fn_list : list of str, optional
        list of ligand filenames
    rec_style : list of dict, optional
        list of dictionaries that change the receptor_chain style
    lig_style : dict, optional
        list of dictionaries that change the ligand style
    """    
    def view_setFrame(index):
        view.setFrame(index).zoomTo().update()
        return

    def add_model(models, i, rec_fn, lig_fn=None):
        models += "MODEL " + str(i) + "\n"
        if rec_fn:
            with open(rec_fn) as f1:
                for x in f1:
                    if x[:6]=="ATOM  " or x[:6]=="HETATM":
                        models += x
        if lig_fn:
            with open(lig_fn) as f1:
                for x in f1:
                    if x[:6]=="ATOM  " or x[:6]=="HETATM":
                        models += f"{x[:17]}LIG{x[20:]}"
                    else:
                        models += x
        models += "ENDMDL\n"        
        return models    

    if rec_style is None:
        rec_style = {"cartoon": {"style": "edged", 'color': 'red'}}
    if lig_style is None:
        lig_style = {"stick": {'color': 'blue'}}
        
    view = py3Dmol.view(width=400, height=300)

    models = ""
    if lig_fn_list:
        for i, (rec, lig) in enumerate(zip(rec_fn_list, lig_fn_list)):
            models = add_model(models, i, rec, lig)
    else:
        for i, rec in enumerate(rec_fn_list):
            models = add_model(models, i, rec)       
    # print(models)
    view.addModelsAsFrames(models)
    for i, chain in enumerate(rec_chain_list):
        view.setStyle({'chain':chain, 'frame':i}, rec_style)

    view.setStyle({'resn':'LIG'}, lig_style)

    view.zoomTo()
    view.show()
    interact(view_setFrame, index=IntSlider(min=0,max=len(rec_chain_list)-1, step=1))

def alt_pca_hdbscan_figure(df_proj:pd.DataFrame() , table_dict:dict =dict()):
    """
    Altair tool that plots a PCA plot and allowing outputting the selection

    Parameters
    ----------
    df_proj : pd.DataFrame
        input dataframe with columns named pca1 and pca2
    table_dict : dict, optional
        dictionary filtering the which data in the dataframe should be output
        {dataframe.column = output title}
    Returns
    -------
    alt.JupyterChart
        class that can access the selection outputs on the chart
        [1] https://altair-viz.github.io/user_guide/jupyter_chart.html
    """
    brush = alt.selection_interval(name="interval")
    
    scatter = alt.Chart(df_proj).mark_point().encode(
        x=alt.X('pca1:Q',scale=alt.Scale(zero=False)), 
        y=alt.Y('pca2:Q',scale=alt.Scale(zero=False)), 
        color=alt.condition(brush, 'label:N', alt.value('grey')),
    ).add_params(brush)

    ranked_text = alt.Chart(df_proj).mark_text(align='right').encode(
        y=alt.Y('row_number:O',axis=None)
    ).transform_filter(
        brush
    ).transform_window(
        row_number='row_number()'
    ).transform_filter(
        'datum.row_number < 15'
    )

    alt_text = []
    for col, title in table_dict.items():
        _ = ranked_text.encode(text=col).properties(title=alt.TitleParams(text=title, align='right'))
        alt_text.append(_)
    text = alt.hconcat(*alt_text) # Combine data tables

    # Build chart
    chart = alt.hconcat(
        scatter,
        text
    ).resolve_legend(
        color="independent"
    ).configure_view(strokeWidth=0)

    jchart = alt.JupyterChart(chart)
    return(jchart)

def jchart_query(df_proj:pd.DataFrame(), jchart: alt.JupyterChart, return_col: list):
    """
    alt.JupyterChart helpter function that prints out the selection

    Parameters
    ----------
    df_proj : pd.DataFrame
        input dataframe
    jchart : alt.JupyterChart
        class that can access the state of interactive charts
    return_col : list of string
        list of column names to be output

    Returns
    -------
    _type_
        _description_
    """
    filter = " and ".join([
        f"{v[0]} <= `{k}` <= {v[1]}"
        for k, v in jchart.selections.interval.value.items()
    ])
    df_q = df_proj.query(filter)
    # print(df_q)
    if return_col:
        return df_q[return_col]