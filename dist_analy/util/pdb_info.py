"""Utility functions for requesting URLs over HTTP"""
"""Code adapted from williamgilpin at https://github.com/williamgilpin/pypdb
   under the MIT License
"""

from typing import Optional
from datetime import datetime
import json
import time
import requests
import warnings
import csv
import ast
import pandas as pd


def pdb_read_csv(csv_file: str, chain: bool = False):
    """Read csv generated by pdb_chain_csv or pdb_csv

    Parameters
    ----------
    csv_file : str
        Input csv file
    chain : bool
        True or false if the dictionary has a chain column

    Returns
    -------
    dict
        dict of the csv file
    pandas.DataFrame
        DataFrame of the csv file

    """

    if chain:
        dict_items = ['PDB ID', 'Chain', 'Title', 'Date', 'Description', 'Method', 'Resolution', \
            'Polymer Binders', 'Small Polymer Binders', 'Major Binders', 'Minor Binders', \
            'Modifications', 'Mutations', "UniProt ID", 'Paper DOI',]
    else:
        dict_items = ['PDB ID', 'Title', 'Date', 'Description', 'Method', 'Resolution', \
            'Polymer Binders', 'Major Binders', 'Minor Binders', \
            'Modifications', 'Mutations', 'Paper DOI']
    df = pd.read_csv(csv_file, header=0, usecols=dict_items)
    for col in ['Polymer Binders', 'Major Binders', 'Minor Binders','Modifications', 'Mutations',]:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if type(x)==str else [])
    return(df.set_index('PDB ID').T.to_dict(), df)

def pdb_chain_csv(pdb_list: list, uniprot: str, csv_file: str):
    """ Creates a csv files of information parsed from the rcsb database about
    the pdb structures. Includes information about the resolution, binders, and
    modifications.

    Parameters
    ----------
    pdb_list : list
        list of PDB files to get information about
    uniprot : str
        UniProt accession ID
    csv_file : str
        output filename

    """
    info_root = 'https://data.rcsb.org/rest/v1/core/entry/'
    uniprot_root_url = 'https://data.rcsb.org/rest/v1/core/uniprot/'
    info_root_1 = 'https://data.rcsb.org/rest/v1/core/polymer_entity/'
    info_root_2 = 'https://data.rcsb.org/rest/v1/core/nonpolymer_entity/'

    # csv_data = []
    dict_items = ['PDB ID', 'Chain', 'Title', 'Date', 'Description', 'Method', 'Resolution', \
                  'Polymer Binders', 'Small Polymer Binders', 'Major Binders', 'Minor Binders', \
                  'Modifications', 'Mutations', 'Paper DOI', 'UniProt ID']
    pdb_dict = {}
    for pdb in pdb_list:
        try:
            data = get_any_info(info_root,pdb)
            polymer_entry_all = [entry_id for entry_id in data['rcsb_entry_container_identifiers']['polymer_entity_ids']]
                # try:
                #     uniprot_query = get_any_info(uniprot_root_url, pdb, str(entry_id))[0]\
                #         ['rcsb_uniprot_container_identifiers']['uniprot_id']
                #     if uniprot_query == uniprot:
                #         entry_all.append(int(entry_id))
                # except ValueError:
                #     print("unable to find data with %s entity #%s, skipping"%(pdb,entry_id))
            chain_info_all = [get_any_info(info_root_1, pdb, entry) for entry in polymer_entry_all]
            num_poly_entity = [int(i)-1 for i in data["rcsb_entry_container_identifiers"]["polymer_entity_ids"]]

            for i, (entry, chain_info) in enumerate(zip(polymer_entry_all, chain_info_all)):
                chain_all = []
                # try:
                #     chain_info = get_any_info(info_root_1, pdb, entry)
                # except ValueError:
                #     print("unable to find data with %s entry #%s, skipping"%(pdb,entry))

                chain_all.append(chain_info['entity_poly']['pdbx_strand_id'].split(','))
                chain_all = [item for sublist in chain_all for item in sublist]
                curr_desc = chain_info_all[i]["rcsb_polymer_entity"]["pdbx_description"]

                for chain in chain_all:
                    temp_pdb = pdb + "_" + chain
                    pdb_dict[temp_pdb] = {}

                    pdb_dict[temp_pdb]['PDB ID'] = pdb
                    pdb_dict[temp_pdb]['Chain'] = chain
                    pdb_dict[temp_pdb]["Title"] = data['struct']['title']
                    try:
                        pdb_dict[temp_pdb]["Description"] = chain_info["rcsb_polymer_entity"]["pdbx_description"]
                    except KeyError:
                        pdb_dict[temp_pdb]["Description"] = None

                    pdb_dict[temp_pdb]["Method"] = data["rcsb_entry_info"]["experimental_method"]
                    date_obj = datetime.strptime(data["rcsb_accession_info"]["deposit_date"], "%Y-%m-%dT%H:%M:%S%z")
                    pdb_dict[temp_pdb]["Date"] = date_obj.strftime("%Y-%m-%d")
                    try:
                        pdb_dict[temp_pdb]["Resolution"] = data["rcsb_entry_info"]["resolution_combined"][0]
                    except:
                        pdb_dict[temp_pdb]["Resolution"] = None

                    try:
                        pdb_dict[temp_pdb]['Modifications'] = chain_info["rcsb_polymer_entity_container_identifiers"]["chem_comp_nstd_monomers"]
                    except KeyError:
                        if "Modifications" in pdb_dict[temp_pdb]: pass
                        else:
                            pdb_dict[temp_pdb]['Modifications'] = None
                    try:
                        pdb_dict[temp_pdb]['Mutations'] = chain_info["rcsb_polymer_entity"]['pdbx_mutation'].split(",")
                    except KeyError:
                        if "Mutations" in pdb_dict[temp_pdb]: pass
                        else:
                            pdb_dict[temp_pdb]['Mutations'] = None

                    pdb_dict[temp_pdb]["Major Binders"] = []
                    pdb_dict[temp_pdb]["Minor Binders"] = []
                    pdb_dict[temp_pdb]["Polymer Binders"] = []
                    pdb_dict[temp_pdb]["Small Polymer Binders"] = []

                    for j in num_poly_entity:
                        partner_desc = chain_info_all[j]["rcsb_polymer_entity"]["pdbx_description"]
                        if (i != j) and (curr_desc != partner_desc):
                            try:
                                pdb_dict[temp_pdb]["Polymer Binders"].append(partner_desc)
                                if float(chain_info_all[j]["rcsb_polymer_entity"]["formula_weight"]) < 2.0:
                                    pdb_dict[temp_pdb]["Small Polymer Binders"].append(partner_desc)
                            except KeyError:
                                pdb_dict[temp_pdb]["Polymer Binders"].append(None)

                    try:
                        num_nonpoly_entity =  data["rcsb_entry_container_identifiers"]["non_polymer_entity_ids"]
                        for i in num_nonpoly_entity:
                            data_1 = get_any_info(info_root_2,pdb,i)
                            chain_list = data_1["rcsb_nonpolymer_entity_container_identifiers"]["auth_asym_ids"]
                            if float(data_1["rcsb_nonpolymer_entity"]["formula_weight"]) > 0.1: # units kDa
                                pdb_dict[temp_pdb]["Major Binders"].append(data_1["pdbx_entity_nonpoly"]["comp_id"])
                            else:
                                pdb_dict[temp_pdb]["Minor Binders"].append(data_1["pdbx_entity_nonpoly"]["comp_id"])
                    except KeyError:
                        print("unable to get binders of %s_%s...the structure may be apo or report their binders as a polymers"%(pdb,chain))

                    try:
                        pdb_dict[temp_pdb]["Paper DOI"] = data["citation"][0]["pdbx_database_id_doi"]
                    except:
                        pdb_dict[temp_pdb]["Paper DOI"] = None

                    try:
                        pdb_dict[temp_pdb]["UniProt ID"] = chain_info['rcsb_polymer_entity_container_identifiers']['uniprot_ids']
                    except:
                        pdb_dict[temp_pdb]["UniProt ID"] = None

        except Exception as e:
            print("%s unable to get info for %s"%(e, pdb))

    with open(csv_file, 'w') as f1:
        writer = csv.DictWriter(f1, fieldnames = dict_items)
        writer.writeheader()
        for pdb, values in pdb_dict.items():
            writer.writerow(values)
    return(pd.DataFrame.from_dict(pdb_dict.values()), pdb_dict)


def pdb_csv(pdb_list: list, uniprot: str, csv_file: str):
    """ Creates a csv files of information parsed from the rcsb database about
    the pdb structures. Includes information about the resolution, binders, and
    modifications.

    Parameters
    ----------
    pdb_list : list
        list of PDB files to get information about
    uniprot : str
        UniProt accession ID
    csv_file : str
        output filename

    """
    info_root = 'https://data.rcsb.org/rest/v1/core/entry/'
    info_root_1 = 'https://data.rcsb.org/rest/v1/core/polymer_entity/'
    info_root_2 = 'https://data.rcsb.org/rest/v1/core/nonpolymer_entity/'

    # csv_data = []
    dict_items = ['PDB ID', 'Title', 'Date', 'Description', 'Method', 'Resolution', \
                  'Polymer Binders', 'Major Binders', 'Minor Binders', \
                  'Modifications', 'Mutations', 'Paper DOI']
    pdb_dict = {}
    for pdb in pdb_list:
        # print(pdb)
        pdb_dict[pdb] = {}

        data = get_any_info(info_root,pdb)
        pdb_dict[pdb]['PDB ID'] = pdb
        pdb_dict[pdb]["Title"] = data['struct']['title']
        try:
            pdb_dict[pdb]["Description"] = data['struct']['pdbx_descriptor']
        except KeyError:
            pdb_dict[pdb]["Description"] = None
        date_obj = datetime.strptime(data["rcsb_accession_info"]["deposit_date"], "%Y-%m-%dT%H:%M:%S%z")
        pdb_dict[pdb]["Date"] = date_obj.strftime("%Y-%m-%d")
        pdb_dict[pdb]["Method"] = data["rcsb_entry_info"]["experimental_method"]
        try:
            pdb_dict[pdb]["Resolution"] = data["rcsb_entry_info"]["diffrn_resolution_high"]["value"]
        except:
            pdb_dict[pdb]["Resolution"] = None
        # try:
        #     pdb_dict[pdb]["Minor Binders"] = data['rcsb_entry_info']["nonpolymer_bound_components"]
        # except KeyError:
        #     pdb_dict[pdb]["Minor Binders"] = None
        major_binders = []
        minor_binders = []
        try:
            num_nonpoly_entity =  data["rcsb_entry_container_identifiers"]["non_polymer_entity_ids"]
            for i in num_nonpoly_entity:
                data_1 = get_any_info(info_root_2,pdb,i)
                if float(data_1["rcsb_nonpolymer_entity"]["formula_weight"]) > 0.1: # units kDa
                    major_binders.append(data_1["pdbx_entity_nonpoly"]["comp_id"])
                else:
                    minor_binders.append(data_1["pdbx_entity_nonpoly"]["comp_id"])
            # for binding_dict in data["rcsb_binding_affinity"]:
            #     if binding_dict["comp_id"] not in major_binders:
            #         major_binders.append(binding_dict["comp_id"])
            # pdb_dict[pdb]["Major Binders"] = major_binders
        except KeyError:
            pdb_dict[pdb]["Major Binders"] = None

        pdb_dict[pdb]["Major Binders"] = major_binders
        pdb_dict[pdb]["Minor Binders"] = minor_binders
        polymer_binders = []
        mutations = []

        num_poly_entity = data["rcsb_entry_container_identifiers"]["polymer_entity_ids"]
        for i in num_poly_entity:
            data_1 = get_any_info(info_root_1,pdb,i)
            try:
                for x in data_1["rcsb_polymer_entity_container_identifiers"]["uniprot_ids"]:
                    if x == uniprot:
                        try:
                            pdb_dict[pdb]['Modifications'] = data_1["rcsb_polymer_entity_container_identifiers"]["chem_comp_nstd_monomers"]
                        except KeyError:
                            pdb_dict[pdb]['Modifications'] = None
                    else:
                        polymer_binders.append(data_1["rcsb_polymer_entity"]["pdbx_description"])
            except KeyError:
                polymer_binders.append(data_1["rcsb_polymer_entity"]["pdbx_description"])
            try:
                pdb_dict[pdb]['Mutations'] = data_1["rcsb_polymer_entity"]['pdbx_mutation'].split(",")
            except KeyError:
                pdb_dict[pdb]['Mutations'] = None
        pdb_dict[pdb]["Polymer Binders"] = polymer_binders
        try:
            pdb_dict[pdb]["Paper DOI"] =  data["citation"][0]["pdbx_database_id_doi"]
        except:
            pdb_dict[pdb]["Paper DOI"] = None
        # csv_data.append(pdb_dict)
    with open(csv_file, 'w') as f1:
        writer = csv.DictWriter(f1, fieldnames = dict_items)
        writer.writeheader()
        for pdb, values in pdb_dict.items():
            # print(values)
            writer.writerow(values)
    return(pd.DataFrame.from_dict(pdb_dict.values()), pdb_dict)
    # return(pd.DataFrame.from_dict(pdb_dict))

def get_any_info(url_root: str, *url_append, **request_dict):
    """ Code adapted from williamgilpin at https://github.com/williamgilpin/pypdb
    under the MIT License

    PDB information url_root: ''
    Uniprot sequence url_root: 'http://www.ebi.ac.uk/proteins/api/proteins/'
    Uniprot protein url_root: 'https://data.rcsb.org/rest/v1/core/uniprot/'
    SIFTS url_root: 'https://www.ebi.ac.uk/pdbe/api/mappings/'
    KLIFs url_root: 'https://klifs.net/api/kinase_names'
    can maybe move to utils

    Reference the appropriate documentation to see the available parameters
    KLIFs info: https://klifs.net/swagger/
    RCSB info: https://data.rcsb.org/redoc/index.html
    PDBe info: https://www.ebi.ac.uk/pdbe/api/doc/

    can maybe make a constants class that has all of the requisite schema
    eg: https://data.rcsb.org/#data-schema

    Parameters
    ----------
    url_root : str
        base url
    *url_append : str
        any strings to append to the end of the url request
    **request_dict : dict
        any dictionary that will be converted to parameters to append to the
        end of the url
    Returns
    -------
    dict
        An ordered dictionary object corresponding to entry information


    Examples
    --------
    get_any_info('4eoq', 'https://data.rcsb.org/rest/v1/core/uniprot/', '1')
    would request the uniprot information from entity 1 of 4eoq at the url:
    https://data.rcsb.org/rest/v1/core/uniprot/4eoq/1

    """

    """ does this work with inputting a json dictionary?"""

    # if args is a list:
    if url_append and request_dict:
        raise ValueError("cannot have url_append and request_dict together")

    if url_append:
        for app in url_append:
            url_root += str(app) +'/'
            # print(url_root)
        response = request_limited(url_root)
    elif request_dict:
        response = request_limited(url_root, params=request_dict)
    else:
        response = request_limited(url_root)
    # print(response.url)
    if response and response.status_code == 200:
        pass
    else:
        raise ValueError("json retrieval failed, returning None")
        return None

    result  = str(response.text)
    out = json.loads(result)

    return out


def request_limited(url: str, rtype: str="GET",
                    num_attempts: int = 3, sleep_time=0.5, **kwargs
                    ) -> Optional[requests.models.Response]:
    """
    HTML request with rate-limiting base on response code


    Parameters
    ----------
    url : str
        The url for the request
    rtype : str
        The request type (oneof ["GET", "POST"])
    num_attempts : int
        In case of a failed retrieval, the number of attempts to try again
    sleep_time : int
        The amount of time to wait between requests, in case of
        API rate limits
    **kwargs : dict
        The keyword arguments to pass to the request

    Returns
    -------

    response : requests.models.Response
        The server response object. Only returned if request was successful,
        otherwise returns None.

    """

    if rtype not in ["GET", "POST"]:
        warnings.warn("Request type not recognized")
        return None

    total_attempts = 0
    while (total_attempts <= num_attempts):
        if rtype == "GET":
            response = requests.get(url, **kwargs)

        elif rtype == "POST":
            response = requests.post(url, **kwargs)

        if response.status_code == 200:
            return response

        if response.status_code == 429:
            curr_sleep = (1 + total_attempts)*sleep_time
            warnings.warn("Too many requests, waiting " + str(curr_sleep) + " s")
            time.sleep(curr_sleep)
        elif 500 <= response.status_code < 600:
            warnings.warn("Server error encountered. Retrying")

        if response.status_code == 404:
            warnings.warn("No data found")
            return None

        total_attempts += 1

    warnings.warn("Too many failures on requests. Exiting...")
    return None
