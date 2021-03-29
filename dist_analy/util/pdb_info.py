"""Utility functions for requesting URLs over HTTP"""
"""Code adapted from williamgilpin at https://github.com/williamgilpin/pypdb
   under the MIT License
"""

from typing import Optional

import json
import time
import requests
import warnings
import csv

def pdb_csv(pdb_list: list, uniprot: str, csv_file: str):
    """ Creates a csv files of information parsed from the rcsb database about
    the pdb structures. Includes information about the resolution, binders, amd
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

    csv_data = []
    dict_items = ['PDB ID', 'Title', 'Description', 'Method', 'Resolution', \
                  'Polymer Binders', 'Major Binders', 'Minor Binders', 'Modifications']
    for pdb in pdb_list:
        print(pdb)
        pdb_dict = {}
        polymer_binders = []
        data = get_any_info(info_root,pdb)
        pdb_dict['PDB ID'] = pdb
        pdb_dict["Title"] = data['struct']['title']
        try:
            pdb_dict["Description"] = data['struct']['pdbx_descriptor']
        except KeyError:
            pdb_dict["Description"] = None
        pdb_dict["Method"] = data["rcsb_entry_info"]["experimental_method"]
        try:
            pdb_dict["Resolution"] = data["pdbx_vrpt_summary"]["pdbresolution"]
        except:
            pdb_dict["Resolution"] = None
        try:
            pdb_dict["Minor Binders"] = data['rcsb_entry_info']["nonpolymer_bound_components"]
        except KeyError:
            pdb_dict["Minor Binders"] = None
        num_poly_entity = data["rcsb_entry_container_identifiers"]["polymer_entity_ids"]
        major_binders = []
        try:
            for binding_dict in data["rcsb_binding_affinity"]:
                if binding_dict["comp_id"] not in major_binders:
                    major_binders.append(binding_dict["comp_id"])
            pdb_dict["Major Binders"] = major_binders
        except KeyError:
            pdb_dict["Major Binders"] = None
        for i in num_poly_entity:
            data_1 = get_any_info(info_root_1,pdb,i)
            try:
                for x in data_1["rcsb_polymer_entity_container_identifiers"]["uniprot_ids"]:
                    if x == uniprot:
                        try:
                            pdb_dict['Modifications'] = data_1["rcsb_polymer_entity_container_identifiers"]["chem_comp_nstd_monomers"]
                        except KeyError:
                            pdb_dict['Modifications'] = None
                    else:
                        polymer_binders.append(data_1["rcsb_polymer_entity"]["pdbx_description"])
            except KeyError:
                polymer_binders.append(data_1["rcsb_polymer_entity"]["pdbx_description"])
        pdb_dict["Polymer Binders"] = polymer_binders
        csv_data.append(pdb_dict)
    with open(csv_file, 'w') as f1:
        writer = csv.DictWriter(f1, fieldnames = dict_items)
        writer.writeheader()
        writer.writerows(csv_data)

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
        response = request_limited(url_root)

    if request_dict:
        response = request_limited(url_root, params=request_dict)

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
