from io import BytesIO
# from ftplib import FTP, all_errors
import gzip
import json
import xml.etree.ElementTree as ET
import warnings
from urllib.request import urlopen
from urllib.error import URLError
from Bio.PDB import PDBParser, Select, PDBIO
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from numpy import all
import prody.atomic.chain
from dist_analy.util import http_requests

"""
Things that need to be done:
- process and read pdb files:
    x remove waters
    x remove ligands
    x retain only the pdb structures based on uniprot (split chains into multiple files)
    consider how to treat multiple occupancies
    handle insertions (see yuwei's code) - let user treat insertion code
- x compare the residues numbering to SIFTS
- x prepare files to be read for dist_analy
- test for insertions
- create excel file of list of PDB files and their binders
- create flag for checking SIFTs, uniprot checking,
- check dry_apo_pdb when creating a class for import_pdb. will there only be one
  dry_apo_pdb instance and the append res_check list continue to build for each
  function call of dry_apo_pdb(*NCAA) or will there be multiple instances

create a class that opens the FTP login and keeps it open until done processing
all of the PDB structures?

temporarily the pypdb files are copied here until it has been further
developed
"""
CANON = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE',\
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'}
# NONCANON = ['TPO'] ##
# if amino acid then return (add a parameter to include any noncanonical amino acids)
class Dry_Apo_PDB(Select):
    """ https://biopython.org/docs/1.75/api/Bio.PDB.PDBIO.html
    Bio.PDB.Select class for removing water molecules, ligands, and retaining
    the canonical residues and any 3-letter noncanonical residues passed in

    - TODO: investigate about how
    make close after running

    Parameters
    ----------
    *args : list of strings
        list of 3-letter codes added to the list of residues

    Attributes
    ----------
    res_check : list
        list of 3-letter codes that will not be parsed out
    self,residue : Bio.PDB.residue
        residue object passed in

    Example
    -------
    dry_apo_pdb('TPO', 'ASD')
    """
    def __init__(self, *args):
        super().__init__()
        self.res_check = CANON #+ NONCANON
        for arg in args:
            self.res_check.add(arg)
        print(self.res_check)
    def accept_residue(self, residue):
        if residue.get_resname() == "HOH":
            return 0
        elif residue.get_resname() in self.res_check:
            return 1
        else:
            return 0

class PDB_Processer:
    def __init__(self, NCAA:list = [], check_SIFTs:bool=True, ftp_url:str='ftp.ebi.ac.uk/pub/databases/msd/sifts/xml', \
                 filter_warnings = True
                ):
        self.check_SIFTs=check_SIFTs
        # if self.check_SIFTs:
        #     self.ftp = FTP(ftp_url)
        #     self.ftp.login()
        #     self.ftp.set_debuglevel(2)
        if filter_warnings:
            warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        self.select = Dry_Apo_PDB(*NCAA)
        self.io = PDBIO()

        self.parse = PDBParser()
        # create setters to change this?
        self.info_root_url = 'https://data.rcsb.org/rest/v1/core/entry/'
        self.uniprot_root_url = 'https://data.rcsb.org/rest/v1/core/uniprot/'
        self.entity_root_url = 'https://data.rcsb.org/rest/v1/core/polymer_entity/'

    def process_pdb(self, filename: str, path: str, outpath: str, uniprot: str):
        """ Accepts a PDB file and finds the protein chain(s) that matches the
        desired UNIPROT ID and creates a new PDB at the desired output path.
        It also checks the numbering of the protein chain to the SIFTS database.

        Parameters
        ----------
        filename : str
            name of PDB file with .pdb extension
        path : str
            path to the PDB file
        outpath : str
            Desired path to output
        uniprot : str
            UNIPROT ID of the protein
        *NCAA : tuple
            list of noncanonical amino acids that should not be processed out

        Returns
        -------
        bool
            If successful process, then returns True. If not returns false

        """
        ''' to do
        importing pdb from structure. if no pdb file then search for it
        - how to handle insertions -> look at yuwei's package
        - make it easy to import noncanonical amino acids. add in kwargs that allow
          the inclusion of NCAA
        '''
        pdb = filename.split('.')[0]
        structure = self.parse.get_structure(pdb, file=path+filename)[0]
        # print(structure.header.keys())

        info = self._get_any_info(pdb, self.info_root_url)
        entry_all = []

        for entry_id in info['rcsb_entry_container_identifiers']['polymer_entity_ids']:
            try:
                uniprot_query = self._get_any_info(pdb, self.uniprot_root_url, str(entry_id))[0]\
                    ['rcsb_uniprot_container_identifiers']['uniprot_id']
                if not uniprot_query:
                    raise RuntimeError('Unable to find UNIPROT associated with %s'%pdb)
                if uniprot_query == uniprot:
                    entry_all.append(int(entry_id))
            except ValueError:
                print("unable to find data with %s entity #%s, skipping"%(pdb,entry_id))
                pass
        chain_all = []

        if not entry_all:
            raise RuntimeError('%s is not associated with uniprot id %s'%(pdb,uniprot))
            return False

        for entry in entry_all:
            chain_info = self._get_any_info(pdb, self.entity_root_url, entry)
            chain_all.append(chain_info['entity_poly']['pdbx_strand_id'].split(','))

        chain_all = [item for sublist in chain_all for item in sublist]

        if not chain_all:
            raise RuntimeError('Unable to find appropriate chains in %s'%(pdb))
            return False
        # # print(chain_all)
        # if not chain_all:
        #     raise RuntimeError('%s is not associated with uniprot id %s'%(pdb,uniprot))
        #     return False
        if self.check_SIFTs:
            xml_str = self._get_xml_str(pdb)

        process_pdb_list = []
        for chain in chain_all:
            # print(chain, len(list(structure[chain].get_residues())))
            prot = structure[chain]
            if self.check_SIFTs:
                repl_dict = self._xml_replace_dict(xml_str, chain)
                truthy = []
                for key in repl_dict.keys():
                    truthy.append(key == repl_dict[key])

                if all(truthy) == False:
                    self._replace_with_dict(prot, repl_dict)

            self.io.set_structure(structure[chain])
            self.io.save("%s/%s_%s.pdb"%(outpath,pdb,chain), select=self.select)
            process_pdb_list.append("%s_%s.pdb"%(pdb,chain))

        return process_pdb_list

    @staticmethod
    def _get_any_info(pdb: str, url_root: str, *args):
        """ Code adapted from williamgilpin at https://github.com/williamgilpin/pypdb
        under the MIT License

        Uniprot url_root: 'https://data.rcsb.org/rest/v1/core/uniprot/'
        SIFTS url_root: 'https://www.ebi.ac.uk/pdbe/api/mappings/'

        can maybe move to utils

        Parameters
        ----------
        pdb : str
            Description of parameter `pdb`.
        url_root : str
            Description of parameter `url_root`.
        *args : type
            Description of parameter `*args`.

        Returns
        -------
        type
            Description of returned object.

        Examples
        --------
        get_any_info('4eoq', 'https://data.rcsb.org/rest/v1/core/uniprot/', '1')
        would request the uniprot information from entity 1 of 4eoq at the url:
        https://data.rcsb.org/rest/v1/core/uniprot/4eoq/1

        """

        url = url_root + pdb + "/"
        """ does this work with inputting a json dictionary?"""

        for arg in args:
            url += str(arg)

        response = http_requests.request_limited(url)

        if response and response.status_code == 200:
            pass
        else:
            raise ValueError("json retrieval failed, returning None")
            return None

        result  = str(response.text)
        out = json.loads(result)

        return out
## make the FTP login part of the initialization of the class
    def _get_xml_str(self, pdb: str, num_attempts: int = 3, ftp_url:str='ftp.ebi.ac.uk',):
        """ Using the PDBe ftp to download to local memory and unzip the XML file
        encoding the specified PDB residues numbering to the SIFTs database.

        todo: find a way to only login once

        Parameters
        ----------
        pdb : str
            pdb id
        ftp_url : str
            ftp url to pdbe

        Returns
        -------
        str
            returns string of the xml data

        """
        pdb = pdb.lower()
        total_attempts = 0
        while (total_attempts <= num_attempts):
            try:
                with urlopen('ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/xml/%s.xml.gz'%pdb) as gz_file:
                    with gzip.GzipFile(fileobj=gz_file) as zippy:
                        data = zippy.read()
                        return data
                # with bytesio as BytesIO():
                #     self.ftp.retrbinary("RETR /%s.xml.gz"%pdb, callback=bytesio.write)
                #     bytesio.seek(0)
                #     with gzip.GzipFile(fileobj=bytesio) as zippy
                #         data = zippy.read()
            except URLError as e:
                print ("%s, retrying: #%i" % (e,total_attempts))
            # except all_errors as e:
            #     print ("%s, retrying: #%i" % (e,total_attempts))

            total_attempts += 1


    @staticmethod
    def _xml_replace_dict(xml_string: str, chain: str):
        """ Reads in the XML string to create a dictionary of the corresponding PDB
        residue numbering to the SIFTs residue numbering

        Parameters
        ----------
        xml_string : str
            str passed in from get_xml_str()
        chain : str
            specified PDB chain

        Returns
        -------
        dict
            Residue mapping from PDB to SIFTs

        """
        tree = ET.ElementTree(ET.fromstring(xml_string))
        root = tree.getroot()

        pre = '{'+root.attrib['{http://www.w3.org/2001/XMLSchema-instance}schemaLocation'].split()[0]+'}'
        dict_repl = {}
        excl = ['Cloning artifact', 'Not_Observed', 'Expression tag']
        for entity in root.iter('%sentity'%pre):
            if entity.attrib['entityId']==chain:
                for res in entity.iter('%sresidue'%pre):
                    skip = False
                    for detail in res.iter('%sresidueDetail'%pre):
                        if detail.text in excl:
                            skip=True
                            continue
                    if skip:
                        continue
                    pdb_id, unip_id = -999, -999 # default values to ignore
                    for crossref in res.iter('%scrossRefDb'%pre):
                #         print(crossref.keys())
                        if crossref.attrib['dbSource']=="PDB":
                            pdb_id = crossref.attrib['dbResNum']
                #             print(pdb_id)
                        if crossref.attrib['dbSource']=="UniProt":
                            unip_id = crossref.attrib['dbResNum']
                #             print(unip_id)
                    dict_repl[pdb_id] = unip_id
            return dict_repl

    @staticmethod
    def _replace_with_dict(chain_object: prody.atomic.chain, replace_dict: dict):
        """ Reassign residue numbering of chain object by the replace dictionary

        Parameters
        ----------
        chain_object : prody.atomic.chain
            chain object of the protein
        replace_dict : dict
            dictionary of SIFTs mapping

        Returns
        -------

        """
        for residue in reversed(list(chain_object.get_residues())):
            res_id = list(residue.id)
            if str(res_id[1]) in replace_dict:
                repl_id = replace_dict[str(res_id[1])]
                if repl_id == -999:
                    continue
                res_id[1] = int(repl_id)
                residue.id = tuple(res_id)
