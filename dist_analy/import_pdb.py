# from io import BytesIO
# from ftplib import FTP, all_errors
import gzip
import json
import xml.etree.ElementTree as ET
import warnings
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError
from Bio.PDB import PDBParser, Select, PDBIO
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from numpy import all
import prody.atomic.chain
from dist_analy.util import pdb_info

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
x create flag for checking SIFTs, uniprot checking,
x check dry_apo_pdb when creating a class for import_pdb. will there only be one
  dry_apo_pdb instance and the append res_check list continue to build for each
  function call of dry_apo_pdb(*NCAA) or will there be multiple instances
- suppress urllib output
- xrl_repl_dict implement conversion to any other residue numbering system
- add flag to process NMR structures
- use prody instead of biopython to read in pdb files.
"""
CANON = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE',\
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'}
# NONCANON = ['TPO'] ##
# if amino acid then return (add a parameter to include any noncanonical amino acids)
class Dry_Apo_PDB(Select):
    """
    Bio.PDB.Select class for removing water molecules, ligands, and retaining
    the canonical residues and any 3-letter noncanonical residues passed in

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

    Notes
    -----
    https://biopython.org/docs/1.75/api/Bio.PDB.PDBIO.html

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
    def __init__(self, NCAA:list = [], check_SIFTs:bool=True,
                 check_database:bool = True, filter_warnings:bool = True):
        """ PDB_Processer class constructor

        Parameters
        ----------
        NCAA : list, optional
            list of three letter non-canonical amino acid codes to exclude from
            being parsed
        check_SIFTs : bool, default: True
            flag to check the SIFTs database and create a repl_dict
        check_database : bool, default: True
            flag to check the RCSB database to match protein chains to UniProt accesion numbers
        filter_warnings : bool, default: True
            flag to filter PDBConstructionWarnings

        Returns
        -------

        """
        self.check_SIFTs=check_SIFTs
        self.check_database=check_database
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
        self.ftp_url = 'ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/xml/'
        self.info_root_url = 'https://data.rcsb.org/rest/v1/core/entry/'
        self.uniprot_root_url = 'https://data.rcsb.org/rest/v1/core/uniprot/'
        self.entity_root_url = 'https://data.rcsb.org/rest/v1/core/polymer_entity/'

    def get_database_info(self, pdb: str, uniprot: str):
        """ process_pdb helper function that searches the RCSB database for
        which chain in the given PDB file matches the given UniProt accession
        number. It can then check the SIFTs databases to create a dictionary to
        map the PDB residue IDs to the UniProt residue numbering.

        Parameters
        ----------
        pdb : str
            PDB id
        uniprot : str
            UniProt accession number

        Returns
        -------
        list
            list of PDB chain IDs corresponding to the UniProt accession number

        """
        info = pdb_info.get_any_info(self.info_root_url, pdb)
        entry_all = []

        for entry_id in info['rcsb_entry_container_identifiers']['polymer_entity_ids']:
            try:
                uniprot_query = pdb_info.get_any_info(self.uniprot_root_url, pdb, str(entry_id))[0]\
                    ['rcsb_uniprot_container_identifiers']['uniprot_id']
                if not uniprot_query:
                    raise RuntimeError('Unable to find UniProt associated with %s'%pdb)
                if uniprot_query == uniprot:
                    entry_all.append(int(entry_id))
            except ValueError:
                print("unable to find data with %s entity #%s, skipping"%(pdb,entry_id))
                pass
        chain_all = []

        if not entry_all:
            raise RuntimeError('%s is not associated with uniprot id %s'%(pdb,uniprot))
            return None

        for entry in entry_all:
            chain_info = pdb_info.get_any_info(self.entity_root_url, pdb, entry)
            chain_all.append(chain_info['entity_poly']['pdbx_strand_id'].split(','))

        chain_all = [item for sublist in chain_all for item in sublist]

        if not chain_all:
            raise RuntimeError('Unable to find appropriate chains in %s'%(pdb))
            return None

        return chain_all

    def process_pdb(self, filename: str, path: str, outpath: str, uniprot: str = '',
                    chain_all: list = [], repl_dict: dict = {}
                    ):
        """ Accepts a PDB file and finds the protein chain(s) that matches the
        desired UniProt ID and creates a new PDB at the desired output path.
        It also checks the numbering of the protein chain to the SIFTS database.

        Parameters
        ----------
        filename : str
            name of PDB file with .pdb extension
        path : str
            path to the PDB file
        outpath : str
            Desired path to output
        UniProt : str, optional
            UniProt accession number of the protein
        chain_all: list, optional
            list of chains that represent the desired protein
        repl_dict: dict, optional
            dictionary mapping the pdb residue ID to the desired residue ID


        Returns
        -------
        list
            returns list of processed pdb files

        """
        ''' to do
        - importing pdb from structure. if no pdb file then search for it
        - importing structure from pdb file (not from rcsb: think md simulation frame)
         and renumber the residues based on uniprot sequence
        - do not write out processed pdb file, just store on disk (unsure if this is possible)
        - delete the original file flag
        - how to handle insertions -> look at yuwei's package
        '''

        check_SIFTs_ = self.check_SIFTs
        Path(outpath).mkdir(parents=True, exist_ok=True)

        pdb = filename.split('.')[0]
        structure = self.parse.get_structure(pdb, file=path+filename)

        if self.check_database:
            if not uniprot:
                raise ValueError("To check the databases need to pass in a UniProt accession number")
            else:
                chain_all = self.get_database_info(pdb, uniprot)
        if not self.check_database and not chain_all:
            raise ValueError("If not check_database, must pass in a list of chains to chain_all")

        if check_SIFTs_:
            xml_str = self._get_xml_str(pdb, ftp_url=self.ftp_url)
            if not xml_str:
                warnings.warn("%s: unable to get SIFTs residues, skipping residue numbering check"%pdb)
                check_SIFTs_ = False

        process_pdb_list = []
        for chain in chain_all:
            # print(chain, len(list(structure[chain].get_residues())))
            for i,struct in enumerate(structure):
                prot = struct[chain]
                if check_SIFTs_:
                    # print(chain)
                    if not repl_dict:
                        repl_dict = self._xml_replace_dict(xml_str, chain)
                if repl_dict:
                    truthy = []
                    for key in repl_dict.keys():
                        truthy.append(key == repl_dict[key])

                    if all(truthy) == False:
                        # print(repl_dict)
                        self._replace_with_dict(prot, repl_dict)

                self.io.set_structure(prot)
                if len(structure) > 1:
                    outf = "%s_%s_%i.pdb"%(pdb,chain,i)
                else:
                    outf = "%s_%s.pdb"%(pdb,chain)
                self.io.save("%s/%s"%(outpath,outf), select=self.select)
                process_pdb_list.append(outf)

        return process_pdb_list

    # def find_uniprot_seq

## make the FTP login part of the initialization of the class
    @staticmethod
    def _get_xml_str(pdb: str, num_attempts: int = 3, ftp_url:str='ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/xml/'):
        """ Using the PDBe ftp to download to local memory and unzip the XML file
        encoding the specified PDB residues numbering to the SIFTs database.

        todo: find a way to only login once
        suppress output of urllib

        Parameters
        ----------
        pdb : str
            PDB id
        num_attemps: int, default: 3
            number of attempts to access ftp_url
        ftp_url : str, default: ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/xml/
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
                with urlopen(ftp_url+'%s.xml.gz'%pdb) as gz_file:
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

        warnings.warn("Too many failures to get %s.xml.gz. Return none and exit..."%pdb)
        return None


    @staticmethod
    def _xml_replace_dict(xml_string: str, chain: str):
        """ Reads in the XML string to create a dictionary of the corresponding PDB
        residue numbering to the UniProt residue numbering via the SIFTs database.
        Any residues that don't have a corresponding UniProt numbering will be
        ignored and given IDs starting from -999

        todo:
            implement conversion to any other residue numbering system

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
            # print(entity.attrib)
            if entity.attrib['entityId']==chain:
                ignore_count = 0
                for res in entity.iter('%sresidue'%pre):
                    skip = False
                    for detail in res.iter('%sresidueDetail'%pre):
                        if detail.text in excl:
                            skip=True
                            continue
                    if skip:
                        continue
                    pdb_id = res.find(".//%scrossRefDb[@dbSource='PDB']"%pre).attrib['dbResNum']
                    try:
                        unip_id = res.find(".//%scrossRefDb[@dbSource='UniProt']"%pre).attrib['dbResNum']
                    except AttributeError:
                        unip_id = -999 + ignore_count
                        ignore_count += 1
                    dict_repl[pdb_id] = unip_id
        return dict_repl

    @staticmethod
    def _replace_with_dict(chain_object: prody.atomic.chain, replace_dict: dict):
        """ Reassign residue numbering of chain object by the replace dictionary
        An error may occur when renumbering the residues of the chain object to
        a residue number that has already been assigned. To circumvent this,
        the function first attempts shifting the residues forward. If that fail
        it attempts to shift the residues backwards.

        Parameters
        ----------
        chain_object : prody.atomic.chain
            chain object of the protein
        replace_dict : dict
            dictionary of SIFTs mapping

        Returns
        -------

        """
        try:
            try:
                # count = 0
                for residue in reversed(list(chain_object.get_residues())):
                    res_id = list(residue.id)
                    if str(res_id[1]) in replace_dict:
                        repl_id = replace_dict[str(res_id[1])]
                        res_id[1] = int(repl_id)
                        residue.id = tuple(res_id)
            except ValueError:
                # count = 0
                for residue in list(chain_object.get_residues()):
                    res_id = list(residue.id)
                    if str(res_id[1]) in replace_dict:
                        repl_id = replace_dict[str(res_id[1])]
                        res_id[1] = int(repl_id)
                        residue.id = tuple(res_id)
        except ValueError:
            count = 0
            for residue in reversed(list(chain_object.get_residues())):
                res_id = list(residue.id)
                res_id[1] = -999 + count
                residue.id = tuple(res_id)
                count += 1
            for residue in reversed(list(chain_object.get_residues())):
                res_id = list(residue.id)
                if str(res_id[1]) in replace_dict:
                    repl_id = replace_dict[str(res_id[1])]
                    res_id[1] = int(repl_id)
                    residue.id = tuple(res_id)
