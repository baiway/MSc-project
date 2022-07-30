from pathlib import Path
import subprocess
import f90nml
import xarray as xr
import logging

class GS2Encoder:
    def __init__(self, template_filename=None, target_filename="GS2_input.in"):
        self.template_filename = template_filename
        self.target_filename = target_filename

    def encode(self, params={}, target_dir=''):
        """Substitutes `params` into a template application input, saves in `target_dir`
        Parameters
        ----------
        params        : dict
            Parameter informations in dictionary.
        target_dir    : str
            Path to directory where application input will be written.
        """

        # Open template filename
        try:
            with open(self.template_filename, "r") as f:
                template = f90nml.read(f)
        except FileNotFoundError:
            raise RuntimeError(
                f"the template file specified ({template}) does not exist")
        
        if not target_dir:
            raise RuntimeError("No target directory specified to encoder")
        
        # Load model parameters (e.g. fprim, tprim) into template dictionary
        try:
            for key, value in params.items():
                group, param = key.split("::")
                template[group][param] = value
        except KeyError as e:
            self._log_substitution_failure(e)   
        
        # Write target input file
        try:
            target_filepath = Path(target_dir) / self.target_filename
            with open(target_filepath, "w") as new_f:
                f90nml.write(template, new_f)
        except FileNotFoundError:
            raise RuntimeError(
                f"the directory specified ({target_dir}) does not exist")
        
    def _log_substitution_failure(self, exception):
        reasoning = (f"\nFailed substituting into template input file: "
                     f"{self.template_fname}.\n"
                     f"KeyError: {str(exception)}.\n")
        logging.error(reasoning)

        raise KeyError(reasoning)


class GS2Decoder:
    def __init__(self, target_filename):
        self.target_filename = target_filename

    def sim_complete(self, run_info=None):
        """Checks whether file `output.exit_reason` exists. 
        If it does, the simulation has completed. 
        Note: may need a more refined test later down the line.
        """

        file_to_print = self._get_output_path(run_info=run_info, 
                                                outfile="GS2_print.txt")
        with open(file_to_print, "r") as f:
            for last_line in f:
                pass
        
        if "Run finished" in last_line:
            return True
        else:
            return False
    
    def parse_sim_output(self, run_info={}):
        """Parses the NetCDF file and converts it to the EasyVVUQ internal dictionary based
        format. The output has the form {"aky": [...], "omega/4": [...], "gamma": [...]}.
        Parameters
        ----------
        run_info: dict
            Information about the run (used to retrieve construct the absolute path
            to the NetCDF file that needs decoding.
        """

        results = {"ky": None, "omega/4": None, "gamma": None}

        run_dir = Path(run_info["run_dir"])
        output_filepath = run_dir / self.target_filename

        with xr.open_dataset(output_filepath, engine="netcdf4") as ds:
            results["ky"] = ds.ky.data.tolist()
            results["omega/4"] = (ds.omega_average.isel(ri=0, t=-1).squeeze().data / 4).tolist()
            results["gamma"] = ds.omega_average.isel(ri=1, t=-1).squeeze().data.tolist()
        
        return results

    @staticmethod
    def _get_output_path(run_info=None, outfile=None):
        """Constructs absolute path from the `target_filename` and the `run_dir` parameter
        in the `run_info` retrieved from the database.
        Parameters
        ----------
        run_info: dict
            Run info as retrieved from the database.
        outfile: str
            Filename of the file to be parsed.
        Returns
        -------
        str
            An absolute path to the output file in the run directory.
        """
        run_path = Path(run_info["run_dir"])
        if not run_path.is_dir():
            raise RuntimeError(f"Run directory does not exist: {run_path}")
        return run_path / outfile
