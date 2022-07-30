import unittest
import f90nml
import numpy as np
import xarray as xr
from pathlib import Path
from gs2uq import GS2Encoder, GS2Decoder

class TestUQIO(unittest.TestCase):
    def test_GS2Encoder(self):
        """Substitutes `param = {"species_parameters_1::fprim": 9001}` 
        into a template application input, saves it as "GS2_test_input.in", then
        reads the file. If the file matches the input value, the test is passed.
        """

        target_dir = Path(__file__).parent / "tests"

        encoder = GS2Encoder(template_filename=f"{target_dir}/test.in",
                             target_filename="GS2_test_input.in")

        filepath = target_dir / encoder.target_filename

        # Remove "GS2_test_input.in" if it already exists
        if filepath.is_file():
            filepath.unlink()

        # Create input file "GS2_test_input.in"
        param = {"species_parameters_1::fprim": 9001}
        encoder.encode(params=param, target_dir=target_dir)

        # Read "GS2_test_input.in" using f90nml
        with open(filepath, "r") as f:
            param_dict = f90nml.read(f)
        fprim = param_dict["species_parameters_1"]["fprim"]

        # Test GS2Encoder.encode():    
        self.assertEqual(fprim, 9001)

    
    def test_GS2Decoder(self):
        """Parses the NetCDF file `test.out.nc` in `tests` folder and converts it to the 
        EasyVVUQ internal dictionary based format. The output should be 
        results = {
            "ky": np.array([0.00, 0.10, 0.20, ..., 1.00]),
            "omega/4": np.array([10.5, 10.5, 10.5, ..., 10.5]), 
            "gamma": np.array([1337, 1337, 1337, ..., 1337])}
        """

        decoder = GS2Decoder(target_filename="test.out.nc")
        run_info = {"run_dir": "./tests"}
        run_info["run_dir"]
        results = decoder.parse_sim_output(run_info=run_info)

        ky = np.linspace(0, 1.0, 11).tolist()
        qtr_omegas = np.full(11, 42/4).tolist()
        gammas = np.full(11, 1337).tolist()

        # Test GS2Decoder.parse_sim_output():
        self.assertEqual(results["ky"], ky)
        self.assertEqual(results["omega/4"], qtr_omegas)
        self.assertEqual(results["gamma"], gammas)
        """
        np.testing.assert_allclose(results["ky"], ky)
        np.testing.assert_allclose(results["omega/4"], qtr_omegas)
        np.testing.assert_allclose(results["gamma"], gammas)
        """

        # Test GS2Decoder.sim_complete()
        self.assertEqual(decoder.sim_complete(run_info=run_info), True)


if __name__ == "__main__":
    unittest.main()
    
