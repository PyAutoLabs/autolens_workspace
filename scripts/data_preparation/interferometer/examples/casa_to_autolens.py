"""
THIS SCRIPT IS INCOMPLETE AND IN DEVELOPMENT BUT IT HOPEFULLY IS INFORMATIVE ENOUGH TO HELP YOU GET STARTED.

If not contact us on SLACK!

__CASA to PyAutoLens__

The interferometer object in AutoLens takes as arguments the visibilities, uv_wavelengths and noise_map (i.e. the
SIGMA associated with the visibilities). The uv_wavelengths should have a shape (n_vis, 2) where n_vis is the total
number of visibilities and the two different columns correspond to the u, v components of the uv_wavenelengths.

The visibilities and noise map are complex arrays of shape (n_vis, ) where the real and imag of the array are the real
and imag parts of the visibilities and SIGMA.

- Exporting visibilities and uv_wavelengths with CASA.

ALMA data are stored in .ms data structure. The shape of the visibilities in this structure has a shape
of (2, n_spw, n_c, n_v, 2) where n_spw in the number of spectral windows, n_c is the number of channels, n_v is the
number of visibilities. The first two rows of this array correspond to the 2 different polarisations.
"""
import numpy as np
import os

import autolens as al

"""
You can use the CASA task "split" to create 4 new different .ms directories, each corresponding to a different spw.
"""

# split(
#     vis="name.ms",
#     outputvis="name_spw_0.ms",
#     keepmms=True,
#     field="SPT-0418",
#     spw="0",
#     datacolumn="data",
#     keepflags=True
# )

"""
Now each "name_spw.ms" will have the visibilities stored in an array shaped (2, n_c, n_v, 2). In order to reduce the 
size of the visibilities we often average visibilities corresponding to different channels. 

We can use the CASA task "split", setting the argument width=<number of channels> (e.g. width=128), again for this 
step (note, however, that different spws can have different number of channels)
"""

# split(
#     vis="name_spw_0.ms",
#     outputvis="name_spw_0_chanaveraged.ms",
#     keepmms=True,
#     field="SPT-0418",
#     width=128,
#     datacolumn="data",
#     keepflags=True
# )

"""
To return the visibilities and uv_wavelengths (in units of wavelengths, NOT meters - which is the way they are 
stored in the .ms) from each .ms directory that you created in the previous step use the following scripts (save 
them in a .fits format):
"""
#
# def getcol_wrapper(ms, table, colname):
#
#     if os.path.isdir(ms):
#         tb.open(
#             "{}/{}".format(ms, table)
#         )
#
#         col = np.squeeze(
#             tb.getcol(colname)
#         )
#
#         tb.close()
#     else:
#         raise IOError(
#             "{} does not exist".format(ms)
#         )
#
#     return col
#
# def get_visibilities(ms):
#
#     if os.path.isdir(ms):
#         data = getcol_wrapper(
#             ms=ms,
#             table="",
#             colname="DATA"
#         )
#     else:
#         raise IOError(
#             "{} does not exisxt".format(ms)
#         )
#
#     visibilities = np.stack(
#         arrays=(data.real, data.imag),
#         axis=-1
#     )
#
#     return visibilities
#
#
# def get_uv_wavelengths(ms):
#
#     def convert_array_to_wavelengths(array, frequency):
#
#         if astropy_is_imported:
#             array_converted = (
#                 (array * units.m) * (frequency * units.Hz) / constants.c
#             ).decompose()
#
#             array_converted = array_converted.value
#         else:
#             factor = 3.3356409519815204e-09
#             array_converted = array * frequency * factor
#
#         return array_converted
#
#     if os.path.isdir(ms):
#         uvw = getcol_wrapper(
#             ms=ms,
#             table="",
#             colname="UVW"
#         )
#     else:
#         raise IOError(
#             "{} does not exist".format(ms)
#         )
#
#     chan_freq = getcol_wrapper(
#         ms=ms,
#         table="SPECTRAL_WINDOW",
#         colname="CHAN_FREQ"
#     )
#
#     chan_freq_shape = np.shape(chan_freq)
#
#     if np.shape(chan_freq):
#
#         u_wavelengths, v_wavelengths = np.zeros(
#             shape=(
#                 2,
#                 chan_freq_shape[0],
#                 uvw.shape[1]
#             )
#         )
#
#         for i in range(chan_freq_shape[0]):
#             u_wavelengths[i, :] = convert_array_to_wavelengths(array=uvw[0, :], frequency=chan_freq[i])
#             v_wavelengths[i, :] = convert_array_to_wavelengths(array=uvw[1, :], frequency=chan_freq[i])
#
#     else:
#
#         u_wavelengths = convert_array_to_wavelengths(array=uvw[0, :], frequency=chan_freq)
#         v_wavelengths = convert_array_to_wavelengths(array=uvw[1, :], frequency=chan_freq)
#
#
#     uv_wavelengths = np.stack(
#         arrays=(u_wavelengths, v_wavelengths),
#         axis=-1
#     )
#
#     return uv_wavelengths

"""
Now we have the visibilities and uv_wavelengths arrays with shapes, (2, n_v, 2) and (n_v, 2), respectively. Note that 
the same point in uv-space has 2 visibilities associated with it, corresponding to the two different polarisations. 
You can further average the visibilities for the two polarisations "np.average(visibilities, axis=0)".

Now back to where we started ... AutoLens requires you to create the object "interferometer":
"""
#
# dataset = al.Interferometer(
#     visibilities=,
#     noise_map=,
#     uv_wavelengths=,
#     real_space_mask=,
#     settings=al.SettingsInterferometer(
#         transformer_class=al.TransformerNUFFT,
#     ),
# )

"""
Assuming you have averaged the polarisations you can simply use:
"""

# visibilities=al.Visibilities(visibilities[:, 0] + 1j * visibilities[:, 1])
# uv_wavelengths=uv_wavelengths

"""
If not, and visibilities still have a shape (2, n_v, 2) then add the following lines before initialising the 
interferometer object:
"""

# visibilities = visibilities.reshape(int(visibilities.shape[0] * visibilities.shape[1]), 2)
# uv_wavelengths = np.concatenate((uv_wavelengths, uv_wavelengths), axis=0)
# uv_wavelengths = uv_wavelengths.reshape(int(uv_wavelengths.shape[0] * uv_wavelengths.shape[1]), 2)

"""
**NOTE: You have visibilities from 4 spectral windows. Concatenate them in one single array**

- Exporting SIGMA with CASA.

The procedure to export the SIGMA using CASA is the same as the visibilities with ones extra step. By default the 
SIGMA are computed in a relative way and they do not reflect the real scatter in the visibility data. 

We can use the CASA task "statwt" to rescale the visibilities according to their scatter. Execute the following 
command after you performed the first "split"
"""
#
# statwt(
#     vis="name_spw_0.ms",
#     datacolumn="data"
# )

"""
Repeat the steps above, but use the following script to return the SIGMA
"""

# def get_sigma(ms):
#
#     sigma = getcol_wrapper(
#         ms=ms,
#         table="",
#         colname="SIGMA"
#     )
#
#     return sigma

"""
CASA assigns the same error to both the real and imag components of the visibilities, so the sigma array will 
have a shape (2, n_v, ). Use, "sigma=np.stack(arrays=(sigma, sigma),axis=-1)".
"""
