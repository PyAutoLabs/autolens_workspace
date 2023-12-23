"""
Plots: Publication Subhalo Scan
===============================

Scientific papers have specific requirements on producing plots and figures so that they look good in the paper.
This includes large labels, clear axis ticks and minimizing white space.

This example illustrates how to plot an image-plane image (e.g. the observed data of a strong lens, or the
image-plane model-image of a fit) with a subhalo scanning map overlaid, which gives the increase in Bayesian
evidence of every model including a subhalo.

We will not perform a model-fit to set up the evidence values and instead simply asssume an input set of values
for efficiency.

__Start Here Notebook__

If any code in this script is unclear, refer to the `plot/start_here.ipynb` notebook.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from os import path
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

First, lets load example imaging of of a strong lens as an `Imaging` object.
"""
dataset_name = "simple__no_lens_light"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    radius=3.0, shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales
)

dataset = dataset.apply_mask(mask=mask)

"""
We first set up all the publication settings described in `autolens_workspace/scripts/plot/publication/image.py`
which make image-plane images look publication quality.

Checkout that example for an explanation of why we use these settings.
"""
mat_plot_image = aplt.MatPlot2D(
    title=aplt.Title(label=f"Subhalo Scanning Publication Plot", fontsize=24),
    yticks=aplt.YTicks(
        fontsize=22,
        manual_suffix='"',
        manual_values=[-2.5, 0.0, 2.5],
        rotation="vertical",
        va="center",
    ),
    xticks=aplt.XTicks(fontsize=22, manual_suffix='"', manual_values=[-2.5, 0.0, 2.5]),
    ylabel=aplt.YLabel(ylabel=""),
    xlabel=aplt.XLabel(xlabel=""),
    colorbar=aplt.Colorbar(
        manual_tick_values=[0.0, 0.3, 0.6], manual_tick_labels=[0.0, 0.3, 0.6]
    ),
    colorbar_tickparams=aplt.ColorbarTickParams(labelsize=22, labelrotation=90),
)

"""
We now plot the image without the subhalo scanning map, to remind ourselves what it looks like.
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset, mat_plot_2d=mat_plot_image)
dataset_plotter.figures_2d(
    data=True,
)

"""
__Subhalo Scanning Overlay__

To plot a subhalo scan, we overlay an `Array2D` of values where each value is the increase in log Bayesian evidence
of the subhalo model compared to model without it.

The tutorial `autolens_workspace/results/advanced/result_subhalo_grid.py` shows how to compute this quantity via
a full model-fit.

In this example, we will simply manually define a 5 x 5 grid of values which we will plot.
"""
evidence_array = al.Array2D.no_mask(
    values=[
        [-1.0, -1.0, 1.0, 2.0, 0.0],
        [0.0, 1.0, 2.0, 3.0, 2.0],
        [0.0, 3.0, 5.0, 20.0, 8.0],
        [0.0, 1.0, 1.0, 15.0, 5.0],
        [0.0, 1.0, 1.0, 0.0, 0.0],
    ],
    pixel_scales=1.0,
)

evidence_max = np.max(evidence_array)
evidence_half = evidence_max / 2.0

visuals = aplt.Visuals2D(array_overlay=evidence_array)
include = aplt.Include2D(border=False)

mat_plot = (
    aplt.MatPlot2D(
        array_overlay=aplt.ArrayOverlay(
            cmap="RdYlGn", alpha=0.6, vmin=0.0, vmax=evidence_max
        ),
        colorbar=aplt.Colorbar(
            manual_tick_values=[0.0, evidence_half, evidence_max],
            manual_tick_labels=[
                0.0,
                np.round(evidence_half, 1),
                np.round(evidence_max, 1),
            ],
        ),
    )
    + mat_plot_image
)

"""
Before discussing the figure, lets look at the plot:
"""
dataset_plotter = aplt.ImagingPlotter(
    dataset=dataset,
    mat_plot_2d=mat_plot,
    visuals_2d=visuals,
    include_2d=include,
)
dataset_plotter.figures_2d(
    data=True,
)

"""
Now lets justify the choice of settings we used above:

 - `cmap="cool"`: in order to ensure low evidence values (e.g. the zeros) and high evidence values are visible clear
   we require a colormap which is distinct from the colormap of the image. The colormap cool achieves this, as the 
   light blue is a distinct background from the dark blue and pink detection distinct from the data colormap.
   
 - `alpha=0.6`: we want the background image to be visible, so that we can compare it to the subhalo scanning map.
   The alpha parameter makes the subhalo scanning map appear transluscent, so that we can see both. 
   
 - `vmin / vmax / colorbar`: These properties are customized so that the colorbar runs from zero to the maximum
   Bayesian evidence, making it easy to see the subhalo scanning map values.
"""
