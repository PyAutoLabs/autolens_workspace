"""
__Sub Gridding__

The calculation above uses a `Grid2D` object, with a `sub-size=1`, meaning it does not perform oversampling to
evaluate the light profile flux at every image pixel.

**PyAutoLens** has alternative methods of computing the lens galaxy images above, which uses a grid whose sub-size
adaptively increases depending on a required fractional accuracy of the light profile.

 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/structures/grids/two_d/grid_iterate.py
"""
# masked_imaging_iterate = dataset.apply_mask(mask=mask)
# masked_imaging_iterate = masked_imaging_iterate.apply_over_sampling(
#     over_sampling=al.OverSamplingDataset(uniform=al.OverSamplingIterate())
# )
#
# image_iterate = lens_galaxy.image_2d_from(grid=masked_imaging_iterate.grid)
# blurring_image_iterate = lens_galaxy.image_2d_from(grid=masked_dataset.grids.blurring)


"""
__Sourrce Plane Interpolation__

For the `VoronoiNoInterp` pixelization used in this example, every image-sub pixel maps to a single source Voronoi
pixel. Therefore, the plural use of `pix_indexes` is not required. However, for other pixelizations each sub-pixel
can map to multiple source pixels with an interpolation weight (e.g. `Delaunay` triangulation or a `Voronoi` mesh
which uses natural neighbor interpolation).

`MapperVoronoiNoInterp.pix_index_for_sub_slim_index`: 
https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/inversion/mappers/voronoi.py

`pixelization_index_for_voronoi_sub_slim_index_from`: 
 https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/util/mapper_util.py 
"""


"""
The number of pixels that each sub-pixel maps too is also stored and extracted. This is used for speeding up 
the calculation of the `mapping_matrix` described next.

As discussed above, because for the `VoronoiNoInterp` pixelization where every sub-pixel maps to one source pixel,
every entry of this array will be equal to 1.
"""
# pix_sizes_for_sub_slim_index = mapper.pix_sizes_for_sub_slim_index

"""
When each sub-pixel maps to multiple source pixels, the mappings are described via an interpolation weight. For 
example, for a `Delaunay` triangulation, every sub-pixel maps to 3 Delaunay triangles based on which triangle
it lands in.

For the `VoronoiNoInterp` pixelization where every sub-pixel maps to a single source pixel without inteprolation,
every entry of this weight array is 1.0.
"""
# pix_weights_for_sub_slim_index = mapper.pix_weights_for_sub_slim_index
