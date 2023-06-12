"""
This script still needs writing, I have kept some notes on questions asked by users which may help you...

"""

"""
 > For interferometric data, which units PyAutoLens uses for brightness? I think they are in Jy/arcsec^2 (?) since 
   I have computed the magnification from from my original image in Jy/beam I just wanted to be sure that the conversions 
   I have assumed are fine.

This is correct, the units of brightness are Jy/arcsec^2
"""


"""
>  -When converting the reconstructed source image to a .fits file, what is the best image shape to assume in the 
    interpolation ? By now I am using the same shape as the native one used when defining the real space mask at the 
    beginning, should it be fine? And also, what are the brightness units here?

The reconstruction is essentially a devonvolved image which if you sum up all pixel you get the total flux of the 
source. If if was a regular grid in each pixel the units are Jy/pixel or Jy/arcsec^2

The shape of the grid is really you're choice, so long as it does not extend spatially beyond the extent of the source
reconstruction (as there is no reconstructed source values here in order to enable an accurate interpolation). 

I would use a shape which covers the whole source you are reconstructing (with a bit of padding), and gives visually
appealing data you can use for whatever science you're interested in.

I would imagine the shape of the real space mask is much larger than you really need, but probably doesn't do much
harm either.

One caveat may be magnifications. It could be that as you make the shape of the interpolation grid bigger, the
magnification changes. I'm not sure on this but would advise you experiment to see if the magnifications 
seem "stable" (assuming you are estimate a magnification at some point).
"""
