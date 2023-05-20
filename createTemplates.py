#import nuke
import torch
import viz
from rotoshapes import shapeFromRotopaint
import pyperclip

def ensure_continuous(curves):
    # The curves parameter is assumed to be a numpy array of shape (n_curves, 3, 2)
    # where n_curves is the number of curves.
    n_curves = curves.shape[0]

    for i in range(1, n_curves):
        # Get the end point of the previous curve and the start point of the current curve
        prev_end = curves[i - 1, -1, :]
        curr_start = curves[i, 0, :]

        # Compute the midpoint
        midpoint = (prev_end + curr_start) / 2

        # Update the end point of the previous curve and the start point of the current curve
        curves[i - 1, -1, :] = midpoint
        curves[i, 0, :] = midpoint

    # Make sure the end point of the last curve matches the start point of the first curve
    curves[-1, -1, :] = curves[0, 0, :]

    return curves


nukescript=r"D:/DeepParametricShapes/nukeScripts/templates.nk"
nuke.scriptOpen(nukescript)
rotonode=nuke.toNode("pupil")
shape=shapeFromRotopaint(rotonode)
curves=shape.convert_to_bezier_format(shape.npCtrlPoints)
#convert list of 4x2 numpy arrays to tensor
tensor=torch.zeros(len(curves),4,2)
for i in range(len(curves)):
    tensor[i]=torch.from_numpy(curves[i])
viz.plotCubicSpline(tensor)

reduced_degree_curves=shape.reduceDegree()
#bezier curve format needs to be converted back to tensor
tensor=torch.zeros(len(reduced_degree_curves),3,2)
for i in range(len(reduced_degree_curves)):
    tensor[i]=torch.from_numpy(reduced_degree_curves[i].nodes.T)

# normalise the control points to be in the range [0, 1]



# we need to make sure that the last point in each curve and the first point in the next curve are the same, taking the midpoint
# of the two points
tensor=ensure_continuous(tensor)

#add dimension for num shapes
tensor=tensor.unsqueeze(0)

viz.plotQuadraticSpline(tensor)
subCurves=shape.subdivideCurves(reduced_degree_curves)
tensor=torch.zeros(len(reduced_degree_curves),3,2)
for i in range(len(reduced_degree_curves)):
    tensor[i]=torch.from_numpy(reduced_degree_curves[i].nodes.T)



tensor=ensure_continuous(tensor)

#add dimension for num shapes
tensor=tensor.unsqueeze(0)
#flatten tensor, turn into list and copy to clipboard

viz.plotQuadraticSpline(tensor)
tensor = tensor / 512
tensor=tensor.flatten()
tensor=tensor.tolist()
#copy list to clipboard
pyperclip.copy(str(tensor))



nuke.scriptClose()