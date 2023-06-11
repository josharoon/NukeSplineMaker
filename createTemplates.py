#import nuke
import numpy as np
import torch
import viz
from rotoshapes import shapeFromRotopaint
import pyperclip
from skimage.util import view_as_windows


topology = [15, 4, 4]


def bezier_to_polygonal(x):
    x = np.reshape(x, [-1, 2])
    x = view_as_windows(x,2, step=3)
    return x.reshape([-1])


def inverse_apply_templates(expanded_curves, topology):
    splits = [4 * n_curves for n_curves in topology]
    loops = np.split(expanded_curves[:4 * sum(topology)], [sum(splits[:i]) for i in range(1, len(splits))])
    template_loops = []
    for loop, n_curves in zip(loops, topology):
        if loop.shape[0]>0:
            polygonal_bezier = bezier_to_polygonal(loop)
            template_loops.append(polygonal_bezier)
    if len(template_loops) > 1:
        template = np.concatenate(template_loops, axis=0)
    else:
        template = template_loops[0]

    # Pad with 0.5s if the number of points is less than 4*sum(topology)
    padding_count = 4 * sum(topology) - len(template)
    if padding_count > 0:
        template = np.concatenate((template, [0.5] * padding_count))

    return list(template)


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




def prepRotoTemplate(rotoNode, nCurves=15):
    
    shape = shapeFromRotopaint(rotoNode)
    curves = shape.convert_to_bezier_format(shape.npCtrlPoints)
    # convert list of 4x2 numpy arrays to tensor
    tensor = torch.zeros(len(curves), 4, 2)
    for i in range(len(curves)):
        tensor[i] = torch.from_numpy(curves[i])
    viz.plotCubicSpline(tensor)
    reduced_degree_curves = shape.reduceDegree()
    # bezier curve format needs to be converted back to tensor
    tensor = torch.zeros(len(reduced_degree_curves), 3, 2)
    for i in range(len(reduced_degree_curves)):
        tensor[i] = torch.from_numpy(reduced_degree_curves[i].nodes.T)
    tensor = ensure_continuous(tensor)
    # add dimension for num shapes
    tensor = tensor.unsqueeze(0)
    viz.plotQuadraticSpline(tensor)
    subCurves = shape.subdivideCurves(reduced_degree_curves, nCurves=nCurves)
    tensor = torch.zeros(len(reduced_degree_curves), 3, 2)
    for i in range(len(reduced_degree_curves)):
        tensor[i] = torch.from_numpy(reduced_degree_curves[i].nodes.T)
    tensor = ensure_continuous(tensor)
    # add dimension for num shapes
    tensor = tensor.unsqueeze(0)
    # flatten tensor, turn into list and copy to clipboard
    viz.plotQuadraticSpline(tensor)
    tensor = tensor / 512  # normalise the control points to be in the range [0, 1]

    return tensor

nukescript=r"D:/DeepParametricShapes/nukeScripts/templates.nk"
nuke.scriptOpen(nukescript)
rotonode=nuke.toNode("pupil2")
rotonode2=nuke.toNode("Instrument2")
tensor=prepRotoTemplate(rotonode,nCurves=4)
tensor2=prepRotoTemplate(rotonode2,nCurves=15)
tensor=torch.cat([tensor2,tensor],dim=1)
nuke.scriptClose()


# now convert to template format and test


reshapeNuke = torch.tensor(tensor).reshape(-1, 3, 2)
template =inverse_apply_templates(reshapeNuke.numpy(),topology)#45x2=90
pyperclip.copy(str(list(template)))
