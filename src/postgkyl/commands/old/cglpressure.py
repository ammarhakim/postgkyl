import click
import numpy as np

from postgkyl.commands import tm
from postgkyl.tools.stack import pushStack, peakStack, antiSqueeze, addStack
from postgkyl.utils import verb_print



def getParPerp(pij, B):
  tmp = np.copy(pij[..., 0:2])

  pxx = pij[..., 0]
  pxy = pij[..., 1]
  pxz = pij[..., 2]
  pyy = pij[..., 3]
  pyz = pij[..., 4]
  pzz = pij[..., 5]

  b = np.sqrt(B[..., 0] * B[..., 0] + B[..., 1] * B[..., 1] + B[..., 2] * B[..., 2])
  bx = B[..., 0] / b
  by = B[..., 1] / b
  bz = B[..., 2] / b

  tmp[..., 0] = (
      bx * bx * pxx
      + by * by * pyy
      + bz * bz * pzz
      + 2.0 * (bx * by * pxy + bx * bz * pxz + by * bz * pyz)
  )
  tmp[..., 1] = (pxx + pyy + pzz - tmp[..., 0]) / 2.0

  return tmp


def getAgyro(pij, B):
  tmp = np.copy(pij[..., 0:6])

  pxx = pij[..., 0]
  pxy = pij[..., 1]
  pxz = pij[..., 2]
  pyy = pij[..., 3]
  pyz = pij[..., 4]
  pzz = pij[..., 5]

  b = np.sqrt(B[..., 0] * B[..., 0] + B[..., 1] * B[..., 1] + B[..., 2] * B[..., 2])
  bx = B[..., 0] / b
  by = B[..., 1] / b
  bz = B[..., 2] / b

  ppar = (
      bx * bx * pxx
      + by * by * pyy
      + bz * bz * pzz
      + 2.0 * (bx * by * pxy + bx * bz * pxz + by * bz * pyz)
  )
  pper = (pxx + pyy + pzz - ppar) / 2.0

  tmp[..., 0] = pxx - (ppar * bx * bx + pper * (1 - bx * bx))  # xx
  tmp[..., 1] = pxy - (ppar * bx * by + pper * (0 - bx * by))  # xy
  tmp[..., 2] = pxz - (ppar * bx * bz + pper * (0 - bx * bz))  # xz
  tmp[..., 3] = pyy - (ppar * by * by + pper * (1 - by * by))  # yy
  tmp[..., 4] = pyz - (ppar * by * bz + pper * (0 - by * bz))  # yz
  tmp[..., 5] = pzz - (ppar * bz * bz + pper * (1 - bz * bz))  # zz

  return tmp


@click.command()
@click.option(
    "--agyro",
    is_flag=True,
    default=False,
    help="Compute the agyrotropic part of pressure tensor instead",
)
@click.pass_context
def cglpressure(ctx, **inputs):
  """Extract parallel and perpendicular pressures from pressure-tensor
  and magnetic field. Pressure-tensor must be the first dataset and
  magnetic field the second dataset. A two component field
  (parallel, perpendicular) is returned. Optionally, the command can
  extract the six components of the agyrotropic part of the pressure
  tensor.

  """
  verb_print(ctx, "Starting CGL pressure")

  coords, pij = peakStack(ctx, ctx.obj["sets"][0])
  coords, B = peakStack(ctx, ctx.obj["sets"][1])

  if inputs["agyro"]:
    tmp = getAgyro(pij, B)
  else:
    tmp = getParPerp(pij, B)

  tmp = antiSqueeze(coords, tmp)

  idx = addStack(ctx)
  ctx.obj["type"].append("hist")
  pushStack(ctx, idx, coords, tmp, "CGL")
  ctx.obj["sets"] = [idx]

  verb_print(ctx, "Finishing CGL pressure")
