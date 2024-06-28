import numpy
from sympy import *

from optparse import OptionParser


def createInterpMatrix(dim, order, basis_type, interp, modal=True, c2p=False):
  if c2p:
    interp += 1
  # end
  interpList = numpy.zeros(interp)
  for i in range(interp):
    if c2p:
      interpList[i] = -1.0 + float(i) * 2.0 / (interp - 1)
    else:
      interpList[i] = -1.0 * (interp - 1) / interp + float(i) * 2.0 / interp
    # end
  # end

  # The following is for gkhybrid only.
  interpListND = list()
  for d in range(dim):
    interp_true = interp
    if basis_type == "gkhybrid":
      # 1x1v, 1x2v, 2x2v, 3x2v cases, with p=2 in the first velocity dim.
      if (
          ((dim == 2 or dim == 3) and d == 1)
          or (dim == 4 and d == 2)
          or (dim == 5 and d == 3)
      ):
        interp_true = interp + 1
      # end
    elif basis_type == "hybrid":
      # 1x1v, 2x2v, 2x2v, 3x2v cases, with p=2 in the first velocity dim.
      if d == dim - 1:
        interp_true = interp + 1
      # end
    # end

    interpListND.append(numpy.zeros(interp_true))
    for i in range(interp_true):
      if c2p:
        interpListND[d][i] = -1.0 + float(i) * 2.0 / (interp_true - 1)
      else:
        interpListND[d][i] = (
            -1.0 * (interp_true - 1) / interp_true + float(i) * 2.0 / interp_true
        )
      # end
    # end
  # end

  if dim == 1:
    x = Symbol("x")
    if modal:
      if order == 0:
        functionVector = Matrix([[0.7071067811865468]])
        interpMatrix = numpy.zeros((interpList.shape[0], functionVector.shape[0]))
        for i in range(0, interpList.shape[0]):
          for j in range(0, functionVector.shape[0]):
            interpMatrix[i, j] = functionVector[j].subs(x, interpList[i])
          # end
        # end
      elif order == 1:
        functionVector = Matrix([[0.7071067811865468], [1.224744871391589 * x]])
        interpMatrix = numpy.zeros((interpList.shape[0], functionVector.shape[0]))
        for i in range(0, interpList.shape[0]):
          for j in range(0, functionVector.shape[0]):
            interpMatrix[i, j] = functionVector[j].subs(x, interpList[i])
          # end
        # end
      elif order == 2:
        functionVector = Matrix(
            [
                [0.7071067811865468],
                [1.224744871391589 * x],
                [2.371708245126285 * x**2 - 0.7905694150420951],
            ]
        )
        interpMatrix = numpy.zeros((interpList.shape[0], functionVector.shape[0]))
        for i in range(0, interpList.shape[0]):
          for j in range(0, functionVector.shape[0]):
            interpMatrix[i, j] = functionVector[j].subs(x, interpList[i])
          # end
        # end
      elif order == 3:
        functionVector = Matrix(
            [
                [0.7071067811865468],
                [1.224744871391589 * x],
                [2.371708245126285 * x**2 - 0.7905694150420951],
                [4.677071733467427 * x**3 - 2.806243040080457 * x],
            ]
        )
        interpMatrix = numpy.zeros((interpList.shape[0], functionVector.shape[0]))
        for i in range(0, interpList.shape[0]):
          for j in range(0, functionVector.shape[0]):
            interpMatrix[i, j] = functionVector[j].subs(x, interpList[i])
          # end
        # end
      elif order == 4:
        functionVector = Matrix(
            [
                [0.7071067811865468],
                [1.224744871391589 * x],
                [2.371708245126285 * x**2 - 0.7905694150420951],
                [4.677071733467427 * x**3 - 2.806243040080457 * x],
                [
                    9.280776503073431 * x**4
                    - 7.954951288348656 * x**2
                    + 0.7954951288348655
                ],
            ]
        )
        interpMatrix = numpy.zeros((interpList.shape[0], functionVector.shape[0]))
        for i in range(0, interpList.shape[0]):
          for j in range(0, functionVector.shape[0]):
            interpMatrix[i, j] = functionVector[j].subs(x, interpList[i])
          # end
        # end
      else:
        raise NameError(
            "interpMatrix: Order {} is not supported!\nPolynomial order must be <5".format(
                order
            )
        )
      # end
    else:
      if order == 1:
        functionVector = Matrix([[0.5 - 0.5 * x], [0.5 + 0.5 * x]])
        interpMatrix = numpy.zeros((interpList.shape[0], functionVector.shape[0]))
        for i in range(0, interpList.shape[0]):
          for j in range(0, functionVector.shape[0]):
            interpMatrix[i, j] = functionVector[j].subs(x, interpList[i])
          # end
        # end
      elif order == 2:
        functionVector = Matrix(
            [[0.5 * x**2 - 0.5 * x], [1.0 - x**2], [0.5 * x**2 + 0.5 * x]]
        )
        interpMatrix = numpy.zeros((interpList.shape[0], functionVector.shape[0]))
        for i in range(0, interpList.shape[0]):
          for j in range(0, functionVector.shape[0]):
            interpMatrix[i, j] = functionVector[j].subs(x, interpList[i])
          # end
        # end
      elif order == 3:
        functionVector = Matrix(
            [
                [-(9.0 * x**3) / 16.0 + (9.0 * x**2) / 16.0 + x / 16.0 - 1 / 16.0],
                [
                    (27.0 * x**3) / 16.0
                    - (9.0 * x**2) / 16.0
                    - (27.0 * x) / 16.0
                    + 9.0 / 16.0
                ],
                [
                    (27.0 * x) / 16.0
                    - (9.0 * x**2) / 16.0
                    - (27.0 * x**3) / 16.0
                    + 9.0 / 16.0
                ],
                [(9.0 * x**3) / 16.0 + (9.0 * x**2) / 16.0 - x / 16.0 - 1 / 16.0],
            ]
        )
        interpMatrix = numpy.zeros((interpList.shape[0], functionVector.shape[0]))
        for i in range(0, interpList.shape[0]):
          for j in range(0, functionVector.shape[0]):
            interpMatrix[i, j] = functionVector[j].subs(x, interpList[i])
          # end
        # end
      elif order == 4:
        functionVector = Matrix(
            [
                [(2.0 * x**4) / 3.0 - (2.0 * x**3) / 3.0 - x**2 / 6.0 + x / 6.0],
                [
                    -(8.0 * x**4) / 3.0
                    + (4.0 * x**3) / 3.0
                    + (8.0 * x**2) / 3.0
                    - (4.0 * x) / 3.0
                ],
                [4.0 * x**4 - 5.0 * x**2 + 1.0],
                [
                    -(8.0 * x**4) / 3.0
                    - (4.0 * x**3) / 3.0
                    + (8.0 * x**2) / 3.0
                    + (4.0 * x) / 3.0
                ],
                [(2.0 * x**4) / 3.0 + (2.0 * x**3) / 3.0 - x**2 / 6.0 - x / 6.0],
            ]
        )
        interpMatrix = numpy.zeros((interpList.shape[0], functionVector.shape[0]))
        for i in range(0, interpList.shape[0]):
          for j in range(0, functionVector.shape[0]):
            interpMatrix[i, j] = functionVector[j].subs(x, interpList[i])
          # end
        # end
      else:
        raise NameError(
            "interpMatrix: Order {} is not supported!\nPolynomial order must be <5".format(
                order
            )
        )
      # end
    # end
  elif dim == 2:
    x = Symbol("x")
    y = Symbol("y")
    if modal and basis_type == "maximal-order":
      if order == 1:
        functionVector = Matrix(
            [[0.5], [0.8660254037844385 * x], [0.8660254037844385 * y]]
        )
        interpMatrix = numpy.zeros(
            (interpList.shape[0] * interpList.shape[0], functionVector.shape[0])
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, functionVector.shape[0]):
              interpMatrix[j + i * interpList.shape[0], k] = (
                  functionVector[k].subs(x, interpList[j]).subs(y, interpList[i])
              )

      elif order == 2:
        functionVector = Matrix(
            [
                [0.5],
                [0.8660254037844385 * x],
                [0.8660254037844385 * y],
                [1.5 * x * y],
                [1.677050983124845 * x**2 - 0.5590169943749485],
                [1.677050983124845 * y**2 - 0.5590169943749485],
            ]
        )
        interpMatrix = numpy.zeros(
            (interpList.shape[0] * interpList.shape[0], functionVector.shape[0])
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, functionVector.shape[0]):
              interpMatrix[j + i * interpList.shape[0], k] = (
                  functionVector[k].subs(x, interpList[j]).subs(y, interpList[i])
              )

      elif order == 3:
        functionVector = Matrix(
            [
                [0.5],
                [0.8660254037844385 * x],
                [0.8660254037844385 * y],
                [1.5 * x * y],
                [1.677050983124845 * x**2 - 0.5590169943749485],
                [1.677050983124845 * y**2 - 0.5590169943749485],
                [2.904737509655563 * x**2 * y - 0.9682458365518544 * y],
                [2.904737509655563 * x * y**2 - 0.9682458365518544 * x],
                [3.307189138830737 * x**3 - 1.984313483298442 * x],
                [3.307189138830737 * y**3 - 1.984313483298442 * y],
            ]
        )
        interpMatrix = numpy.zeros(
            (interpList.shape[0] * interpList.shape[0], functionVector.shape[0])
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, functionVector.shape[0]):
              interpMatrix[j + i * interpList.shape[0], k] = (
                  functionVector[k].subs(x, interpList[j]).subs(y, interpList[i])
              )

      elif order == 4:
        functionVector = Matrix(
            [
                [0.5],
                [0.8660254037844385 * x],
                [0.8660254037844385 * y],
                [1.5 * x * y],
                [1.677050983124845 * x**2 - 0.5590169943749485],
                [1.677050983124845 * y**2 - 0.5590169943749485],
                [2.904737509655563 * x**2 * y - 0.9682458365518544 * y],
                [2.904737509655563 * x * y**2 - 0.9682458365518544 * x],
                [3.307189138830737 * x**3 - 1.984313483298442 * x],
                [3.307189138830737 * y**3 - 1.984313483298442 * y],
                [5.625 * x**2 * y**2 - 1.875 * y**2 - 1.875 * x**2 + 0.625],
                [5.728219618694792 * x**3 * y - 3.436931771216875 * x * y],
                [5.728219618694792 * x * y**3 - 3.436931771216875 * x * y],
                [6.5625 * x**4 - 5.625 * x**2 + 0.5625],
                [6.5625 * y**4 - 5.625 * y**2 + 0.5625],
            ]
        )
        interpMatrix = numpy.zeros(
            (interpList.shape[0] * interpList.shape[0], functionVector.shape[0])
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, functionVector.shape[0]):
              interpMatrix[j + i * interpList.shape[0], k] = (
                  functionVector[k].subs(x, interpList[j]).subs(y, interpList[i])
              )
      else:
        raise NameError(
            "interpMatrix: Order {} is not supported!\nPolynomial order must be <5".format(
                order
            )
        )

    elif modal and basis_type == "serendipity":
      if order == 0:
        functionVector = Matrix([[0.5]])
        interpMatrix = numpy.zeros(
            (interpList.shape[0] * interpList.shape[0], functionVector.shape[0])
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, functionVector.shape[0]):
              interpMatrix[j + i * interpList.shape[0], k] = (
                  functionVector[k].subs(x, interpList[j]).subs(y, interpList[i])
              )
      elif order == 1:
        functionVector = Matrix(
            [[0.5], [0.8660254037844385 * x], [0.8660254037844385 * y], [1.5 * x * y]]
        )
        interpMatrix = numpy.zeros(
            (interpList.shape[0] * interpList.shape[0], functionVector.shape[0])
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, functionVector.shape[0]):
              interpMatrix[j + i * interpList.shape[0], k] = (
                  functionVector[k].subs(x, interpList[j]).subs(y, interpList[i])
              )

      elif order == 2:
        functionVector = Matrix(
            [
                [0.5],
                [0.8660254037844385 * x],
                [0.8660254037844385 * y],
                [1.5 * x * y],
                [1.677050983124845 * x**2 - 0.5590169943749485],
                [1.677050983124845 * y**2 - 0.5590169943749485],
                [2.904737509655563 * x**2 * y - 0.9682458365518544 * y],
                [2.904737509655563 * x * y**2 - 0.9682458365518544 * x],
            ]
        )
        interpMatrix = numpy.zeros(
            (interpList.shape[0] * interpList.shape[0], functionVector.shape[0])
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, functionVector.shape[0]):
              interpMatrix[j + i * interpList.shape[0], k] = (
                  functionVector[k].subs(x, interpList[j]).subs(y, interpList[i])
              )

      elif order == 3:
        functionVector = Matrix(
            [
                [0.5],
                [0.8660254037844385 * x],
                [0.8660254037844385 * y],
                [1.5 * x * y],
                [1.677050983124845 * x**2 - 0.5590169943749485],
                [1.677050983124845 * y**2 - 0.5590169943749485],
                [2.904737509655563 * x**2 * y - 0.9682458365518544 * y],
                [2.904737509655563 * x * y**2 - 0.9682458365518544 * x],
                [3.307189138830737 * x**3 - 1.984313483298442 * x],
                [3.307189138830737 * y**3 - 1.984313483298442 * y],
                [5.728219618694792 * x**3 * y - 3.436931771216875 * x * y],
                [5.728219618694792 * x * y**3 - 3.436931771216875 * x * y],
            ]
        )
        interpMatrix = numpy.zeros(
            (interpList.shape[0] * interpList.shape[0], functionVector.shape[0])
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, functionVector.shape[0]):
              interpMatrix[j + i * interpList.shape[0], k] = (
                  functionVector[k].subs(x, interpList[j]).subs(y, interpList[i])
              )

      elif order == 4:
        functionVector = Matrix(
            [
                [0.5],
                [0.8660254037844385 * x],
                [0.8660254037844385 * y],
                [1.5 * x * y],
                [1.677050983124845 * x**2 - 0.5590169943749485],
                [1.677050983124845 * y**2 - 0.5590169943749485],
                [2.904737509655563 * x**2 * y - 0.9682458365518544 * y],
                [2.904737509655563 * x * y**2 - 0.9682458365518544 * x],
                [3.307189138830737 * x**3 - 1.984313483298442 * x],
                [3.307189138830737 * y**3 - 1.984313483298442 * y],
                [5.625 * x**2 * y**2 - 1.875 * y**2 - 1.875 * x**2 + 0.625],
                [5.728219618694792 * x**3 * y - 3.436931771216875 * x * y],
                [5.728219618694792 * x * y**3 - 3.436931771216875 * x * y],
                [6.5625 * x**4 - 5.625 * x**2 + 0.5625],
                [6.5625 * y**4 - 5.625 * y**2 + 0.5625],
                [
                    11.36658342467074 * x**4 * y
                    - 9.74278579257492 * x**2 * y
                    + 0.9742785792574921 * y
                ],
                [
                    11.36658342467074 * x * y**4
                    - 9.74278579257492 * x * y**2
                    + 0.9742785792574921 * x
                ],
            ]
        )
        interpMatrix = numpy.zeros(
            (interpList.shape[0] * interpList.shape[0], functionVector.shape[0])
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, functionVector.shape[0]):
              interpMatrix[j + i * interpList.shape[0], k] = (
                  functionVector[k].subs(x, interpList[j]).subs(y, interpList[i])
              )
      else:
        raise NameError(
            "interpMatrix: Order {} is not supported!\nPolynomial order must be <5".format(
                order
            )
        )

    elif modal and basis_type == "tensor":
      if order == 1:
        functionVector = Matrix(
            [[0.5], [0.8660254037844385 * x], [0.8660254037844385 * y], [1.5 * x * y]]
        )
        interpMatrix = numpy.zeros(
            (interpList.shape[0] * interpList.shape[0], functionVector.shape[0])
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, functionVector.shape[0]):
              interpMatrix[j + i * interpList.shape[0], k] = (
                  functionVector[k].subs(x, interpList[j]).subs(y, interpList[i])
              )

      elif order == 2:
        functionVector = Matrix(
            [
                [0.5],
                [0.8660254037844385 * x],
                [0.8660254037844385 * y],
                [1.5 * x * y],
                [1.677050983124845 * x**2 - 0.5590169943749485],
                [1.677050983124845 * y**2 - 0.5590169943749485],
                [2.904737509655563 * x**2 * y - 0.9682458365518544 * y],
                [2.904737509655563 * x * y**2 - 0.9682458365518544 * x],
                [5.625 * x**2 * y**2 - 1.875 * y**2 - 1.875 * x**2 + 0.625],
            ]
        )
        interpMatrix = numpy.zeros(
            (interpList.shape[0] * interpList.shape[0], functionVector.shape[0])
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, functionVector.shape[0]):
              interpMatrix[j + i * interpList.shape[0], k] = (
                  functionVector[k].subs(x, interpList[j]).subs(y, interpList[i])
              )

      elif order == 3:
        functionVector = Matrix(
            [
                [0.5],
                [0.8660254037844385 * x],
                [0.8660254037844385 * y],
                [1.5 * x * y],
                [1.677050983124845 * x**2 - 0.5590169943749485],
                [1.677050983124845 * y**2 - 0.5590169943749485],
                [2.904737509655563 * x**2 * y - 0.9682458365518544 * y],
                [2.904737509655563 * x * y**2 - 0.9682458365518544 * x],
                [3.307189138830737 * x**3 - 1.984313483298442 * x],
                [3.307189138830737 * y**3 - 1.984313483298442 * y],
                [5.625 * x**2 * y**2 - 1.875 * y**2 - 1.875 * x**2 + 0.625],
                [5.728219618694792 * x**3 * y - 3.436931771216875 * x * y],
                [5.728219618694792 * x * y**3 - 3.436931771216875 * x * y],
                [
                    11.09264959331178 * x**3 * y**2
                    - 6.655589755987068 * x * y**2
                    - 3.69754986443726 * x**3
                    + 2.218529918662355 * x
                ],
                [
                    11.09264959331178 * x**2 * y**3
                    - 3.69754986443726 * y**3
                    - 6.655589755987068 * x**2 * y
                    + 2.218529918662355 * y
                ],
                [
                    21.875 * x**3 * y**3
                    - 13.125 * x * y**3
                    - 13.125 * x**3 * y
                    + 7.875 * x * y
                ],
            ]
        )
        interpMatrix = numpy.zeros(
            (interpList.shape[0] * interpList.shape[0], functionVector.shape[0])
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, functionVector.shape[0]):
              interpMatrix[j + i * interpList.shape[0], k] = (
                  functionVector[k].subs(x, interpList[j]).subs(y, interpList[i])
              )
      else:
        raise NameError(
            "interpMatrix: Order {} is not supported!\nPolynomial order must be <4".format(
                order
            )
        )

    elif modal == False and basis_type == "serendipity":
      if order == 1:
        functionVector = Matrix(
            [
                [(x * y) / 4.0 - y / 4.0 - x / 4.0 + 1.0 / 4.0],
                [x / 4.0 - y / 4.0 - (x * y) / 4.0 + 1.0 / 4.0],
                [y / 4.0 - x / 4.0 - (x * y) / 4.0 + 1.0 / 4.0],
                [x / 4.0 + y / 4.0 + (x * y) / 4.0 + 1.0 / 4.0],
            ]
        )
        interpMatrix = numpy.zeros(
            (interpList.shape[0] * interpList.shape[0], functionVector.shape[0])
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, functionVector.shape[0]):
              interpMatrix[j + i * interpList.shape[0], k] = (
                  functionVector[k].subs(x, interpList[j]).subs(y, interpList[i])
              )

      elif order == 2:
        functionVector = Matrix(
            [
                [
                    -(x**2 * y) / 4.0
                    + x**2 / 4.0
                    - (x * y**2) / 4.0
                    + (x * y) / 4.0
                    + y**2 / 4.0
                    - 1 / 4.0
                ],
                [(x**2 * y) / 2.0 - y / 2.0 - x**2 / 2.0 + 1.0 / 2.0],
                [
                    -(x**2 * y) / 4.0
                    + x**2 / 4.0
                    + (x * y**2) / 4.0
                    - (x * y) / 4.0
                    + y**2 / 4.0
                    - 1 / 4.0
                ],
                [(x * y**2) / 2.0 - x / 2.0 - y**2 / 2.0 + 1 / 2.0],
                [x / 2.0 - (x * y**2) / 2.0 - y**2 / 2.0 + 1.0 / 2.0],
                [
                    (x**2 * y) / 4.0
                    + x**2 / 4.0
                    - (x * y**2) / 4.0
                    - (x * y) / 4.0
                    + y**2 / 4.0
                    - 1 / 4.0
                ],
                [y / 2.0 - (x**2 * y) / 2.0 - x**2 / 2.0 + 1.0 / 2.0],
                [
                    (x**2 * y) / 4.0
                    + x**2 / 4.0
                    + (x * y**2) / 4.0
                    + (x * y) / 4.0
                    + y**2 / 4.0
                    - 1 / 4.0
                ],
            ]
        )
        interpMatrix = numpy.zeros(
            (interpList.shape[0] * interpList.shape[0], functionVector.shape[0])
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, functionVector.shape[0]):
              interpMatrix[j + i * interpList.shape[0], k] = (
                  functionVector[k].subs(x, interpList[j]).subs(y, interpList[i])
              )
      else:
        raise NameError(
            "interpMatrix: Order {} is not supported!\nPolynomial order must be <3 for nodal Serendipity in 2D".format(
                order
            )
        )

    elif modal and basis_type == "gkhybrid":
      if order == 1:
        functionVector = Matrix(
            [
                [0.5],
                [0.8660254037844386 * x],
                [0.8660254037844386 * y],
                [1.5 * x * y],
                [1.677050983124842 * (y**2 - 0.3333333333333333)],
                [2.904737509655563 * (x * y**2 - 0.3333333333333333 * x)],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpListND[0].shape[0] * interpListND[1].shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpListND[1].shape[0]):
          for j in range(0, interpListND[0].shape[0]):
            for k in range(0, functionVector.shape[0]):
              interpMatrix[j + i * interpListND[0].shape[0], k] = (
                  functionVector[k]
                  .subs(x, interpListND[0][j])
                  .subs(y, interpListND[1][i])
              )

      else:
        raise NameError(
            "interpMatrix: Order {} is not supported!\nPolynomial order must be =1".format(
                order
            )
        )

    elif modal and basis_type == "hybrid":
      if order == 1:
        functionVector = Matrix(
            [
                [0.5],
                [0.8660254037844386 * x],
                [0.8660254037844386 * y],
                [1.5 * x * y],
                [1.677050983124842 * (y**2 - 0.3333333333333333)],
                [2.904737509655563 * (x * y**2 - 0.3333333333333333 * x)],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpListND[0].shape[0] * interpListND[1].shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpListND[1].shape[0]):
          for j in range(0, interpListND[0].shape[0]):
            for k in range(0, functionVector.shape[0]):
              interpMatrix[j + i * interpListND[0].shape[0], k] = (
                  functionVector[k]
                  .subs(x, interpListND[0][j])
                  .subs(y, interpListND[1][i])
              )
      else:
        raise NameError(
            "interpMatrix: Order {} is not supported!\nPolynomial order must be =1".format(
                order
            )
        )

    else:
      raise NameError(
          "interpMatrix: Basis {} is not supported!\nSupported basis are currently 'nodal Serendipity', 'modal Serendipity', and 'modal maximal order'".format(
              basis_type
          )
      )
  elif dim == 3:
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")
    if modal and basis_type == "maximal-order":
      if order == 1:
        functionVector = Matrix(
            [
                [0.3535533905932734],
                [0.6123724356957931 * x],
                [0.6123724356957931 * y],
                [0.6123724356957931 * z],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0] * interpList.shape[0] * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, functionVector.shape[0]):
                interpMatrix[
                    k
                    + j * interpList.shape[0]
                    + i * interpList.shape[0] * interpList.shape[0],
                    l,
                ] = (
                    functionVector[l]
                    .subs(x, interpList[k])
                    .subs(y, interpList[j])
                    .subs(z, interpList[i])
                )

      elif order == 2:
        functionVector = Matrix(
            [
                [0.3535533905932734],
                [0.6123724356957931 * x],
                [0.6123724356957931 * y],
                [0.6123724356957931 * z],
                [1.060660171779822 * x * y],
                [1.060660171779822 * x * z],
                [1.060660171779822 * y * z],
                [1.185854122563141 * x**2 - 0.3952847075210471],
                [1.185854122563141 * y**2 - 0.3952847075210471],
                [1.185854122563141 * z**2 - 0.3952847075210471],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0] * interpList.shape[0] * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, functionVector.shape[0]):
                interpMatrix[
                    k
                    + j * interpList.shape[0]
                    + i * interpList.shape[0] * interpList.shape[0],
                    l,
                ] = (
                    functionVector[l]
                    .subs(x, interpList[k])
                    .subs(y, interpList[j])
                    .subs(z, interpList[i])
                )

      elif order == 3:
        functionVector = Matrix(
            [
                [0.3535533905932734],
                [0.6123724356957931 * x],
                [0.6123724356957931 * y],
                [0.6123724356957931 * z],
                [1.060660171779822 * x * y],
                [1.060660171779822 * x * z],
                [1.060660171779822 * y * z],
                [1.185854122563141 * x**2 - 0.3952847075210471],
                [1.185854122563141 * y**2 - 0.3952847075210471],
                [1.185854122563141 * z**2 - 0.3952847075210471],
                [1.837117307087383 * x * y * z],
                [2.053959590644373 * x**2 * y - 0.6846531968814578 * y],
                [2.053959590644373 * x * y**2 - 0.6846531968814578 * x],
                [2.053959590644373 * x**2 * z - 0.6846531968814578 * z],
                [2.053959590644373 * y**2 * z - 0.6846531968814578 * z],
                [2.053959590644373 * x * z**2 - 0.6846531968814578 * x],
                [2.053959590644373 * y * z**2 - 0.6846531968814578 * y],
                [2.338535866733713 * x**3 - 1.403121520040228 * x],
                [2.338535866733713 * y**3 - 1.403121520040228 * y],
                [2.338535866733713 * z**3 - 1.403121520040228 * z],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0] * interpList.shape[0] * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, functionVector.shape[0]):
                interpMatrix[
                    k
                    + j * interpList.shape[0]
                    + i * interpList.shape[0] * interpList.shape[0],
                    l,
                ] = (
                    functionVector[l]
                    .subs(x, interpList[k])
                    .subs(y, interpList[j])
                    .subs(z, interpList[i])
                )

      elif order == 4:
        functionVector = Matrix(
            [
                [0.3535533905932734],
                [0.6123724356957931 * x],
                [0.6123724356957931 * y],
                [0.6123724356957931 * z],
                [1.060660171779822 * x * y],
                [1.060660171779822 * x * z],
                [1.060660171779822 * y * z],
                [1.185854122563141 * x**2 - 0.3952847075210471],
                [1.185854122563141 * y**2 - 0.3952847075210471],
                [1.185854122563141 * z**2 - 0.3952847075210471],
                [1.837117307087383 * x * y * z],
                [2.053959590644373 * x**2 * y - 0.6846531968814578 * y],
                [2.053959590644373 * x * y**2 - 0.6846531968814578 * x],
                [2.053959590644373 * x**2 * z - 0.6846531968814578 * z],
                [2.053959590644373 * y**2 * z - 0.6846531968814578 * z],
                [2.053959590644373 * x * z**2 - 0.6846531968814578 * x],
                [2.053959590644373 * y * z**2 - 0.6846531968814578 * y],
                [2.338535866733713 * x**3 - 1.403121520040228 * x],
                [2.338535866733713 * y**3 - 1.403121520040228 * y],
                [2.338535866733713 * z**3 - 1.403121520040228 * z],
                [3.557562367689424 * x**2 * y * z - 1.185854122563141 * y * z],
                [3.557562367689424 * x * y**2 * z - 1.185854122563141 * x * z],
                [3.557562367689424 * x * y * z**2 - 1.185854122563141 * x * y],
                [
                    3.977475644174331 * x**2 * y**2
                    - 1.325825214724777 * y**2
                    - 1.325825214724777 * x**2
                    + 0.4419417382415923
                ],
                [
                    3.977475644174331 * x**2 * z**2
                    - 1.325825214724777 * z**2
                    - 1.325825214724777 * x**2
                    + 0.4419417382415923
                ],
                [
                    3.977475644174331 * y**2 * z**2
                    - 1.325825214724777 * z**2
                    - 1.325825214724777 * y**2
                    + 0.4419417382415923
                ],
                [4.050462936504911 * x**3 * y - 2.430277761902947 * x * y],
                [4.050462936504911 * x * y**3 - 2.430277761902947 * x * y],
                [4.050462936504911 * x**3 * z - 2.430277761902947 * x * z],
                [4.050462936504911 * y**3 * z - 2.430277761902947 * y * z],
                [4.050462936504911 * x * z**3 - 2.430277761902947 * x * z],
                [4.050462936504911 * y * z**3 - 2.430277761902947 * y * z],
                [
                    4.640388251536713 * x**4
                    - 3.977475644174326 * x**2
                    + 0.3977475644174325
                ],
                [
                    4.640388251536713 * y**4
                    - 3.977475644174326 * y**2
                    + 0.3977475644174325
                ],
                [
                    4.640388251536713 * z**4
                    - 3.977475644174326 * z**2
                    + 0.3977475644174325
                ],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0] * interpList.shape[0] * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, functionVector.shape[0]):
                interpMatrix[
                    k
                    + j * interpList.shape[0]
                    + i * interpList.shape[0] * interpList.shape[0],
                    l,
                ] = (
                    functionVector[l]
                    .subs(x, interpList[k])
                    .subs(y, interpList[j])
                    .subs(z, interpList[i])
                )
      else:
        raise NameError(
            "interpMatrix: Order {} is not supported!\nPolynomial order must be <5".format(
                order
            )
        )

    elif modal and basis_type == "serendipity":
      if order == 0:
        functionVector = Matrix([[0.3535533905932734]])
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0] * interpList.shape[0] * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, functionVector.shape[0]):
                interpMatrix[
                    k
                    + j * interpList.shape[0]
                    + i * interpList.shape[0] * interpList.shape[0],
                    l,
                ] = (
                    functionVector[l]
                    .subs(x, interpList[k])
                    .subs(y, interpList[j])
                    .subs(z, interpList[i])
                )
      elif order == 1:
        functionVector = Matrix(
            [
                [0.3535533905932734],
                [0.6123724356957931 * x],
                [0.6123724356957931 * y],
                [0.6123724356957931 * z],
                [1.060660171779822 * x * y],
                [1.060660171779822 * x * z],
                [1.060660171779822 * y * z],
                [1.837117307087383 * x * y * z],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0] * interpList.shape[0] * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, functionVector.shape[0]):
                interpMatrix[
                    k
                    + j * interpList.shape[0]
                    + i * interpList.shape[0] * interpList.shape[0],
                    l,
                ] = (
                    functionVector[l]
                    .subs(x, interpList[k])
                    .subs(y, interpList[j])
                    .subs(z, interpList[i])
                )

      elif order == 2:
        functionVector = Matrix(
            [
                [0.3535533905932734],
                [0.6123724356957931 * x],
                [0.6123724356957931 * y],
                [0.6123724356957931 * z],
                [1.060660171779822 * x * y],
                [1.060660171779822 * x * z],
                [1.060660171779822 * y * z],
                [1.185854122563141 * x**2 - 0.3952847075210471],
                [1.185854122563141 * y**2 - 0.3952847075210471],
                [1.185854122563141 * z**2 - 0.3952847075210471],
                [1.837117307087383 * x * y * z],
                [2.053959590644373 * x**2 * y - 0.6846531968814578 * y],
                [2.053959590644373 * x * y**2 - 0.6846531968814578 * x],
                [2.053959590644373 * x**2 * z - 0.6846531968814578 * z],
                [2.053959590644373 * y**2 * z - 0.6846531968814578 * z],
                [2.053959590644373 * x * z**2 - 0.6846531968814578 * x],
                [2.053959590644373 * y * z**2 - 0.6846531968814578 * y],
                [3.557562367689424 * x**2 * y * z - 1.185854122563141 * y * z],
                [3.557562367689424 * x * y**2 * z - 1.185854122563141 * x * z],
                [3.557562367689424 * x * y * z**2 - 1.185854122563141 * x * y],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0] * interpList.shape[0] * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, functionVector.shape[0]):
                interpMatrix[
                    k
                    + j * interpList.shape[0]
                    + i * interpList.shape[0] * interpList.shape[0],
                    l,
                ] = (
                    functionVector[l]
                    .subs(x, interpList[k])
                    .subs(y, interpList[j])
                    .subs(z, interpList[i])
                )

      elif order == 3:
        functionVector = Matrix(
            [
                [0.3535533905932734],
                [0.6123724356957931 * x],
                [0.6123724356957931 * y],
                [0.6123724356957931 * z],
                [1.060660171779822 * x * y],
                [1.060660171779822 * x * z],
                [1.060660171779822 * y * z],
                [1.185854122563141 * x**2 - 0.3952847075210471],
                [1.185854122563141 * y**2 - 0.3952847075210471],
                [1.185854122563141 * z**2 - 0.3952847075210471],
                [1.837117307087383 * x * y * z],
                [2.053959590644373 * x**2 * y - 0.6846531968814578 * y],
                [2.053959590644373 * x * y**2 - 0.6846531968814578 * x],
                [2.053959590644373 * x**2 * z - 0.6846531968814578 * z],
                [2.053959590644373 * y**2 * z - 0.6846531968814578 * z],
                [2.053959590644373 * x * z**2 - 0.6846531968814578 * x],
                [2.053959590644373 * y * z**2 - 0.6846531968814578 * y],
                [2.338535866733713 * x**3 - 1.403121520040228 * x],
                [2.338535866733713 * y**3 - 1.403121520040228 * y],
                [2.338535866733713 * z**3 - 1.403121520040228 * z],
                [3.557562367689424 * x**2 * y * z - 1.185854122563141 * y * z],
                [3.557562367689424 * x * y**2 * z - 1.185854122563141 * x * z],
                [3.557562367689424 * x * y * z**2 - 1.185854122563141 * x * y],
                [4.050462936504911 * x**3 * y - 2.430277761902947 * x * y],
                [4.050462936504911 * x * y**3 - 2.430277761902947 * x * y],
                [4.050462936504911 * x**3 * z - 2.430277761902947 * x * z],
                [4.050462936504911 * y**3 * z - 2.430277761902947 * y * z],
                [4.050462936504911 * x * z**3 - 2.430277761902947 * x * z],
                [4.050462936504911 * y * z**3 - 2.430277761902947 * y * z],
                [7.015607600201137 * x**3 * y * z - 4.209364560120682 * x * y * z],
                [7.015607600201137 * x * y**3 * z - 4.209364560120682 * x * y * z],
                [7.015607600201137 * x * y * z**3 - 4.209364560120682 * x * y * z],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0] * interpList.shape[0] * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, functionVector.shape[0]):
                interpMatrix[
                    k
                    + j * interpList.shape[0]
                    + i * interpList.shape[0] * interpList.shape[0],
                    l,
                ] = (
                    functionVector[l]
                    .subs(x, interpList[k])
                    .subs(y, interpList[j])
                    .subs(z, interpList[i])
                )

      elif order == 4:
        functionVector = Matrix(
            [
                [0.3535533905932734],
                [0.6123724356957931 * x],
                [0.6123724356957931 * y],
                [0.6123724356957931 * z],
                [1.060660171779822 * x * y],
                [1.060660171779822 * x * z],
                [1.060660171779822 * y * z],
                [1.185854122563141 * x**2 - 0.3952847075210471],
                [1.185854122563141 * y**2 - 0.3952847075210471],
                [1.185854122563141 * z**2 - 0.3952847075210471],
                [1.837117307087383 * x * y * z],
                [2.053959590644373 * x**2 * y - 0.6846531968814578 * y],
                [2.053959590644373 * x * y**2 - 0.6846531968814578 * x],
                [2.053959590644373 * x**2 * z - 0.6846531968814578 * z],
                [2.053959590644373 * y**2 * z - 0.6846531968814578 * z],
                [2.053959590644373 * x * z**2 - 0.6846531968814578 * x],
                [2.053959590644373 * y * z**2 - 0.6846531968814578 * y],
                [2.338535866733713 * x**3 - 1.403121520040228 * x],
                [2.338535866733713 * y**3 - 1.403121520040228 * y],
                [2.338535866733713 * z**3 - 1.403121520040228 * z],
                [3.557562367689424 * x**2 * y * z - 1.185854122563141 * y * z],
                [3.557562367689424 * x * y**2 * z - 1.185854122563141 * x * z],
                [3.557562367689424 * x * y * z**2 - 1.185854122563141 * x * y],
                [
                    3.977475644174331 * x**2 * y**2
                    - 1.325825214724777 * y**2
                    - 1.325825214724777 * x**2
                    + 0.4419417382415923
                ],
                [
                    3.977475644174331 * x**2 * z**2
                    - 1.325825214724777 * z**2
                    - 1.325825214724777 * x**2
                    + 0.4419417382415923
                ],
                [
                    3.977475644174331 * y**2 * z**2
                    - 1.325825214724777 * z**2
                    - 1.325825214724777 * y**2
                    + 0.4419417382415923
                ],
                [4.050462936504911 * x**3 * y - 2.430277761902947 * x * y],
                [4.050462936504911 * x * y**3 - 2.430277761902947 * x * y],
                [4.050462936504911 * x**3 * z - 2.430277761902947 * x * z],
                [4.050462936504911 * y**3 * z - 2.430277761902947 * y * z],
                [4.050462936504911 * x * z**3 - 2.430277761902947 * x * z],
                [4.050462936504911 * y * z**3 - 2.430277761902947 * y * z],
                [
                    4.640388251536713 * x**4
                    - 3.977475644174326 * x**2
                    + 0.3977475644174325
                ],
                [
                    4.640388251536713 * y**4
                    - 3.977475644174326 * y**2
                    + 0.3977475644174325
                ],
                [
                    4.640388251536713 * z**4
                    - 3.977475644174326 * z**2
                    + 0.3977475644174325
                ],
                [
                    6.889189901577672 * x**2 * y**2 * z
                    - 2.296396633859224 * y**2 * z
                    - 2.296396633859224 * x**2 * z
                    + 0.7654655446197414 * z
                ],
                [
                    6.889189901577672 * x**2 * y * z**2
                    - 2.296396633859224 * y * z**2
                    - 2.296396633859224 * x**2 * y
                    + 0.7654655446197414 * y
                ],
                [
                    6.889189901577672 * x * y**2 * z**2
                    - 2.296396633859224 * x * z**2
                    - 2.296396633859224 * x * y**2
                    + 0.7654655446197414 * x
                ],
                [7.015607600201137 * x**3 * y * z - 4.209364560120682 * x * y * z],
                [7.015607600201137 * x * y**3 * z - 4.209364560120682 * x * y * z],
                [7.015607600201137 * x * y * z**3 - 4.209364560120682 * x * y * z],
                [
                    8.03738821850729 * x**4 * y
                    - 6.889189901577677 * x**2 * y
                    + 0.6889189901577677 * y
                ],
                [
                    8.03738821850729 * x * y**4
                    - 6.889189901577677 * x * y**2
                    + 0.6889189901577677 * x
                ],
                [
                    8.03738821850729 * x**4 * z
                    - 6.889189901577677 * x**2 * z
                    + 0.6889189901577677 * z
                ],
                [
                    8.03738821850729 * y**4 * z
                    - 6.889189901577677 * y**2 * z
                    + 0.6889189901577677 * z
                ],
                [
                    8.03738821850729 * x * z**4
                    - 6.889189901577677 * x * z**2
                    + 0.6889189901577677 * x
                ],
                [
                    8.03738821850729 * y * z**4
                    - 6.889189901577677 * y * z**2
                    + 0.6889189901577677 * y
                ],
                [
                    13.92116475461014 * x**4 * y * z
                    - 11.93242693252298 * x**2 * y * z
                    + 1.193242693252298 * y * z
                ],
                [
                    13.92116475461014 * x * y**4 * z
                    - 11.93242693252298 * x * y**2 * z
                    + 1.193242693252298 * x * z
                ],
                [
                    13.92116475461014 * x * y * z**4
                    - 11.93242693252298 * x * y * z**2
                    + 1.193242693252298 * x * y
                ],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0] * interpList.shape[0] * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, functionVector.shape[0]):
                interpMatrix[
                    k
                    + j * interpList.shape[0]
                    + i * interpList.shape[0] * interpList.shape[0],
                    l,
                ] = (
                    functionVector[l]
                    .subs(x, interpList[k])
                    .subs(y, interpList[j])
                    .subs(z, interpList[i])
                )
      else:
        raise NameError(
            "interpMatrix: Order {} is not supported!\nPolynomial order must be <5".format(
                order
            )
        )

    elif modal and basis_type == "tensor":
      if order == 1:
        functionVector = Matrix(
            [
                [0.3535533905932734],
                [0.6123724356957931 * x],
                [0.6123724356957931 * y],
                [0.6123724356957931 * z],
                [1.060660171779822 * x * y],
                [1.060660171779822 * x * z],
                [1.060660171779822 * y * z],
                [1.837117307087383 * x * y * z],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0] * interpList.shape[0] * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, functionVector.shape[0]):
                interpMatrix[
                    k
                    + j * interpList.shape[0]
                    + i * interpList.shape[0] * interpList.shape[0],
                    l,
                ] = (
                    functionVector[l]
                    .subs(x, interpList[k])
                    .subs(y, interpList[j])
                    .subs(z, interpList[i])
                )

      elif order == 2:
        functionVector = Matrix(
            [
                [0.3535533905932734],
                [0.6123724356957931 * x],
                [0.6123724356957931 * y],
                [0.6123724356957931 * z],
                [1.060660171779822 * x * y],
                [1.060660171779822 * x * z],
                [1.060660171779822 * y * z],
                [1.185854122563141 * x**2 - 0.3952847075210471],
                [1.185854122563141 * y**2 - 0.3952847075210471],
                [1.185854122563141 * z**2 - 0.3952847075210471],
                [1.837117307087383 * x * y * z],
                [2.053959590644373 * x**2 * y - 0.6846531968814578 * y],
                [2.053959590644373 * x * y**2 - 0.6846531968814578 * x],
                [2.053959590644373 * x**2 * z - 0.6846531968814578 * z],
                [2.053959590644373 * y**2 * z - 0.6846531968814578 * z],
                [2.053959590644373 * x * z**2 - 0.6846531968814578 * x],
                [2.053959590644373 * y * z**2 - 0.6846531968814578 * y],
                [3.557562367689424 * x**2 * y * z - 1.185854122563141 * y * z],
                [3.557562367689424 * x * y**2 * z - 1.185854122563141 * x * z],
                [3.557562367689424 * x * y * z**2 - 1.185854122563141 * x * y],
                [
                    3.977475644174328 * x**2 * y**2
                    - 1.325825214724776 * y**2
                    - 1.325825214724776 * x**2
                    + 0.441941738241592
                ],
                [
                    3.977475644174328 * x**2 * z**2
                    - 1.325825214724776 * z**2
                    - 1.325825214724776 * x**2
                    + 0.441941738241592
                ],
                [
                    3.977475644174328 * y**2 * z**2
                    - 1.325825214724776 * z**2
                    - 1.325825214724776 * y**2
                    + 0.441941738241592
                ],
                [
                    6.889189901577683 * x**2 * y**2 * z
                    - 2.296396633859227 * y**2 * z
                    - 2.296396633859227 * x**2 * z
                    + 0.7654655446197425 * z
                ],
                [
                    6.889189901577683 * x**2 * y * z**2
                    - 2.296396633859227 * y * z**2
                    - 2.296396633859227 * x**2 * y
                    + 0.7654655446197425 * y
                ],
                [
                    6.889189901577683 * x * y**2 * z**2
                    - 2.296396633859227 * x * z**2
                    - 2.296396633859227 * x * y**2
                    + 0.7654655446197425 * x
                ],
                [
                    13.34085887883535 * x**2 * y**2 * z**2
                    - 4.446952959611782 * y**2 * z**2
                    - 4.446952959611782 * x**2 * z**2
                    + 1.482317653203927 * z**2
                    - 4.446952959611782 * x**2 * y**2
                    + 1.482317653203927 * y**2
                    + 1.482317653203927 * x**2
                    - 0.4941058844013091
                ],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0] * interpList.shape[0] * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, functionVector.shape[0]):
                interpMatrix[
                    k
                    + j * interpList.shape[0]
                    + i * interpList.shape[0] * interpList.shape[0],
                    l,
                ] = (
                    functionVector[l]
                    .subs(x, interpList[k])
                    .subs(y, interpList[j])
                    .subs(z, interpList[i])
                )

      elif order == 3:
        functionVector = Matrix(
            [
                [0.3535533905932734],
                [0.6123724356957931 * x],
                [0.6123724356957931 * y],
                [0.6123724356957931 * z],
                [1.060660171779822 * x * y],
                [1.060660171779822 * x * z],
                [1.060660171779822 * y * z],
                [1.185854122563141 * x**2 - 0.3952847075210471],
                [1.185854122563141 * y**2 - 0.3952847075210471],
                [1.185854122563141 * z**2 - 0.3952847075210471],
                [1.837117307087383 * x * y * z],
                [2.053959590644373 * x**2 * y - 0.6846531968814578 * y],
                [2.053959590644373 * x * y**2 - 0.6846531968814578 * x],
                [2.053959590644373 * x**2 * z - 0.6846531968814578 * z],
                [2.053959590644373 * y**2 * z - 0.6846531968814578 * z],
                [2.053959590644373 * x * z**2 - 0.6846531968814578 * x],
                [2.053959590644373 * y * z**2 - 0.6846531968814578 * y],
                [2.338535866733713 * x**3 - 1.403121520040228 * x],
                [2.338535866733713 * y**3 - 1.403121520040228 * y],
                [2.338535866733713 * z**3 - 1.403121520040228 * z],
                [3.557562367689424 * x**2 * y * z - 1.185854122563141 * y * z],
                [3.557562367689424 * x * y**2 * z - 1.185854122563141 * x * z],
                [3.557562367689424 * x * y * z**2 - 1.185854122563141 * x * y],
                [
                    3.977475644174328 * x**2 * y**2
                    - 1.325825214724776 * y**2
                    - 1.325825214724776 * x**2
                    + 0.441941738241592
                ],
                [
                    3.977475644174328 * x**2 * z**2
                    - 1.325825214724776 * z**2
                    - 1.325825214724776 * x**2
                    + 0.441941738241592
                ],
                [
                    3.977475644174328 * y**2 * z**2
                    - 1.325825214724776 * z**2
                    - 1.325825214724776 * y**2
                    + 0.441941738241592
                ],
                [4.050462936504911 * x**3 * y - 2.430277761902947 * x * y],
                [4.050462936504911 * x * y**3 - 2.430277761902947 * x * y],
                [4.050462936504911 * x**3 * z - 2.430277761902947 * x * z],
                [4.050462936504911 * y**3 * z - 2.430277761902947 * y * z],
                [4.050462936504911 * x * z**3 - 2.430277761902947 * x * z],
                [4.050462936504911 * y * z**3 - 2.430277761902947 * y * z],
                [
                    6.889189901577683 * x**2 * y**2 * z
                    - 2.296396633859227 * y**2 * z
                    - 2.296396633859227 * x**2 * z
                    + 0.7654655446197425 * z
                ],
                [
                    6.889189901577683 * x**2 * y * z**2
                    - 2.296396633859227 * y * z**2
                    - 2.296396633859227 * x**2 * y
                    + 0.7654655446197425 * y
                ],
                [
                    6.889189901577683 * x * y**2 * z**2
                    - 2.296396633859227 * x * z**2
                    - 2.296396633859227 * x * y**2
                    + 0.7654655446197425 * x
                ],
                [7.015607600201137 * x**3 * y * z - 4.209364560120682 * x * y * z],
                [7.015607600201137 * x * y**3 * z - 4.209364560120682 * x * y * z],
                [7.015607600201137 * x * y * z**3 - 4.209364560120682 * x * y * z],
                [
                    7.843687748756954 * x**3 * y**2
                    - 4.706212649254172 * x * y**2
                    - 2.614562582918984 * x**3
                    + 1.56873754975139 * x
                ],
                [
                    7.843687748756954 * x**2 * y**3
                    - 2.614562582918984 * y**3
                    - 4.706212649254172 * x**2 * y
                    + 1.56873754975139 * y
                ],
                [
                    7.843687748756954 * x**3 * z**2
                    - 4.706212649254172 * x * z**2
                    - 2.614562582918984 * x**3
                    + 1.56873754975139 * x
                ],
                [
                    7.843687748756954 * y**3 * z**2
                    - 4.706212649254172 * y * z**2
                    - 2.614562582918984 * y**3
                    + 1.56873754975139 * y
                ],
                [
                    7.843687748756954 * x**2 * z**3
                    - 2.614562582918984 * z**3
                    - 4.706212649254172 * x**2 * z
                    + 1.56873754975139 * z
                ],
                [
                    7.843687748756954 * y**2 * z**3
                    - 2.614562582918984 * z**3
                    - 4.706212649254172 * y**2 * z
                    + 1.56873754975139 * z
                ],
                [
                    13.34085887883535 * x**2 * y**2 * z**2
                    - 4.446952959611782 * y**2 * z**2
                    - 4.446952959611782 * x**2 * z**2
                    + 1.482317653203927 * z**2
                    - 4.446952959611782 * x**2 * y**2
                    + 1.482317653203927 * y**2
                    + 1.482317653203927 * x**2
                    - 0.4941058844013091
                ],
                [
                    13.58566569955259 * x**3 * y**2 * z
                    - 8.151399419731556 * x * y**2 * z
                    - 4.528555233184197 * x**3 * z
                    + 2.717133139910518 * x * z
                ],
                [
                    13.58566569955259 * x**2 * y**3 * z
                    - 4.528555233184197 * y**3 * z
                    - 8.151399419731556 * x**2 * y * z
                    + 2.717133139910518 * y * z
                ],
                [
                    13.58566569955259 * x**3 * y * z**2
                    - 8.151399419731556 * x * y * z**2
                    - 4.528555233184197 * x**3 * y
                    + 2.717133139910518 * x * y
                ],
                [
                    13.58566569955259 * x * y**3 * z**2
                    - 8.151399419731556 * x * y * z**2
                    - 4.528555233184197 * x * y**3
                    + 2.717133139910518 * x * y
                ],
                [
                    13.58566569955259 * x**2 * y * z**3
                    - 4.528555233184197 * y * z**3
                    - 8.151399419731556 * x**2 * y * z
                    + 2.717133139910518 * y * z
                ],
                [
                    13.58566569955259 * x * y**2 * z**3
                    - 4.528555233184197 * x * z**3
                    - 8.151399419731556 * x * y**2 * z
                    + 2.717133139910518 * x * z
                ],
                [
                    15.46796083845572 * x**3 * y**3
                    - 9.280776503073431 * x * y**3
                    - 9.280776503073431 * x**3 * y
                    + 5.568465901844059 * x * y
                ],
                [
                    15.46796083845572 * x**3 * z**3
                    - 9.280776503073431 * x * z**3
                    - 9.280776503073431 * x**3 * z
                    + 5.568465901844059 * x * z
                ],
                [
                    15.46796083845572 * y**3 * z**3
                    - 9.280776503073431 * y * z**3
                    - 9.280776503073431 * y**3 * z
                    + 5.568465901844059 * y * z
                ],
                [
                    26.30852850075426 * x**3 * y**2 * z**2
                    - 15.78511710045256 * x * y**2 * z**2
                    - 8.76950950025142 * x**3 * z**2
                    + 5.261705700150851 * x * z**2
                    - 8.76950950025142 * x**3 * y**2
                    + 5.261705700150851 * x * y**2
                    + 2.92316983341714 * x**3
                    - 1.753901900050284 * x
                ],
                [
                    26.30852850075426 * x**2 * y**3 * z**2
                    - 8.76950950025142 * y**3 * z**2
                    - 15.78511710045256 * x**2 * y * z**2
                    + 5.261705700150851 * y * z**2
                    - 8.76950950025142 * x**2 * y**3
                    + 2.92316983341714 * y**3
                    + 5.261705700150851 * x**2 * y
                    - 1.753901900050284 * y
                ],
                [
                    26.30852850075426 * x**2 * y**2 * z**3
                    - 8.76950950025142 * y**2 * z**3
                    - 8.76950950025142 * x**2 * z**3
                    + 2.92316983341714 * z**3
                    - 15.78511710045256 * x**2 * y**2 * z
                    + 5.261705700150851 * y**2 * z
                    + 5.261705700150851 * x**2 * z
                    - 1.753901900050284 * z
                ],
                [
                    26.791294061691 * x**3 * y**3 * z
                    - 16.0747764370146 * x * y**3 * z
                    - 16.0747764370146 * x**3 * y * z
                    + 9.644865862208759 * x * y * z
                ],
                [
                    26.791294061691 * x**3 * y * z**3
                    - 16.0747764370146 * x * y * z**3
                    - 16.0747764370146 * x**3 * y * z
                    + 9.644865862208759 * x * y * z
                ],
                [
                    26.791294061691 * x * y**3 * z**3
                    - 16.0747764370146 * x * y * z**3
                    - 16.0747764370146 * x * y**3 * z
                    + 9.644865862208759 * x * y * z
                ],
                [
                    51.88111786213746 * x**3 * y**3 * z**2
                    - 31.12867071728247 * x * y**3 * z**2
                    - 31.12867071728247 * x**3 * y * z**2
                    + 18.67720243036948 * x * y * z**2
                    - 17.29370595404582 * x**3 * y**3
                    + 10.37622357242749 * x * y**3
                    + 10.37622357242749 * x**3 * y
                    - 6.225734143456492 * x * y
                ],
                [
                    51.88111786213746 * x**3 * y**2 * z**3
                    - 31.12867071728247 * x * y**2 * z**3
                    - 17.29370595404582 * x**3 * z**3
                    + 10.37622357242749 * x * z**3
                    - 31.12867071728247 * x**3 * y**2 * z
                    + 18.67720243036948 * x * y**2 * z
                    + 10.37622357242749 * x**3 * z
                    - 6.225734143456492 * x * z
                ],
                [
                    51.88111786213746 * x**2 * y**3 * z**3
                    - 17.29370595404582 * y**3 * z**3
                    - 31.12867071728247 * x**2 * y * z**3
                    + 10.37622357242749 * y * z**3
                    - 31.12867071728247 * x**2 * y**3 * z
                    + 10.37622357242749 * y**3 * z
                    + 18.67720243036948 * x**2 * y * z
                    - 6.225734143456492 * y * z
                ],
                [
                    102.3109441695999 * x**3 * y**3 * z**3
                    - 61.38656650175994 * x * y**3 * z**3
                    - 61.38656650175994 * x**3 * y * z**3
                    + 36.83193990105597 * x * y * z**3
                    - 61.38656650175994 * x**3 * y**3 * z
                    + 36.83193990105597 * x * y**3 * z
                    + 36.83193990105597 * x**3 * y * z
                    - 22.09916394063358 * x * y * z
                ],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0] * interpList.shape[0] * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, functionVector.shape[0]):
                interpMatrix[
                    k
                    + j * interpList.shape[0]
                    + i * interpList.shape[0] * interpList.shape[0],
                    l,
                ] = (
                    functionVector[l]
                    .subs(x, interpList[k])
                    .subs(y, interpList[j])
                    .subs(z, interpList[i])
                )
      else:
        raise NameError(
            "interpMatrix: Order {} is not supported!\nPolynomial order must be <4".format(
                order
            )
        )

    elif modal and basis_type == "gkhybrid":
      if order == 1:
        functionVector = Matrix(
            [
                [0.3535533905932737],
                [0.6123724356957944 * x],
                [0.6123724356957944 * y],
                [0.6123724356957944 * z],
                [1.060660171779821 * x * y],
                [1.060660171779821 * x * z],
                [1.060660171779821 * y * z],
                [1.837117307087383 * x * y * z],
                [1.185854122563142 * (y**2 - 0.3333333333333333)],
                [2.053959590644372 * (x * y**2 - 0.3333333333333333 * x)],
                [2.053959590644372 * (y**2 * z - 0.3333333333333333 * z)],
                [3.557562367689425 * (x * y**2 * z - 0.3333333333333333 * x * z)],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpListND[0].shape[0]
                * interpListND[1].shape[0]
                * interpListND[2].shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpListND[2].shape[0]):
          for j in range(0, interpListND[1].shape[0]):
            for k in range(0, interpListND[0].shape[0]):
              for l in range(0, functionVector.shape[0]):
                interpMatrix[
                    k
                    + j * interpListND[0].shape[0]
                    + i * interpListND[1].shape[0] * interpListND[0].shape[0],
                    l,
                ] = (
                    functionVector[l]
                    .subs(x, interpListND[0][k])
                    .subs(y, interpListND[1][j])
                    .subs(z, interpListND[2][i])
                )

      else:
        raise NameError(
            "interpMatrix: Order {} is not supported!\nPolynomial order must be =1".format(
                order
            )
        )

    elif modal and basis_type == "hybrid":
      if order == 1:
        functionVector = Matrix(
            [
                [0.3535533905932737],
                [0.6123724356957945 * x],
                [0.6123724356957945 * y],
                [0.6123724356957945 * z],
                [1.060660171779821 * x * y],
                [1.060660171779821 * x * z],
                [1.060660171779821 * y * z],
                [1.837117307087384 * x * y * z],
                [1.185854122563142 * (z**2 - 0.3333333333333333)],
                [2.053959590644373 * (x * z**2 - 0.3333333333333333 * x)],
                [2.053959590644373 * (y * z**2 - 0.3333333333333333 * y)],
                [3.557562367689427 * (x * y * z**2 - 0.3333333333333332 * x * y)],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpListND[0].shape[0]
                * interpListND[1].shape[0]
                * interpListND[2].shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpListND[2].shape[0]):
          for j in range(0, interpListND[1].shape[0]):
            for k in range(0, interpListND[0].shape[0]):
              for l in range(0, functionVector.shape[0]):
                interpMatrix[
                    k
                    + j * interpListND[0].shape[0]
                    + i * interpListND[1].shape[0] * interpListND[0].shape[0],
                    l,
                ] = (
                    functionVector[l]
                    .subs(x, interpListND[0][k])
                    .subs(y, interpListND[1][j])
                    .subs(z, interpListND[2][i])
                )

      else:
        raise NameError(
            "interpMatrix: Order {} is not supported!\nPolynomial order must be =1".format(
                order
            )
        )

    elif modal == False and basis_type == "serendipity":
      if order == 1:
        functionVector = Matrix(
            [
                [
                    (x * y) / 8.0
                    - y / 8.0
                    - z / 8.0
                    - x / 8.0
                    + (x * z) / 8.0
                    + (y * z) / 8.0
                    - (x * y * z) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    x / 8.0
                    - y / 8.0
                    - z / 8.0
                    - (x * y) / 8.0
                    - (x * z) / 8.0
                    + (y * z) / 8.0
                    + (x * y * z) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    y / 8.0
                    - x / 8.0
                    - z / 8.0
                    - (x * y) / 8.0
                    + (x * z) / 8.0
                    - (y * z) / 8.0
                    + (x * y * z) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    x / 8.0
                    + y / 8.0
                    - z / 8.0
                    + (x * y) / 8.0
                    - (x * z) / 8.0
                    - (y * z) / 8.0
                    - (x * y * z) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    z / 8.0
                    - y / 8.0
                    - x / 8.0
                    + (x * y) / 8.0
                    - (x * z) / 8.0
                    - (y * z) / 8.0
                    + (x * y * z) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    x / 8.0
                    - y / 8.0
                    + z / 8.0
                    - (x * y) / 8.0
                    + (x * z) / 8.0
                    - (y * z) / 8.0
                    - (x * y * z) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    y / 8.0
                    - x / 8.0
                    + z / 8.0
                    - (x * y) / 8.0
                    - (x * z) / 8.0
                    + (y * z) / 8.0
                    - (x * y * z) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    x / 8.0
                    + y / 8.0
                    + z / 8.0
                    + (x * y) / 8.0
                    + (x * z) / 8.0
                    + (y * z) / 8.0
                    + (x * y * z) / 8.0
                    + 1.0 / 8.0
                ],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0] * interpList.shape[0] * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, functionVector.shape[0]):
                interpMatrix[
                    k
                    + j * interpList.shape[0]
                    + i * interpList.shape[0] * interpList.shape[0],
                    l,
                ] = (
                    functionVector[l]
                    .subs(x, interpList[k])
                    .subs(y, interpList[j])
                    .subs(z, interpList[i])
                )

      elif order == 2:
        functionVector = Matrix(
            [
                [
                    (x**2 * y * z) / 8.0
                    - (x**2 * y) / 8.0
                    - (x**2 * z) / 8.0
                    + x**2 / 8.0
                    + (x * y**2 * z) / 8.0
                    - (x * y**2) / 8.0
                    + (x * y * z**2) / 8.0
                    - (x * y * z) / 8.0
                    - (x * z**2) / 8.0
                    + x / 8.0
                    - (y**2 * z) / 8.0
                    + y**2 / 8.0
                    - (y * z**2) / 8.0
                    + y / 8.0
                    + z**2 / 8.0
                    + z / 8.0
                    - 1.0 / 4.0
                ],
                [
                    (y * z) / 4.0
                    - z / 4.0
                    - y / 4.0
                    + (x**2 * y) / 4.0
                    + (x**2 * z) / 4.0
                    - x**2 / 4.0
                    - (x**2 * y * z) / 4.0
                    + 1.0 / 4.0
                ],
                [
                    (x**2 * y * z) / 8.0
                    - (x**2 * y) / 8.0
                    - (x**2 * z) / 8.0
                    + x**2 / 8.0
                    - (x * y**2 * z) / 8.0
                    + (x * y**2) / 8.0
                    - (x * y * z**2) / 8.0
                    + (x * y * z) / 8.0
                    + (x * z**2) / 8.0
                    - x / 8.0
                    - (y**2 * z) / 8.0
                    + y**2 / 8.0
                    - (y * z**2) / 8.0
                    + y / 8.0
                    + z**2 / 8.0
                    + z / 8.0
                    - 1.0 / 4.0
                ],
                [
                    (x * z) / 4.0
                    - z / 4.0
                    - x / 4.0
                    + (x * y**2) / 4.0
                    + (y**2 * z) / 4.0
                    - y**2 / 4.0
                    - (x * y**2 * z) / 4.0
                    + 1.0 / 4.0
                ],
                [
                    x / 4.0
                    - z / 4.0
                    - (x * z) / 4.0
                    - (x * y**2) / 4.0
                    + (y**2 * z) / 4.0
                    - y**2 / 4.0
                    + (x * y**2 * z) / 4.0
                    + 1.0 / 4.0
                ],
                [
                    -(x**2 * y * z) / 8.0
                    + (x**2 * y) / 8.0
                    - (x**2 * z) / 8.0
                    + x**2 / 8.0
                    + (x * y**2 * z) / 8.0
                    - (x * y**2) / 8.0
                    - (x * y * z**2) / 8.0
                    + (x * y * z) / 8.0
                    - (x * z**2) / 8.0
                    + x / 8.0
                    - (y**2 * z) / 8.0
                    + y**2 / 8.0
                    + (y * z**2) / 8.0
                    - y / 8.0
                    + z**2 / 8.0
                    + z / 8.0
                    - 1.0 / 4.0
                ],
                [
                    y / 4.0
                    - z / 4.0
                    - (y * z) / 4.0
                    - (x**2 * y) / 4.0
                    + (x**2 * z) / 4.0
                    - x**2 / 4.0
                    + (x**2 * y * z) / 4.0
                    + 1.0 / 4.0
                ],
                [
                    -(x**2 * y * z) / 8.0
                    + (x**2 * y) / 8.0
                    - (x**2 * z) / 8.0
                    + x**2 / 8.0
                    - (x * y**2 * z) / 8.0
                    + (x * y**2) / 8.0
                    + (x * y * z**2) / 8.0
                    - (x * y * z) / 8.0
                    + (x * z**2) / 8.0
                    - x / 8.0
                    - (y**2 * z) / 8.0
                    + y**2 / 8.0
                    + (y * z**2) / 8.0
                    - y / 8.0
                    + z**2 / 8.0
                    + z / 8.0
                    - 1.0 / 4.0
                ],
                [
                    (x * y) / 4.0
                    - y / 4.0
                    - x / 4.0
                    + (x * z**2) / 4.0
                    + (y * z**2) / 4.0
                    - z**2 / 4.0
                    - (x * y * z**2) / 4.0
                    + 1.0 / 4.0
                ],
                [
                    x / 4.0
                    - y / 4.0
                    - (x * y) / 4.0
                    - (x * z**2) / 4.0
                    + (y * z**2) / 4.0
                    - z**2 / 4.0
                    + (x * y * z**2) / 4.0
                    + 1.0 / 4.0
                ],
                [
                    y / 4.0
                    - x / 4.0
                    - (x * y) / 4.0
                    + (x * z**2) / 4.0
                    - (y * z**2) / 4.0
                    - z**2 / 4.0
                    + (x * y * z**2) / 4.0
                    + 1.0 / 4.0
                ],
                [
                    x / 4.0
                    + y / 4.0
                    + (x * y) / 4.0
                    - (x * z**2) / 4.0
                    - (y * z**2) / 4.0
                    - z**2 / 4.0
                    - (x * y * z**2) / 4.0
                    + 1.0 / 4.0
                ],
                [
                    -(x**2 * y * z) / 8.0
                    - (x**2 * y) / 8.0
                    + (x**2 * z) / 8.0
                    + x**2 / 8.0
                    - (x * y**2 * z) / 8.0
                    - (x * y**2) / 8.0
                    + (x * y * z**2) / 8.0
                    + (x * y * z) / 8.0
                    - (x * z**2) / 8.0
                    + x / 8.0
                    + (y**2 * z) / 8.0
                    + y**2 / 8.0
                    - (y * z**2) / 8.0
                    + y / 8.0
                    + z**2 / 8.0
                    - z / 8.0
                    - 1.0 / 4.0
                ],
                [
                    z / 4.0
                    - y / 4.0
                    - (y * z) / 4.0
                    + (x**2 * y) / 4.0
                    - (x**2 * z) / 4.0
                    - x**2 / 4.0
                    + (x**2 * y * z) / 4.0
                    + 1.0 / 4.0
                ],
                [
                    -(x**2 * y * z) / 8.0
                    - (x**2 * y) / 8.0
                    + (x**2 * z) / 8.0
                    + x**2 / 8.0
                    + (x * y**2 * z) / 8.0
                    + (x * y**2) / 8.0
                    - (x * y * z**2) / 8.0
                    - (x * y * z) / 8.0
                    + (x * z**2) / 8.0
                    - x / 8.0
                    + (y**2 * z) / 8.0
                    + y**2 / 8.0
                    - (y * z**2) / 8.0
                    + y / 8.0
                    + z**2 / 8.0
                    - z / 8.0
                    - 1.0 / 4.0
                ],
                [
                    z / 4.0
                    - x / 4.0
                    - (x * z) / 4.0
                    + (x * y**2) / 4.0
                    - (y**2 * z) / 4.0
                    - y**2 / 4.0
                    + (x * y**2 * z) / 4.0
                    + 1.0 / 4.0
                ],
                [
                    x / 4.0
                    + z / 4.0
                    + (x * z) / 4.0
                    - (x * y**2) / 4.0
                    - (y**2 * z) / 4.0
                    - y**2 / 4.0
                    - (x * y**2 * z) / 4.0
                    + 1.0 / 4.0
                ],
                [
                    (x**2 * y * z) / 8.0
                    + (x**2 * y) / 8.0
                    + (x**2 * z) / 8.0
                    + x**2 / 8.0
                    - (x * y**2 * z) / 8.0
                    - (x * y**2) / 8.0
                    - (x * y * z**2) / 8.0
                    - (x * y * z) / 8.0
                    - (x * z**2) / 8.0
                    + x / 8.0
                    + (y**2 * z) / 8.0
                    + y**2 / 8.0
                    + (y * z**2) / 8.0
                    - y / 8.0
                    + z**2 / 8.0
                    - z / 8.0
                    - 1.0 / 4.0
                ],
                [
                    y / 4.0
                    + z / 4.0
                    + (y * z) / 4.0
                    - (x**2 * y) / 4.0
                    - (x**2 * z) / 4.0
                    - x**2 / 4.0
                    - (x**2 * y * z) / 4.0
                    + 1.0 / 4.0
                ],
                [
                    (x**2 * y * z) / 8.0
                    + (x**2 * y) / 8.0
                    + (x**2 * z) / 8.0
                    + x**2 / 8.0
                    + (x * y**2 * z) / 8.0
                    + (x * y**2) / 8.0
                    + (x * y * z**2) / 8.0
                    + (x * y * z) / 8.0
                    + (x * z**2) / 8.0
                    - x / 8.0
                    + (y**2 * z) / 8.0
                    + y**2 / 8.0
                    + (y * z**2) / 8.0
                    - y / 8.0
                    + z**2 / 8.0
                    - z / 8.0
                    - 1.0 / 4.0
                ],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0] * interpList.shape[0] * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, functionVector.shape[0]):
                interpMatrix[
                    k
                    + j * interpList.shape[0]
                    + i * interpList.shape[0] * interpList.shape[0],
                    l,
                ] = (
                    functionVector[l]
                    .subs(x, interpList[k])
                    .subs(y, interpList[j])
                    .subs(z, interpList[i])
                )
      else:
        raise NameError(
            "interpMatrix: Order {} is not supported!\nPolynomial order must be <3 for nodal Serendipity in 3D".format(
                order
            )
        )

    else:
      raise NameError(
          "interpMatrix: Basis {} is not supported!\nSupported basis are currently 'nodal Serendipity', 'modal Serendipity', and 'modal maximal order'".format(
              basis_type
          )
      )
  elif dim == 4:
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")
    w = Symbol("w")
    if modal and basis_type == "maximal-order":
      if order == 1:
        functionVector = Matrix(
            [
                [0.25],
                [0.4330127018922192 * x],
                [0.4330127018922192 * y],
                [0.4330127018922192 * z],
                [0.4330127018922192 * w],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, interpList.shape[0]):
                for m in range(0, functionVector.shape[0]):
                  interpMatrix[
                      l
                      + k * interpList.shape[0]
                      + j * interpList.shape[0] * interpList.shape[0]
                      + i
                      * interpList.shape[0]
                      * interpList.shape[0]
                      * interpList.shape[0],
                      m,
                  ] = (
                      functionVector[m]
                      .subs(x, interpList[l])
                      .subs(y, interpList[k])
                      .subs(z, interpList[j])
                      .subs(w, interpList[i])
                  )

      elif order == 2:
        functionVector = Matrix(
            [
                [0.25],
                [0.4330127018922192 * x],
                [0.4330127018922192 * y],
                [0.4330127018922192 * z],
                [0.4330127018922192 * w],
                [0.75 * x * y],
                [0.75 * x * z],
                [0.75 * y * z],
                [0.75 * x * w],
                [0.75 * y * w],
                [0.75 * z * w],
                [0.8385254915624196 * x**2 - 0.2795084971874732],
                [0.8385254915624196 * y**2 - 0.2795084971874732],
                [0.8385254915624196 * z**2 - 0.2795084971874732],
                [0.8385254915624196 * w**2 - 0.2795084971874732],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, interpList.shape[0]):
                for m in range(0, functionVector.shape[0]):
                  interpMatrix[
                      l
                      + k * interpList.shape[0]
                      + j * interpList.shape[0] * interpList.shape[0]
                      + i
                      * interpList.shape[0]
                      * interpList.shape[0]
                      * interpList.shape[0],
                      m,
                  ] = (
                      functionVector[m]
                      .subs(x, interpList[l])
                      .subs(y, interpList[k])
                      .subs(z, interpList[j])
                      .subs(w, interpList[i])
                  )

      elif order == 3:
        functionVector = Matrix(
            [
                [0.25],
                [0.4330127018922192 * x],
                [0.4330127018922192 * y],
                [0.4330127018922192 * z],
                [0.4330127018922192 * w],
                [0.75 * x * y],
                [0.75 * x * z],
                [0.75 * y * z],
                [0.75 * x * w],
                [0.75 * y * w],
                [0.75 * z * w],
                [0.8385254915624196 * x**2 - 0.2795084971874732],
                [0.8385254915624196 * y**2 - 0.2795084971874732],
                [0.8385254915624196 * z**2 - 0.2795084971874732],
                [0.8385254915624196 * w**2 - 0.2795084971874732],
                [1.299038105676659 * x * y * z],
                [1.299038105676659 * x * y * w],
                [1.299038105676659 * x * z * w],
                [1.299038105676659 * y * z * w],
                [1.452368754827781 * x**2 * y - 0.4841229182759272 * y],
                [1.452368754827781 * x * y**2 - 0.4841229182759272 * x],
                [1.452368754827781 * x**2 * z - 0.4841229182759272 * z],
                [1.452368754827781 * y**2 * z - 0.4841229182759272 * z],
                [1.452368754827781 * x * z**2 - 0.4841229182759272 * x],
                [1.452368754827781 * y * z**2 - 0.4841229182759272 * y],
                [1.452368754827781 * x**2 * w - 0.4841229182759272 * w],
                [1.452368754827781 * y**2 * w - 0.4841229182759272 * w],
                [1.452368754827781 * z**2 * w - 0.4841229182759272 * w],
                [1.452368754827781 * x * w**2 - 0.4841229182759272 * x],
                [1.452368754827781 * y * w**2 - 0.4841229182759272 * y],
                [1.452368754827781 * z * w**2 - 0.4841229182759272 * z],
                [1.653594569415366 * x**3 - 0.9921567416492196 * x],
                [1.653594569415366 * y**3 - 0.9921567416492196 * y],
                [1.653594569415366 * z**3 - 0.9921567416492196 * z],
                [1.653594569415366 * w**3 - 0.9921567416492196 * w],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, interpList.shape[0]):
                for m in range(0, functionVector.shape[0]):
                  interpMatrix[
                      l
                      + k * interpList.shape[0]
                      + j * interpList.shape[0] * interpList.shape[0]
                      + i
                      * interpList.shape[0]
                      * interpList.shape[0]
                      * interpList.shape[0],
                      m,
                  ] = (
                      functionVector[m]
                      .subs(x, interpList[l])
                      .subs(y, interpList[k])
                      .subs(z, interpList[j])
                      .subs(w, interpList[i])
                  )

      elif order == 4:
        functionVector = Matrix(
            [
                [0.25],
                [0.4330127018922192 * x],
                [0.4330127018922192 * y],
                [0.4330127018922192 * z],
                [0.4330127018922192 * w],
                [0.75 * x * y],
                [0.75 * x * z],
                [0.75 * y * z],
                [0.75 * x * w],
                [0.75 * y * w],
                [0.75 * z * w],
                [0.8385254915624196 * x**2 - 0.2795084971874732],
                [0.8385254915624196 * y**2 - 0.2795084971874732],
                [0.8385254915624196 * z**2 - 0.2795084971874732],
                [0.8385254915624196 * w**2 - 0.2795084971874732],
                [1.299038105676659 * x * y * z],
                [1.299038105676659 * x * y * w],
                [1.299038105676659 * x * z * w],
                [1.299038105676659 * y * z * w],
                [1.452368754827781 * x**2 * y - 0.4841229182759272 * y],
                [1.452368754827781 * x * y**2 - 0.4841229182759272 * x],
                [1.452368754827781 * x**2 * z - 0.4841229182759272 * z],
                [1.452368754827781 * y**2 * z - 0.4841229182759272 * z],
                [1.452368754827781 * x * z**2 - 0.4841229182759272 * x],
                [1.452368754827781 * y * z**2 - 0.4841229182759272 * y],
                [1.452368754827781 * x**2 * w - 0.4841229182759272 * w],
                [1.452368754827781 * y**2 * w - 0.4841229182759272 * w],
                [1.452368754827781 * z**2 * w - 0.4841229182759272 * w],
                [1.452368754827781 * x * w**2 - 0.4841229182759272 * x],
                [1.452368754827781 * y * w**2 - 0.4841229182759272 * y],
                [1.452368754827781 * z * w**2 - 0.4841229182759272 * z],
                [1.653594569415366 * x**3 - 0.9921567416492196 * x],
                [1.653594569415366 * y**3 - 0.9921567416492196 * y],
                [1.653594569415366 * z**3 - 0.9921567416492196 * z],
                [1.653594569415366 * w**3 - 0.9921567416492196 * w],
                [2.25 * x * y * z * w],
                [2.515576474687268 * x**2 * y * z - 0.8385254915624226 * y * z],
                [2.515576474687268 * x * y**2 * z - 0.8385254915624226 * x * z],
                [2.515576474687268 * x * y * z**2 - 0.8385254915624226 * x * y],
                [2.515576474687268 * x**2 * y * w - 0.8385254915624226 * y * w],
                [2.515576474687268 * x * y**2 * w - 0.8385254915624226 * x * w],
                [2.515576474687268 * x**2 * z * w - 0.8385254915624226 * z * w],
                [2.515576474687268 * y**2 * z * w - 0.8385254915624226 * z * w],
                [2.515576474687268 * x * z**2 * w - 0.8385254915624226 * x * w],
                [2.515576474687268 * y * z**2 * w - 0.8385254915624226 * y * w],
                [2.515576474687268 * x * y * w**2 - 0.8385254915624226 * x * y],
                [2.515576474687268 * x * z * w**2 - 0.8385254915624226 * x * z],
                [2.515576474687268 * y * z * w**2 - 0.8385254915624226 * y * z],
                [2.8125 * x**2 * y**2 - 0.9375 * y**2 - 0.9375 * x**2 + 0.3125],
                [2.8125 * x**2 * z**2 - 0.9375 * z**2 - 0.9375 * x**2 + 0.3125],
                [2.8125 * y**2 * z**2 - 0.9375 * z**2 - 0.9375 * y**2 + 0.3125],
                [2.8125 * x**2 * w**2 - 0.9375 * w**2 - 0.9375 * x**2 + 0.3125],
                [2.8125 * y**2 * w**2 - 0.9375 * w**2 - 0.9375 * y**2 + 0.3125],
                [2.8125 * z**2 * w**2 - 0.9375 * w**2 - 0.9375 * z**2 + 0.3125],
                [2.864109809347398 * x**3 * y - 1.718465885608439 * x * y],
                [2.864109809347398 * x * y**3 - 1.718465885608439 * x * y],
                [2.864109809347398 * x**3 * z - 1.718465885608439 * x * z],
                [2.864109809347398 * y**3 * z - 1.718465885608439 * y * z],
                [2.864109809347398 * x * z**3 - 1.718465885608439 * x * z],
                [2.864109809347398 * y * z**3 - 1.718465885608439 * y * z],
                [2.864109809347398 * x**3 * w - 1.718465885608439 * x * w],
                [2.864109809347398 * y**3 * w - 1.718465885608439 * y * w],
                [2.864109809347398 * z**3 * w - 1.718465885608439 * z * w],
                [2.864109809347398 * x * w**3 - 1.718465885608439 * x * w],
                [2.864109809347398 * y * w**3 - 1.718465885608439 * y * w],
                [2.864109809347398 * z * w**3 - 1.718465885608439 * z * w],
                [3.28125 * x**4 - 2.8125 * x**2 + 0.28125],
                [3.28125 * y**4 - 2.8125 * y**2 + 0.28125],
                [3.28125 * z**4 - 2.8125 * z**2 + 0.28125],
                [3.28125 * w**4 - 2.8125 * w**2 + 0.28125],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, interpList.shape[0]):
                for m in range(0, functionVector.shape[0]):
                  interpMatrix[
                      l
                      + k * interpList.shape[0]
                      + j * interpList.shape[0] * interpList.shape[0]
                      + i
                      * interpList.shape[0]
                      * interpList.shape[0]
                      * interpList.shape[0],
                      m,
                  ] = (
                      functionVector[m]
                      .subs(x, interpList[l])
                      .subs(y, interpList[k])
                      .subs(z, interpList[j])
                      .subs(w, interpList[i])
                  )
      else:
        raise NameError(
            "interpMatrix: Order {} is not supported!\nPolynomial order must be <5".format(
                order
            )
        )
    elif modal and basis_type == "serendipity":
      if order == 0:
        functionVector = Matrix([[0.25]])
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, interpList.shape[0]):
                for m in range(0, functionVector.shape[0]):
                  interpMatrix[
                      l
                      + k * interpList.shape[0]
                      + j * interpList.shape[0] * interpList.shape[0]
                      + i
                      * interpList.shape[0]
                      * interpList.shape[0]
                      * interpList.shape[0],
                      m,
                  ] = (
                      functionVector[m]
                      .subs(x, interpList[l])
                      .subs(y, interpList[k])
                      .subs(z, interpList[j])
                      .subs(w, interpList[i])
                  )

      elif order == 1:
        functionVector = Matrix(
            [
                [0.25],
                [0.4330127018922192 * x],
                [0.4330127018922192 * y],
                [0.4330127018922192 * z],
                [0.4330127018922192 * w],
                [0.75 * x * y],
                [0.75 * x * z],
                [0.75 * y * z],
                [0.75 * x * w],
                [0.75 * y * w],
                [0.75 * z * w],
                [1.299038105676659 * x * y * z],
                [1.299038105676659 * x * y * w],
                [1.299038105676659 * x * z * w],
                [1.299038105676659 * y * z * w],
                [2.25 * x * y * z * w],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, interpList.shape[0]):
                for m in range(0, functionVector.shape[0]):
                  interpMatrix[
                      l
                      + k * interpList.shape[0]
                      + j * interpList.shape[0] * interpList.shape[0]
                      + i
                      * interpList.shape[0]
                      * interpList.shape[0]
                      * interpList.shape[0],
                      m,
                  ] = (
                      functionVector[m]
                      .subs(x, interpList[l])
                      .subs(y, interpList[k])
                      .subs(z, interpList[j])
                      .subs(w, interpList[i])
                  )

      elif order == 2:
        functionVector = Matrix(
            [
                [0.25],
                [0.4330127018922192 * x],
                [0.4330127018922192 * y],
                [0.4330127018922192 * z],
                [0.4330127018922192 * w],
                [0.75 * x * y],
                [0.75 * x * z],
                [0.75 * y * z],
                [0.75 * x * w],
                [0.75 * y * w],
                [0.75 * z * w],
                [0.8385254915624196 * x**2 - 0.2795084971874732],
                [0.8385254915624196 * y**2 - 0.2795084971874732],
                [0.8385254915624196 * z**2 - 0.2795084971874732],
                [0.8385254915624196 * w**2 - 0.2795084971874732],
                [1.299038105676659 * x * y * z],
                [1.299038105676659 * x * y * w],
                [1.299038105676659 * x * z * w],
                [1.299038105676659 * y * z * w],
                [1.452368754827781 * x**2 * y - 0.4841229182759272 * y],
                [1.452368754827781 * x * y**2 - 0.4841229182759272 * x],
                [1.452368754827781 * x**2 * z - 0.4841229182759272 * z],
                [1.452368754827781 * y**2 * z - 0.4841229182759272 * z],
                [1.452368754827781 * x * z**2 - 0.4841229182759272 * x],
                [1.452368754827781 * y * z**2 - 0.4841229182759272 * y],
                [1.452368754827781 * x**2 * w - 0.4841229182759272 * w],
                [1.452368754827781 * y**2 * w - 0.4841229182759272 * w],
                [1.452368754827781 * z**2 * w - 0.4841229182759272 * w],
                [1.452368754827781 * x * w**2 - 0.4841229182759272 * x],
                [1.452368754827781 * y * w**2 - 0.4841229182759272 * y],
                [1.452368754827781 * z * w**2 - 0.4841229182759272 * z],
                [2.25 * x * y * z * w],
                [2.515576474687268 * x**2 * y * z - 0.8385254915624226 * y * z],
                [2.515576474687268 * x * y**2 * z - 0.8385254915624226 * x * z],
                [2.515576474687268 * x * y * z**2 - 0.8385254915624226 * x * y],
                [2.515576474687268 * x**2 * y * w - 0.8385254915624226 * y * w],
                [2.515576474687268 * x * y**2 * w - 0.8385254915624226 * x * w],
                [2.515576474687268 * x**2 * z * w - 0.8385254915624226 * z * w],
                [2.515576474687268 * y**2 * z * w - 0.8385254915624226 * z * w],
                [2.515576474687268 * x * z**2 * w - 0.8385254915624226 * x * w],
                [2.515576474687268 * y * z**2 * w - 0.8385254915624226 * y * w],
                [2.515576474687268 * x * y * w**2 - 0.8385254915624226 * x * y],
                [2.515576474687268 * x * z * w**2 - 0.8385254915624226 * x * z],
                [2.515576474687268 * y * z * w**2 - 0.8385254915624226 * y * z],
                [4.357106264483344 * x**2 * y * z * w - 1.452368754827781 * y * z * w],
                [4.357106264483344 * x * y**2 * z * w - 1.452368754827781 * x * z * w],
                [4.357106264483344 * x * y * z**2 * w - 1.452368754827781 * x * y * w],
                [4.357106264483344 * x * y * z * w**2 - 1.452368754827781 * x * y * z],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, interpList.shape[0]):
                for m in range(0, functionVector.shape[0]):
                  interpMatrix[
                      l
                      + k * interpList.shape[0]
                      + j * interpList.shape[0] * interpList.shape[0]
                      + i
                      * interpList.shape[0]
                      * interpList.shape[0]
                      * interpList.shape[0],
                      m,
                  ] = (
                      functionVector[m]
                      .subs(x, interpList[l])
                      .subs(y, interpList[k])
                      .subs(z, interpList[j])
                      .subs(w, interpList[i])
                  )

      elif order == 3:
        functionVector = Matrix(
            [
                [0.25],
                [0.4330127018922192 * x],
                [0.4330127018922192 * y],
                [0.4330127018922192 * z],
                [0.4330127018922192 * w],
                [0.75 * x * y],
                [0.75 * x * z],
                [0.75 * y * z],
                [0.75 * x * w],
                [0.75 * y * w],
                [0.75 * z * w],
                [0.8385254915624196 * x**2 - 0.2795084971874732],
                [0.8385254915624196 * y**2 - 0.2795084971874732],
                [0.8385254915624196 * z**2 - 0.2795084971874732],
                [0.8385254915624196 * w**2 - 0.2795084971874732],
                [1.299038105676659 * x * y * z],
                [1.299038105676659 * x * y * w],
                [1.299038105676659 * x * z * w],
                [1.299038105676659 * y * z * w],
                [1.452368754827781 * x**2 * y - 0.4841229182759272 * y],
                [1.452368754827781 * x * y**2 - 0.4841229182759272 * x],
                [1.452368754827781 * x**2 * z - 0.4841229182759272 * z],
                [1.452368754827781 * y**2 * z - 0.4841229182759272 * z],
                [1.452368754827781 * x * z**2 - 0.4841229182759272 * x],
                [1.452368754827781 * y * z**2 - 0.4841229182759272 * y],
                [1.452368754827781 * x**2 * w - 0.4841229182759272 * w],
                [1.452368754827781 * y**2 * w - 0.4841229182759272 * w],
                [1.452368754827781 * z**2 * w - 0.4841229182759272 * w],
                [1.452368754827781 * x * w**2 - 0.4841229182759272 * x],
                [1.452368754827781 * y * w**2 - 0.4841229182759272 * y],
                [1.452368754827781 * z * w**2 - 0.4841229182759272 * z],
                [1.653594569415366 * x**3 - 0.9921567416492196 * x],
                [1.653594569415366 * y**3 - 0.9921567416492196 * y],
                [1.653594569415366 * z**3 - 0.9921567416492196 * z],
                [1.653594569415366 * w**3 - 0.9921567416492196 * w],
                [2.25 * x * y * z * w],
                [2.515576474687268 * x**2 * y * z - 0.8385254915624226 * y * z],
                [2.515576474687268 * x * y**2 * z - 0.8385254915624226 * x * z],
                [2.515576474687268 * x * y * z**2 - 0.8385254915624226 * x * y],
                [2.515576474687268 * x**2 * y * w - 0.8385254915624226 * y * w],
                [2.515576474687268 * x * y**2 * w - 0.8385254915624226 * x * w],
                [2.515576474687268 * x**2 * z * w - 0.8385254915624226 * z * w],
                [2.515576474687268 * y**2 * z * w - 0.8385254915624226 * z * w],
                [2.515576474687268 * x * z**2 * w - 0.8385254915624226 * x * w],
                [2.515576474687268 * y * z**2 * w - 0.8385254915624226 * y * w],
                [2.515576474687268 * x * y * w**2 - 0.8385254915624226 * x * y],
                [2.515576474687268 * x * z * w**2 - 0.8385254915624226 * x * z],
                [2.515576474687268 * y * z * w**2 - 0.8385254915624226 * y * z],
                [2.864109809347398 * x**3 * y - 1.718465885608439 * x * y],
                [2.864109809347398 * x * y**3 - 1.718465885608439 * x * y],
                [2.864109809347398 * x**3 * z - 1.718465885608439 * x * z],
                [2.864109809347398 * y**3 * z - 1.718465885608439 * y * z],
                [2.864109809347398 * x * z**3 - 1.718465885608439 * x * z],
                [2.864109809347398 * y * z**3 - 1.718465885608439 * y * z],
                [2.864109809347398 * x**3 * w - 1.718465885608439 * x * w],
                [2.864109809347398 * y**3 * w - 1.718465885608439 * y * w],
                [2.864109809347398 * z**3 * w - 1.718465885608439 * z * w],
                [2.864109809347398 * x * w**3 - 1.718465885608439 * x * w],
                [2.864109809347398 * y * w**3 - 1.718465885608439 * y * w],
                [2.864109809347398 * z * w**3 - 1.718465885608439 * z * w],
                [4.357106264483344 * x**2 * y * z * w - 1.452368754827781 * y * z * w],
                [4.357106264483344 * x * y**2 * z * w - 1.452368754827781 * x * z * w],
                [4.357106264483344 * x * y * z**2 * w - 1.452368754827781 * x * y * w],
                [4.357106264483344 * x * y * z * w**2 - 1.452368754827781 * x * y * z],
                [4.960783708246104 * x**3 * y * z - 2.976470224947662 * x * y * z],
                [4.960783708246104 * x * y**3 * z - 2.976470224947662 * x * y * z],
                [4.960783708246104 * x * y * z**3 - 2.976470224947662 * x * y * z],
                [4.960783708246104 * x**3 * y * w - 2.976470224947662 * x * y * w],
                [4.960783708246104 * x * y**3 * w - 2.976470224947662 * x * y * w],
                [4.960783708246104 * x**3 * z * w - 2.976470224947662 * x * z * w],
                [4.960783708246104 * y**3 * z * w - 2.976470224947662 * y * z * w],
                [4.960783708246104 * x * z**3 * w - 2.976470224947662 * x * z * w],
                [4.960783708246104 * y * z**3 * w - 2.976470224947662 * y * z * w],
                [4.960783708246104 * x * y * w**3 - 2.976470224947662 * x * y * w],
                [4.960783708246104 * x * z * w**3 - 2.976470224947662 * x * z * w],
                [4.960783708246104 * y * z * w**3 - 2.976470224947662 * y * z * w],
                [8.5923294280422 * x**3 * y * z * w - 5.15539765682532 * x * y * z * w],
                [8.5923294280422 * x * y**3 * z * w - 5.15539765682532 * x * y * z * w],
                [8.5923294280422 * x * y * z**3 * w - 5.15539765682532 * x * y * z * w],
                [8.5923294280422 * x * y * z * w**3 - 5.15539765682532 * x * y * z * w],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, interpList.shape[0]):
                for m in range(0, functionVector.shape[0]):
                  interpMatrix[
                      l
                      + k * interpList.shape[0]
                      + j * interpList.shape[0] * interpList.shape[0]
                      + i
                      * interpList.shape[0]
                      * interpList.shape[0]
                      * interpList.shape[0],
                      m,
                  ] = (
                      functionVector[m]
                      .subs(x, interpList[l])
                      .subs(y, interpList[k])
                      .subs(z, interpList[j])
                      .subs(w, interpList[i])
                  )

      elif order == 4:
        functionVector = Matrix(
            [
                [0.25],
                [0.4330127018922192 * x],
                [0.4330127018922192 * y],
                [0.4330127018922192 * z],
                [0.4330127018922192 * w],
                [0.75 * x * y],
                [0.75 * x * z],
                [0.75 * y * z],
                [0.75 * x * w],
                [0.75 * y * w],
                [0.75 * z * w],
                [0.8385254915624196 * x**2 - 0.2795084971874732],
                [0.8385254915624196 * y**2 - 0.2795084971874732],
                [0.8385254915624196 * z**2 - 0.2795084971874732],
                [0.8385254915624196 * w**2 - 0.2795084971874732],
                [1.299038105676659 * x * y * z],
                [1.299038105676659 * x * y * w],
                [1.299038105676659 * x * z * w],
                [1.299038105676659 * y * z * w],
                [1.452368754827781 * x**2 * y - 0.4841229182759272 * y],
                [1.452368754827781 * x * y**2 - 0.4841229182759272 * x],
                [1.452368754827781 * x**2 * z - 0.4841229182759272 * z],
                [1.452368754827781 * y**2 * z - 0.4841229182759272 * z],
                [1.452368754827781 * x * z**2 - 0.4841229182759272 * x],
                [1.452368754827781 * y * z**2 - 0.4841229182759272 * y],
                [1.452368754827781 * x**2 * w - 0.4841229182759272 * w],
                [1.452368754827781 * y**2 * w - 0.4841229182759272 * w],
                [1.452368754827781 * z**2 * w - 0.4841229182759272 * w],
                [1.452368754827781 * x * w**2 - 0.4841229182759272 * x],
                [1.452368754827781 * y * w**2 - 0.4841229182759272 * y],
                [1.452368754827781 * z * w**2 - 0.4841229182759272 * z],
                [1.653594569415366 * x**3 - 0.9921567416492196 * x],
                [1.653594569415366 * y**3 - 0.9921567416492196 * y],
                [1.653594569415366 * z**3 - 0.9921567416492196 * z],
                [1.653594569415366 * w**3 - 0.9921567416492196 * w],
                [2.25 * x * y * z * w],
                [2.515576474687268 * x**2 * y * z - 0.8385254915624226 * y * z],
                [2.515576474687268 * x * y**2 * z - 0.8385254915624226 * x * z],
                [2.515576474687268 * x * y * z**2 - 0.8385254915624226 * x * y],
                [2.515576474687268 * x**2 * y * w - 0.8385254915624226 * y * w],
                [2.515576474687268 * x * y**2 * w - 0.8385254915624226 * x * w],
                [2.515576474687268 * x**2 * z * w - 0.8385254915624226 * z * w],
                [2.515576474687268 * y**2 * z * w - 0.8385254915624226 * z * w],
                [2.515576474687268 * x * z**2 * w - 0.8385254915624226 * x * w],
                [2.515576474687268 * y * z**2 * w - 0.8385254915624226 * y * w],
                [2.515576474687268 * x * y * w**2 - 0.8385254915624226 * x * y],
                [2.515576474687268 * x * z * w**2 - 0.8385254915624226 * x * z],
                [2.515576474687268 * y * z * w**2 - 0.8385254915624226 * y * z],
                [2.8125 * x**2 * y**2 - 0.9375 * y**2 - 0.9375 * x**2 + 0.3125],
                [2.8125 * x**2 * z**2 - 0.9375 * z**2 - 0.9375 * x**2 + 0.3125],
                [2.8125 * y**2 * z**2 - 0.9375 * z**2 - 0.9375 * y**2 + 0.3125],
                [2.8125 * x**2 * w**2 - 0.9375 * w**2 - 0.9375 * x**2 + 0.3125],
                [2.8125 * y**2 * w**2 - 0.9375 * w**2 - 0.9375 * y**2 + 0.3125],
                [2.8125 * z**2 * w**2 - 0.9375 * w**2 - 0.9375 * z**2 + 0.3125],
                [2.864109809347398 * x**3 * y - 1.718465885608439 * x * y],
                [2.864109809347398 * x * y**3 - 1.718465885608439 * x * y],
                [2.864109809347398 * x**3 * z - 1.718465885608439 * x * z],
                [2.864109809347398 * y**3 * z - 1.718465885608439 * y * z],
                [2.864109809347398 * x * z**3 - 1.718465885608439 * x * z],
                [2.864109809347398 * y * z**3 - 1.718465885608439 * y * z],
                [2.864109809347398 * x**3 * w - 1.718465885608439 * x * w],
                [2.864109809347398 * y**3 * w - 1.718465885608439 * y * w],
                [2.864109809347398 * z**3 * w - 1.718465885608439 * z * w],
                [2.864109809347398 * x * w**3 - 1.718465885608439 * x * w],
                [2.864109809347398 * y * w**3 - 1.718465885608439 * y * w],
                [2.864109809347398 * z * w**3 - 1.718465885608439 * z * w],
                [3.28125 * x**4 - 2.8125 * x**2 + 0.28125],
                [3.28125 * y**4 - 2.8125 * y**2 + 0.28125],
                [3.28125 * z**4 - 2.8125 * z**2 + 0.28125],
                [3.28125 * w**4 - 2.8125 * w**2 + 0.28125],
                [4.357106264483344 * x**2 * y * z * w - 1.452368754827781 * y * z * w],
                [4.357106264483344 * x * y**2 * z * w - 1.452368754827781 * x * z * w],
                [4.357106264483344 * x * y * z**2 * w - 1.452368754827781 * x * y * w],
                [4.357106264483344 * x * y * z * w**2 - 1.452368754827781 * x * y * z],
                [
                    4.87139289628746 * x**2 * y**2 * z
                    - 1.62379763209582 * y**2 * z
                    - 1.62379763209582 * x**2 * z
                    + 0.5412658773652733 * z
                ],
                [
                    4.87139289628746 * x**2 * y * z**2
                    - 1.62379763209582 * y * z**2
                    - 1.62379763209582 * x**2 * y
                    + 0.5412658773652733 * y
                ],
                [
                    4.87139289628746 * x * y**2 * z**2
                    - 1.62379763209582 * x * z**2
                    - 1.62379763209582 * x * y**2
                    + 0.5412658773652733 * x
                ],
                [
                    4.87139289628746 * x**2 * y**2 * w
                    - 1.62379763209582 * y**2 * w
                    - 1.62379763209582 * x**2 * w
                    + 0.5412658773652733 * w
                ],
                [
                    4.87139289628746 * x**2 * z**2 * w
                    - 1.62379763209582 * z**2 * w
                    - 1.62379763209582 * x**2 * w
                    + 0.5412658773652733 * w
                ],
                [
                    4.87139289628746 * y**2 * z**2 * w
                    - 1.62379763209582 * z**2 * w
                    - 1.62379763209582 * y**2 * w
                    + 0.5412658773652733 * w
                ],
                [
                    4.87139289628746 * x**2 * y * w**2
                    - 1.62379763209582 * y * w**2
                    - 1.62379763209582 * x**2 * y
                    + 0.5412658773652733 * y
                ],
                [
                    4.87139289628746 * x * y**2 * w**2
                    - 1.62379763209582 * x * w**2
                    - 1.62379763209582 * x * y**2
                    + 0.5412658773652733 * x
                ],
                [
                    4.87139289628746 * x**2 * z * w**2
                    - 1.62379763209582 * z * w**2
                    - 1.62379763209582 * x**2 * z
                    + 0.5412658773652733 * z
                ],
                [
                    4.87139289628746 * y**2 * z * w**2
                    - 1.62379763209582 * z * w**2
                    - 1.62379763209582 * y**2 * z
                    + 0.5412658773652733 * z
                ],
                [
                    4.87139289628746 * x * z**2 * w**2
                    - 1.62379763209582 * x * w**2
                    - 1.62379763209582 * x * z**2
                    + 0.5412658773652733 * x
                ],
                [
                    4.87139289628746 * y * z**2 * w**2
                    - 1.62379763209582 * y * w**2
                    - 1.62379763209582 * y * z**2
                    + 0.5412658773652733 * y
                ],
                [4.960783708246104 * x**3 * y * z - 2.976470224947662 * x * y * z],
                [4.960783708246104 * x * y**3 * z - 2.976470224947662 * x * y * z],
                [4.960783708246104 * x * y * z**3 - 2.976470224947662 * x * y * z],
                [4.960783708246104 * x**3 * y * w - 2.976470224947662 * x * y * w],
                [4.960783708246104 * x * y**3 * w - 2.976470224947662 * x * y * w],
                [4.960783708246104 * x**3 * z * w - 2.976470224947662 * x * z * w],
                [4.960783708246104 * y**3 * z * w - 2.976470224947662 * y * z * w],
                [4.960783708246104 * x * z**3 * w - 2.976470224947662 * x * z * w],
                [4.960783708246104 * y * z**3 * w - 2.976470224947662 * y * z * w],
                [4.960783708246104 * x * y * w**3 - 2.976470224947662 * x * y * w],
                [4.960783708246104 * x * z * w**3 - 2.976470224947662 * x * z * w],
                [4.960783708246104 * y * z * w**3 - 2.976470224947662 * y * z * w],
                [
                    5.68329171233537 * x**4 * y
                    - 4.87139289628746 * x**2 * y
                    + 0.487139289628746 * y
                ],
                [
                    5.68329171233537 * x * y**4
                    - 4.87139289628746 * x * y**2
                    + 0.487139289628746 * x
                ],
                [
                    5.68329171233537 * x**4 * z
                    - 4.87139289628746 * x**2 * z
                    + 0.487139289628746 * z
                ],
                [
                    5.68329171233537 * y**4 * z
                    - 4.87139289628746 * y**2 * z
                    + 0.487139289628746 * z
                ],
                [
                    5.68329171233537 * x * z**4
                    - 4.87139289628746 * x * z**2
                    + 0.487139289628746 * x
                ],
                [
                    5.68329171233537 * y * z**4
                    - 4.87139289628746 * y * z**2
                    + 0.487139289628746 * y
                ],
                [
                    5.68329171233537 * x**4 * w
                    - 4.87139289628746 * x**2 * w
                    + 0.487139289628746 * w
                ],
                [
                    5.68329171233537 * y**4 * w
                    - 4.87139289628746 * y**2 * w
                    + 0.487139289628746 * w
                ],
                [
                    5.68329171233537 * z**4 * w
                    - 4.87139289628746 * z**2 * w
                    + 0.487139289628746 * w
                ],
                [
                    5.68329171233537 * x * w**4
                    - 4.87139289628746 * x * w**2
                    + 0.487139289628746 * x
                ],
                [
                    5.68329171233537 * y * w**4
                    - 4.87139289628746 * y * w**2
                    + 0.487139289628746 * y
                ],
                [
                    5.68329171233537 * z * w**4
                    - 4.87139289628746 * z * w**2
                    + 0.487139289628746 * z
                ],
                [
                    8.4375 * x**2 * y**2 * z * w
                    - 2.8125 * y**2 * z * w
                    - 2.8125 * x**2 * z * w
                    + 0.9375 * z * w
                ],
                [
                    8.4375 * x**2 * y * z**2 * w
                    - 2.8125 * y * z**2 * w
                    - 2.8125 * x**2 * y * w
                    + 0.9375 * y * w
                ],
                [
                    8.4375 * x * y**2 * z**2 * w
                    - 2.8125 * x * z**2 * w
                    - 2.8125 * x * y**2 * w
                    + 0.9375 * x * w
                ],
                [
                    8.4375 * x**2 * y * z * w**2
                    - 2.8125 * y * z * w**2
                    - 2.8125 * x**2 * y * z
                    + 0.9375 * y * z
                ],
                [
                    8.4375 * x * y**2 * z * w**2
                    - 2.8125 * x * z * w**2
                    - 2.8125 * x * y**2 * z
                    + 0.9375 * x * z
                ],
                [
                    8.4375 * x * y * z**2 * w**2
                    - 2.8125 * x * y * w**2
                    - 2.8125 * x * y * z**2
                    + 0.9375 * x * y
                ],
                [8.5923294280422 * x**3 * y * z * w - 5.15539765682532 * x * y * z * w],
                [8.5923294280422 * x * y**3 * z * w - 5.15539765682532 * x * y * z * w],
                [8.5923294280422 * x * y * z**3 * w - 5.15539765682532 * x * y * z * w],
                [8.5923294280422 * x * y * z * w**3 - 5.15539765682532 * x * y * z * w],
                [9.84375 * x**4 * y * z - 8.4375 * x**2 * y * z + 0.84375 * y * z],
                [9.84375 * x * y**4 * z - 8.4375 * x * y**2 * z + 0.84375 * x * z],
                [9.84375 * x * y * z**4 - 8.4375 * x * y * z**2 + 0.84375 * x * y],
                [9.84375 * x**4 * y * w - 8.4375 * x**2 * y * w + 0.84375 * y * w],
                [9.84375 * x * y**4 * w - 8.4375 * x * y**2 * w + 0.84375 * x * w],
                [9.84375 * x**4 * z * w - 8.4375 * x**2 * z * w + 0.84375 * z * w],
                [9.84375 * y**4 * z * w - 8.4375 * y**2 * z * w + 0.84375 * z * w],
                [9.84375 * x * z**4 * w - 8.4375 * x * z**2 * w + 0.84375 * x * w],
                [9.84375 * y * z**4 * w - 8.4375 * y * z**2 * w + 0.84375 * y * w],
                [9.84375 * x * y * w**4 - 8.4375 * x * y * w**2 + 0.84375 * x * y],
                [9.84375 * x * z * w**4 - 8.4375 * x * z * w**2 + 0.84375 * x * z],
                [9.84375 * y * z * w**4 - 8.4375 * y * z * w**2 + 0.84375 * y * z],
                [
                    17.04987513700614 * x**4 * y * z * w
                    - 14.61417868886241 * x**2 * y * z * w
                    + 1.46141786888624 * y * z * w
                ],
                [
                    17.04987513700614 * x * y**4 * z * w
                    - 14.61417868886241 * x * y**2 * z * w
                    + 1.46141786888624 * x * z * w
                ],
                [
                    17.04987513700614 * x * y * z**4 * w
                    - 14.61417868886241 * x * y * z**2 * w
                    + 1.46141786888624 * x * y * w
                ],
                [
                    17.04987513700614 * x * y * z * w**4
                    - 14.61417868886241 * x * y * z * w**2
                    + 1.46141786888624 * x * y * z
                ],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, interpList.shape[0]):
                for m in range(0, functionVector.shape[0]):
                  interpMatrix[
                      l
                      + k * interpList.shape[0]
                      + j * interpList.shape[0] * interpList.shape[0]
                      + i
                      * interpList.shape[0]
                      * interpList.shape[0]
                      * interpList.shape[0],
                      m,
                  ] = (
                      functionVector[m]
                      .subs(x, interpList[l])
                      .subs(y, interpList[k])
                      .subs(z, interpList[j])
                      .subs(w, interpList[i])
                  )
      else:
        raise NameError(
            "interpMatrix: Order {} is not supported!\nPolynomial order must be <5".format(
                order
            )
        )
    elif modal and basis_type == "tensor":
      if order == 1:
        functionVector = Matrix(
            [
                [0.25],
                [0.4330127018922192 * x],
                [0.4330127018922192 * y],
                [0.4330127018922192 * z],
                [0.4330127018922192 * w],
                [0.75 * x * y],
                [0.75 * x * z],
                [0.75 * y * z],
                [0.75 * x * w],
                [0.75 * y * w],
                [0.75 * z * w],
                [1.299038105676659 * x * y * z],
                [1.299038105676659 * x * y * w],
                [1.299038105676659 * x * z * w],
                [1.299038105676659 * y * z * w],
                [2.25 * x * y * z * w],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, interpList.shape[0]):
                for m in range(0, functionVector.shape[0]):
                  interpMatrix[
                      l
                      + k * interpList.shape[0]
                      + j * interpList.shape[0] * interpList.shape[0]
                      + i
                      * interpList.shape[0]
                      * interpList.shape[0]
                      * interpList.shape[0],
                      m,
                  ] = (
                      functionVector[m]
                      .subs(x, interpList[l])
                      .subs(y, interpList[k])
                      .subs(z, interpList[j])
                      .subs(w, interpList[i])
                  )

      elif order == 2:
        functionVector = Matrix(
            [
                [0.25],
                [0.4330127018922193 * x],
                [0.4330127018922193 * y],
                [0.4330127018922193 * z],
                [0.4330127018922193 * w],
                [0.75 * x * y],
                [0.75 * x * z],
                [0.75 * y * z],
                [0.75 * x * w],
                [0.75 * y * w],
                [0.75 * z * w],
                [0.8385254915624212 * x**2 - 0.2795084971874737],
                [0.8385254915624212 * y**2 - 0.2795084971874737],
                [0.8385254915624212 * z**2 - 0.2795084971874737],
                [0.8385254915624212 * w**2 - 0.2795084971874737],
                [1.299038105676658 * x * y * z],
                [1.299038105676658 * x * y * w],
                [1.299038105676658 * x * z * w],
                [1.299038105676658 * y * z * w],
                [1.452368754827781 * x**2 * y - 0.4841229182759271 * y],
                [1.452368754827781 * x * y**2 - 0.4841229182759271 * x],
                [1.452368754827781 * x**2 * z - 0.4841229182759271 * z],
                [1.452368754827781 * y**2 * z - 0.4841229182759271 * z],
                [1.452368754827781 * x * z**2 - 0.4841229182759271 * x],
                [1.452368754827781 * y * z**2 - 0.4841229182759271 * y],
                [1.452368754827781 * x**2 * w - 0.4841229182759271 * w],
                [1.452368754827781 * y**2 * w - 0.4841229182759271 * w],
                [1.452368754827781 * z**2 * w - 0.4841229182759271 * w],
                [1.452368754827781 * x * w**2 - 0.4841229182759271 * x],
                [1.452368754827781 * y * w**2 - 0.4841229182759271 * y],
                [1.452368754827781 * z * w**2 - 0.4841229182759271 * z],
                [2.25 * x * y * z * w],
                [2.515576474687264 * x**2 * y * z - 0.8385254915624212 * y * z],
                [2.515576474687264 * x * y**2 * z - 0.8385254915624212 * x * z],
                [2.515576474687264 * x * y * z**2 - 0.8385254915624212 * x * y],
                [2.515576474687264 * x**2 * y * w - 0.8385254915624212 * y * w],
                [2.515576474687264 * x * y**2 * w - 0.8385254915624212 * x * w],
                [2.515576474687264 * x**2 * z * w - 0.8385254915624212 * z * w],
                [2.515576474687264 * y**2 * z * w - 0.8385254915624212 * z * w],
                [2.515576474687264 * x * z**2 * w - 0.8385254915624212 * x * w],
                [2.515576474687264 * y * z**2 * w - 0.8385254915624212 * y * w],
                [2.515576474687264 * x * y * w**2 - 0.8385254915624212 * x * y],
                [2.515576474687264 * x * z * w**2 - 0.8385254915624212 * x * z],
                [2.515576474687264 * y * z * w**2 - 0.8385254915624212 * y * z],
                [2.8125 * x**2 * y**2 - 0.9375 * y**2 - 0.9375 * x**2 + 0.3125],
                [2.8125 * x**2 * z**2 - 0.9375 * z**2 - 0.9375 * x**2 + 0.3125],
                [2.8125 * y**2 * z**2 - 0.9375 * z**2 - 0.9375 * y**2 + 0.3125],
                [2.8125 * x**2 * w**2 - 0.9375 * w**2 - 0.9375 * x**2 + 0.3125],
                [2.8125 * y**2 * w**2 - 0.9375 * w**2 - 0.9375 * y**2 + 0.3125],
                [2.8125 * z**2 * w**2 - 0.9375 * w**2 - 0.9375 * z**2 + 0.3125],
                [4.357106264483344 * x**2 * y * z * w - 1.452368754827781 * y * z * w],
                [4.357106264483344 * x * y**2 * z * w - 1.452368754827781 * x * z * w],
                [4.357106264483344 * x * y * z**2 * w - 1.452368754827781 * x * y * w],
                [4.357106264483344 * x * y * z * w**2 - 1.452368754827781 * x * y * z],
                [
                    4.871392896287466 * x**2 * y**2 * z
                    - 1.623797632095822 * y**2 * z
                    - 1.623797632095822 * x**2 * z
                    + 0.541265877365274 * z
                ],
                [
                    4.871392896287466 * x**2 * y * z**2
                    - 1.623797632095822 * y * z**2
                    - 1.623797632095822 * x**2 * y
                    + 0.541265877365274 * y
                ],
                [
                    4.871392896287466 * x * y**2 * z**2
                    - 1.623797632095822 * x * z**2
                    - 1.623797632095822 * x * y**2
                    + 0.541265877365274 * x
                ],
                [
                    4.871392896287466 * x**2 * y**2 * w
                    - 1.623797632095822 * y**2 * w
                    - 1.623797632095822 * x**2 * w
                    + 0.541265877365274 * w
                ],
                [
                    4.871392896287466 * x**2 * z**2 * w
                    - 1.623797632095822 * z**2 * w
                    - 1.623797632095822 * x**2 * w
                    + 0.541265877365274 * w
                ],
                [
                    4.871392896287466 * y**2 * z**2 * w
                    - 1.623797632095822 * z**2 * w
                    - 1.623797632095822 * y**2 * w
                    + 0.541265877365274 * w
                ],
                [
                    4.871392896287466 * x**2 * y * w**2
                    - 1.623797632095822 * y * w**2
                    - 1.623797632095822 * x**2 * y
                    + 0.541265877365274 * y
                ],
                [
                    4.871392896287466 * x * y**2 * w**2
                    - 1.623797632095822 * x * w**2
                    - 1.623797632095822 * x * y**2
                    + 0.541265877365274 * x
                ],
                [
                    4.871392896287466 * x**2 * z * w**2
                    - 1.623797632095822 * z * w**2
                    - 1.623797632095822 * x**2 * z
                    + 0.541265877365274 * z
                ],
                [
                    4.871392896287466 * y**2 * z * w**2
                    - 1.623797632095822 * z * w**2
                    - 1.623797632095822 * y**2 * z
                    + 0.541265877365274 * z
                ],
                [
                    4.871392896287466 * x * z**2 * w**2
                    - 1.623797632095822 * x * w**2
                    - 1.623797632095822 * x * z**2
                    + 0.541265877365274 * x
                ],
                [
                    4.871392896287466 * y * z**2 * w**2
                    - 1.623797632095822 * y * w**2
                    - 1.623797632095822 * y * z**2
                    + 0.541265877365274 * y
                ],
                [
                    8.4375 * x**2 * y**2 * z * w
                    - 2.8125 * y**2 * z * w
                    - 2.8125 * x**2 * z * w
                    + 0.9375 * z * w
                ],
                [
                    8.4375 * x**2 * y * z**2 * w
                    - 2.8125 * y * z**2 * w
                    - 2.8125 * x**2 * y * w
                    + 0.9375 * y * w
                ],
                [
                    8.4375 * x * y**2 * z**2 * w
                    - 2.8125 * x * z**2 * w
                    - 2.8125 * x * y**2 * w
                    + 0.9375 * x * w
                ],
                [
                    8.4375 * x**2 * y * z * w**2
                    - 2.8125 * y * z * w**2
                    - 2.8125 * x**2 * y * z
                    + 0.9375 * y * z
                ],
                [
                    8.4375 * x * y**2 * z * w**2
                    - 2.8125 * x * z * w**2
                    - 2.8125 * x * y**2 * z
                    + 0.9375 * x * z
                ],
                [
                    8.4375 * x * y * z**2 * w**2
                    - 2.8125 * x * y * w**2
                    - 2.8125 * x * y * z**2
                    + 0.9375 * x * y
                ],
                [
                    9.43341178007724 * x**2 * y**2 * z**2
                    - 3.14447059335908 * y**2 * z**2
                    - 3.14447059335908 * x**2 * z**2
                    + 1.048156864453027 * z**2
                    - 3.14447059335908 * x**2 * y**2
                    + 1.048156864453027 * y**2
                    + 1.048156864453027 * x**2
                    - 0.3493856214843422
                ],
                [
                    9.43341178007724 * x**2 * y**2 * w**2
                    - 3.14447059335908 * y**2 * w**2
                    - 3.14447059335908 * x**2 * w**2
                    + 1.048156864453027 * w**2
                    - 3.14447059335908 * x**2 * y**2
                    + 1.048156864453027 * y**2
                    + 1.048156864453027 * x**2
                    - 0.3493856214843422
                ],
                [
                    9.43341178007724 * x**2 * z**2 * w**2
                    - 3.14447059335908 * z**2 * w**2
                    - 3.14447059335908 * x**2 * w**2
                    + 1.048156864453027 * w**2
                    - 3.14447059335908 * x**2 * z**2
                    + 1.048156864453027 * z**2
                    + 1.048156864453027 * x**2
                    - 0.3493856214843422
                ],
                [
                    9.43341178007724 * y**2 * z**2 * w**2
                    - 3.14447059335908 * z**2 * w**2
                    - 3.14447059335908 * y**2 * w**2
                    + 1.048156864453027 * w**2
                    - 3.14447059335908 * y**2 * z**2
                    + 1.048156864453027 * z**2
                    + 1.048156864453027 * y**2
                    - 0.3493856214843422
                ],
                [
                    16.33914849181254 * x**2 * y**2 * z**2 * w
                    - 5.44638283060418 * y**2 * z**2 * w
                    - 5.44638283060418 * x**2 * z**2 * w
                    + 1.815460943534727 * z**2 * w
                    - 5.44638283060418 * x**2 * y**2 * w
                    + 1.815460943534727 * y**2 * w
                    + 1.815460943534727 * x**2 * w
                    - 0.6051536478449089 * w
                ],
                [
                    16.33914849181254 * x**2 * y**2 * z * w**2
                    - 5.44638283060418 * y**2 * z * w**2
                    - 5.44638283060418 * x**2 * z * w**2
                    + 1.815460943534727 * z * w**2
                    - 5.44638283060418 * x**2 * y**2 * z
                    + 1.815460943534727 * y**2 * z
                    + 1.815460943534727 * x**2 * z
                    - 0.6051536478449089 * z
                ],
                [
                    16.33914849181254 * x**2 * y * z**2 * w**2
                    - 5.44638283060418 * y * z**2 * w**2
                    - 5.44638283060418 * x**2 * y * w**2
                    + 1.815460943534727 * y * w**2
                    - 5.44638283060418 * x**2 * y * z**2
                    + 1.815460943534727 * y * z**2
                    + 1.815460943534727 * x**2 * y
                    - 0.6051536478449089 * y
                ],
                [
                    16.33914849181254 * x * y**2 * z**2 * w**2
                    - 5.44638283060418 * x * z**2 * w**2
                    - 5.44638283060418 * x * y**2 * w**2
                    + 1.815460943534727 * x * w**2
                    - 5.44638283060418 * x * y**2 * z**2
                    + 1.815460943534727 * x * z**2
                    + 1.815460943534727 * x * y**2
                    - 0.6051536478449089 * x
                ],
                [
                    31.640625 * x**2 * y**2 * z**2 * w**2
                    - 10.546875 * y**2 * z**2 * w**2
                    - 10.546875 * x**2 * z**2 * w**2
                    + 3.515625 * z**2 * w**2
                    - 10.546875 * x**2 * y**2 * w**2
                    + 3.515625 * y**2 * w**2
                    + 3.515625 * x**2 * w**2
                    - 1.171875 * w**2
                    - 10.546875 * x**2 * y**2 * z**2
                    + 3.515625 * y**2 * z**2
                    + 3.515625 * x**2 * z**2
                    - 1.171875 * z**2
                    + 3.515625 * x**2 * y**2
                    - 1.171875 * y**2
                    - 1.171875 * x**2
                    + 0.390625
                ],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, interpList.shape[0]):
                for m in range(0, functionVector.shape[0]):
                  interpMatrix[
                      l
                      + k * interpList.shape[0]
                      + j * interpList.shape[0] * interpList.shape[0]
                      + i
                      * interpList.shape[0]
                      * interpList.shape[0]
                      * interpList.shape[0],
                      m,
                  ] = (
                      functionVector[m]
                      .subs(x, interpList[l])
                      .subs(y, interpList[k])
                      .subs(z, interpList[j])
                      .subs(w, interpList[i])
                  )
      else:
        raise NameError(
            "interpMatrix: Order {} is not supported!\nPolynomial order must be <3".format(
                order
            )
        )

    elif modal and basis_type == "gkhybrid":
      if order == 1:
        functionVector = Matrix(
            [
                [0.25],
                [0.4330127018922193 * x],
                [0.4330127018922193 * y],
                [0.4330127018922193 * z],
                [0.4330127018922193 * w],
                [0.75 * x * y],
                [0.75 * x * z],
                [0.75 * y * z],
                [0.75 * w * x],
                [0.75 * w * y],
                [0.75 * w * z],
                [1.299038105676658 * x * y * z],
                [1.299038105676658 * w * x * y],
                [1.299038105676658 * w * x * z],
                [1.299038105676658 * w * y * z],
                [2.25 * w * x * y * z],
                [0.8385254915624212 * (z**2 - 0.3333333333333333)],
                [1.452368754827781 * (x * z**2 - 0.3333333333333333 * x)],
                [1.452368754827781 * (y * z**2 - 0.3333333333333333 * y)],
                [1.452368754827781 * (w * z**2 - 0.3333333333333333 * w)],
                [2.515576474687264 * (x * y * z**2 - 0.3333333333333333 * x * y)],
                [2.515576474687264 * (w * x * z**2 - 0.3333333333333333 * w * x)],
                [2.515576474687264 * (w * y * z**2 - 0.3333333333333333 * w * y)],
                [
                    4.357106264483344
                    * (w * x * y * z**2 - 0.3333333333333333 * w * x * y)
                ],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpListND[0].shape[0]
                * interpListND[1].shape[0]
                * interpListND[2].shape[0]
                * interpListND[3].shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpListND[3].shape[0]):
          for j in range(0, interpListND[2].shape[0]):
            for k in range(0, interpListND[1].shape[0]):
              for l in range(0, interpListND[0].shape[0]):
                for m in range(0, functionVector.shape[0]):
                  interpMatrix[
                      l
                      + k * interpListND[0].shape[0]
                      + j * interpListND[1].shape[0] * interpListND[0].shape[0]
                      + i
                      * interpListND[2].shape[0]
                      * interpListND[1].shape[0]
                      * interpListND[0].shape[0],
                      m,
                  ] = (
                      functionVector[m]
                      .subs(x, interpListND[0][l])
                      .subs(y, interpListND[1][k])
                      .subs(z, interpListND[2][j])
                      .subs(w, interpListND[3][i])
                  )

      else:
        raise NameError(
            "interpMatrix: Order {} is not supported!\nPolynomial order must be =1".format(
                order
            )
        )

    elif modal and basis_type == "hybrid":
      if order == 1:
        functionVector = Matrix(
            [
                [0.25],
                [0.4330127018922194 * x],
                [0.4330127018922194 * y],
                [0.4330127018922194 * z],
                [0.4330127018922194 * w],
                [0.75 * x * y],
                [0.75 * x * z],
                [0.75 * y * z],
                [0.75 * w * x],
                [0.75 * w * y],
                [0.75 * w * z],
                [1.299038105676658 * x * y * z],
                [1.299038105676658 * w * x * y],
                [1.299038105676658 * w * x * z],
                [1.299038105676658 * w * y * z],
                [2.25 * w * x * y * z],
                [0.8385254915624211 * (w**2 - 0.3333333333333333)],
                [1.452368754827781 * (w**2 * x - 0.3333333333333333 * x)],
                [1.452368754827781 * (w**2 * y - 0.3333333333333333 * y)],
                [1.452368754827781 * (w**2 * z - 0.3333333333333333 * z)],
                [2.515576474687264 * (w**2 * x * y - 0.3333333333333333 * x * y)],
                [2.515576474687264 * (w**2 * x * z - 0.3333333333333333 * x * z)],
                [2.515576474687264 * (w**2 * y * z - 0.3333333333333333 * y * z)],
                [
                    4.357106264483344
                    * (w**2 * x * y * z - 0.3333333333333333 * x * y * z)
                ],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpListND[0].shape[0]
                * interpListND[1].shape[0]
                * interpListND[2].shape[0]
                * interpListND[3].shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpListND[3].shape[0]):
          for j in range(0, interpListND[2].shape[0]):
            for k in range(0, interpListND[1].shape[0]):
              for l in range(0, interpListND[0].shape[0]):
                for m in range(0, functionVector.shape[0]):
                  interpMatrix[
                      l
                      + k * interpListND[0].shape[0]
                      + j * interpListND[1].shape[0] * interpListND[0].shape[0]
                      + i
                      * interpListND[2].shape[0]
                      * interpListND[1].shape[0]
                      * interpListND[0].shape[0],
                      m,
                  ] = (
                      functionVector[m]
                      .subs(x, interpListND[0][l])
                      .subs(y, interpListND[1][k])
                      .subs(z, interpListND[2][j])
                      .subs(w, interpListND[3][i])
                  )
      else:
        raise NameError(
            "interpMatrix: Order {} is not supported!\nPolynomial order must be =1".format(
                order
            )
        )

    elif modal == False and basis_type == "serendipity":
      if order == 1:
        functionVector = Matrix(
            [
                [
                    (w * x) / 16.0
                    - x / 16.0
                    - y / 16.0
                    - z / 16.0
                    - w / 16.0
                    + (w * y) / 16.0
                    + (w * z) / 16.0
                    + (x * y) / 16.0
                    + (x * z) / 16.0
                    + (y * z) / 16.0
                    - (w * x * y) / 16.0
                    - (w * x * z) / 16.0
                    - (w * y * z) / 16.0
                    - (x * y * z) / 16.0
                    + (w * x * y * z) / 16.0
                    + 1.0 / 16.0
                ],
                [
                    x / 16.0
                    - w / 16.0
                    - y / 16.0
                    - z / 16.0
                    - (w * x) / 16.0
                    + (w * y) / 16.0
                    + (w * z) / 16.0
                    - (x * y) / 16.0
                    - (x * z) / 16.0
                    + (y * z) / 16.0
                    + (w * x * y) / 16.0
                    + (w * x * z) / 16.0
                    - (w * y * z) / 16.0
                    + (x * y * z) / 16.0
                    - (w * x * y * z) / 16.0
                    + 1.0 / 16.0
                ],
                [
                    y / 16.0
                    - x / 16.0
                    - w / 16.0
                    - z / 16.0
                    + (w * x) / 16.0
                    - (w * y) / 16.0
                    + (w * z) / 16.0
                    - (x * y) / 16.0
                    + (x * z) / 16.0
                    - (y * z) / 16.0
                    + (w * x * y) / 16.0
                    - (w * x * z) / 16.0
                    + (w * y * z) / 16.0
                    + (x * y * z) / 16.0
                    - (w * x * y * z) / 16.0
                    + 1.0 / 16.0
                ],
                [
                    x / 16.0
                    - w / 16.0
                    + y / 16.0
                    - z / 16.0
                    - (w * x) / 16.0
                    - (w * y) / 16.0
                    + (w * z) / 16.0
                    + (x * y) / 16.0
                    - (x * z) / 16.0
                    - (y * z) / 16.0
                    - (w * x * y) / 16.0
                    + (w * x * z) / 16.0
                    + (w * y * z) / 16.0
                    - (x * y * z) / 16.0
                    + (w * x * y * z) / 16.0
                    + 1.0 / 16.0
                ],
                [
                    z / 16.0
                    - x / 16.0
                    - y / 16.0
                    - w / 16.0
                    + (w * x) / 16.0
                    + (w * y) / 16.0
                    - (w * z) / 16.0
                    + (x * y) / 16.0
                    - (x * z) / 16.0
                    - (y * z) / 16.0
                    - (w * x * y) / 16.0
                    + (w * x * z) / 16.0
                    + (w * y * z) / 16.0
                    + (x * y * z) / 16.0
                    - (w * x * y * z) / 16.0
                    + 1.0 / 16.0
                ],
                [
                    x / 16.0
                    - w / 16.0
                    - y / 16.0
                    + z / 16.0
                    - (w * x) / 16.0
                    + (w * y) / 16.0
                    - (w * z) / 16.0
                    - (x * y) / 16.0
                    + (x * z) / 16.0
                    - (y * z) / 16.0
                    + (w * x * y) / 16.0
                    - (w * x * z) / 16.0
                    + (w * y * z) / 16.0
                    - (x * y * z) / 16.0
                    + (w * x * y * z) / 16.0
                    + 1.0 / 16.0
                ],
                [
                    y / 16.0
                    - x / 16.0
                    - w / 16.0
                    + z / 16.0
                    + (w * x) / 16.0
                    - (w * y) / 16.0
                    - (w * z) / 16.0
                    - (x * y) / 16.0
                    - (x * z) / 16.0
                    + (y * z) / 16.0
                    + (w * x * y) / 16.0
                    + (w * x * z) / 16.0
                    - (w * y * z) / 16.0
                    - (x * y * z) / 16.0
                    + (w * x * y * z) / 16.0
                    + 1.0 / 16.0
                ],
                [
                    x / 16.0
                    - w / 16.0
                    + y / 16.0
                    + z / 16.0
                    - (w * x) / 16.0
                    - (w * y) / 16.0
                    - (w * z) / 16.0
                    + (x * y) / 16.0
                    + (x * z) / 16.0
                    + (y * z) / 16.0
                    - (w * x * y) / 16.0
                    - (w * x * z) / 16.0
                    - (w * y * z) / 16.0
                    + (x * y * z) / 16.0
                    - (w * x * y * z) / 16.0
                    + 1.0 / 16.0
                ],
                [
                    w / 16.0
                    - x / 16.0
                    - y / 16.0
                    - z / 16.0
                    - (w * x) / 16.0
                    - (w * y) / 16.0
                    - (w * z) / 16.0
                    + (x * y) / 16.0
                    + (x * z) / 16.0
                    + (y * z) / 16.0
                    + (w * x * y) / 16.0
                    + (w * x * z) / 16.0
                    + (w * y * z) / 16.0
                    - (x * y * z) / 16.0
                    - (w * x * y * z) / 16.0
                    + 1.0 / 16.0
                ],
                [
                    w / 16.0
                    + x / 16.0
                    - y / 16.0
                    - z / 16.0
                    + (w * x) / 16.0
                    - (w * y) / 16.0
                    - (w * z) / 16.0
                    - (x * y) / 16.0
                    - (x * z) / 16.0
                    + (y * z) / 16.0
                    - (w * x * y) / 16.0
                    - (w * x * z) / 16.0
                    + (w * y * z) / 16.0
                    + (x * y * z) / 16.0
                    + (w * x * y * z) / 16.0
                    + 1.0 / 16.0
                ],
                [
                    w / 16.0
                    - x / 16.0
                    + y / 16.0
                    - z / 16.0
                    - (w * x) / 16.0
                    + (w * y) / 16.0
                    - (w * z) / 16.0
                    - (x * y) / 16.0
                    + (x * z) / 16.0
                    - (y * z) / 16.0
                    - (w * x * y) / 16.0
                    + (w * x * z) / 16.0
                    - (w * y * z) / 16.0
                    + (x * y * z) / 16.0
                    + (w * x * y * z) / 16.0
                    + 1.0 / 16.0
                ],
                [
                    w / 16.0
                    + x / 16.0
                    + y / 16.0
                    - z / 16.0
                    + (w * x) / 16.0
                    + (w * y) / 16.0
                    - (w * z) / 16.0
                    + (x * y) / 16.0
                    - (x * z) / 16.0
                    - (y * z) / 16.0
                    + (w * x * y) / 16.0
                    - (w * x * z) / 16.0
                    - (w * y * z) / 16.0
                    - (x * y * z) / 16.0
                    - (w * x * y * z) / 16.0
                    + 1.0 / 16.0
                ],
                [
                    w / 16.0
                    - x / 16.0
                    - y / 16.0
                    + z / 16.0
                    - (w * x) / 16.0
                    - (w * y) / 16.0
                    + (w * z) / 16.0
                    + (x * y) / 16.0
                    - (x * z) / 16.0
                    - (y * z) / 16.0
                    + (w * x * y) / 16.0
                    - (w * x * z) / 16.0
                    - (w * y * z) / 16.0
                    + (x * y * z) / 16.0
                    + (w * x * y * z) / 16.0
                    + 1.0 / 16.0
                ],
                [
                    w / 16.0
                    + x / 16.0
                    - y / 16.0
                    + z / 16.0
                    + (w * x) / 16.0
                    - (w * y) / 16.0
                    + (w * z) / 16.0
                    - (x * y) / 16.0
                    + (x * z) / 16.0
                    - (y * z) / 16.0
                    - (w * x * y) / 16.0
                    + (w * x * z) / 16.0
                    - (w * y * z) / 16.0
                    - (x * y * z) / 16.0
                    - (w * x * y * z) / 16.0
                    + 1.0 / 16.0
                ],
                [
                    w / 16.0
                    - x / 16.0
                    + y / 16.0
                    + z / 16.0
                    - (w * x) / 16.0
                    + (w * y) / 16.0
                    + (w * z) / 16.0
                    - (x * y) / 16.0
                    - (x * z) / 16.0
                    + (y * z) / 16.0
                    - (w * x * y) / 16.0
                    - (w * x * z) / 16.0
                    + (w * y * z) / 16.0
                    - (x * y * z) / 16.0
                    - (w * x * y * z) / 16.0
                    + 1.0 / 16.0
                ],
                [
                    w / 16.0
                    + x / 16.0
                    + y / 16.0
                    + z / 16.0
                    + (w * x) / 16.0
                    + (w * y) / 16.0
                    + (w * z) / 16.0
                    + (x * y) / 16.0
                    + (x * z) / 16.0
                    + (y * z) / 16.0
                    + (w * x * y) / 16.0
                    + (w * x * z) / 16.0
                    + (w * y * z) / 16.0
                    + (x * y * z) / 16.0
                    + (w * x * y * z) / 16.0
                    + 1.0 / 16.0
                ],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, interpList.shape[0]):
                for m in range(0, functionVector.shape[0]):
                  interpMatrix[
                      l
                      + k * interpList.shape[0]
                      + j * interpList.shape[0] * interpList.shape[0]
                      + i
                      * interpList.shape[0]
                      * interpList.shape[0]
                      * interpList.shape[0],
                      m,
                  ] = (
                      functionVector[m]
                      .subs(x, interpList[l])
                      .subs(y, interpList[k])
                      .subs(z, interpList[j])
                      .subs(w, interpList[i])
                  )

      elif order == 2:
        functionVector = Matrix(
            [
                [
                    -(w**2 * x * y * z) / 16.0
                    + (w**2 * x * y) / 16.0
                    + (w**2 * x * z) / 16.0
                    - (w**2 * x) / 16.0
                    + (w**2 * y * z) / 16.0
                    - (w**2 * y) / 16.0
                    - (w**2 * z) / 16.0
                    + w**2 / 16.0
                    - (w * x**2 * y * z) / 16.0
                    + (w * x**2 * y) / 16.0
                    + (w * x**2 * z) / 16.0
                    - (w * x**2) / 16.0
                    - (w * x * y**2 * z) / 16.0
                    + (w * x * y**2) / 16.0
                    - (w * x * y * z**2) / 16.0
                    + (w * x * y * z) / 16.0
                    + (w * x * z**2) / 16.0
                    - (w * x) / 16.0
                    + (w * y**2 * z) / 16.0
                    - (w * y**2) / 16.0
                    + (w * y * z**2) / 16.0
                    - (w * y) / 16.0
                    - (w * z**2) / 16.0
                    - (w * z) / 16.0
                    + w / 8.0
                    + (x**2 * y * z) / 16.0
                    - (x**2 * y) / 16.0
                    - (x**2 * z) / 16.0
                    + x**2 / 16.0
                    + (x * y**2 * z) / 16.0
                    - (x * y**2) / 16.0
                    + (x * y * z**2) / 16.0
                    - (x * y) / 16.0
                    - (x * z**2) / 16.0
                    - (x * z) / 16.0
                    + x / 8.0
                    - (y**2 * z) / 16.0
                    + y**2 / 16.0
                    - (y * z**2) / 16.0
                    - (y * z) / 16.0
                    + y / 8.0
                    + z**2 / 16.0
                    + z / 8.0
                    - 3.0 / 16.0
                ],
                [
                    (w * y) / 8.0
                    - y / 8.0
                    - z / 8.0
                    - w / 8.0
                    + (w * z) / 8.0
                    + (y * z) / 8.0
                    + (w * x**2) / 8.0
                    + (x**2 * y) / 8.0
                    + (x**2 * z) / 8.0
                    - x**2 / 8.0
                    - (w * x**2 * y) / 8.0
                    - (w * x**2 * z) / 8.0
                    - (x**2 * y * z) / 8.0
                    - (w * y * z) / 8.0
                    + (w * x**2 * y * z) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    (w**2 * x * y * z) / 16.0
                    - (w**2 * x * y) / 16.0
                    - (w**2 * x * z) / 16.0
                    + (w**2 * x) / 16.0
                    + (w**2 * y * z) / 16.0
                    - (w**2 * y) / 16.0
                    - (w**2 * z) / 16.0
                    + w**2 / 16.0
                    - (w * x**2 * y * z) / 16.0
                    + (w * x**2 * y) / 16.0
                    + (w * x**2 * z) / 16.0
                    - (w * x**2) / 16.0
                    + (w * x * y**2 * z) / 16.0
                    - (w * x * y**2) / 16.0
                    + (w * x * y * z**2) / 16.0
                    - (w * x * y * z) / 16.0
                    - (w * x * z**2) / 16.0
                    + (w * x) / 16.0
                    + (w * y**2 * z) / 16.0
                    - (w * y**2) / 16.0
                    + (w * y * z**2) / 16.0
                    - (w * y) / 16.0
                    - (w * z**2) / 16.0
                    - (w * z) / 16.0
                    + w / 8.0
                    + (x**2 * y * z) / 16.0
                    - (x**2 * y) / 16.0
                    - (x**2 * z) / 16.0
                    + x**2 / 16.0
                    - (x * y**2 * z) / 16.0
                    + (x * y**2) / 16.0
                    - (x * y * z**2) / 16.0
                    + (x * y) / 16.0
                    + (x * z**2) / 16.0
                    + (x * z) / 16.0
                    - x / 8.0
                    - (y**2 * z) / 16.0
                    + y**2 / 16.0
                    - (y * z**2) / 16.0
                    - (y * z) / 16.0
                    + y / 8.0
                    + z**2 / 16.0
                    + z / 8.0
                    - 3.0 / 16.0
                ],
                [
                    (w * x) / 8.0
                    - x / 8.0
                    - z / 8.0
                    - w / 8.0
                    + (w * z) / 8.0
                    + (x * z) / 8.0
                    + (w * y**2) / 8.0
                    + (x * y**2) / 8.0
                    + (y**2 * z) / 8.0
                    - y**2 / 8.0
                    - (w * x * y**2) / 8.0
                    - (w * y**2 * z) / 8.0
                    - (x * y**2 * z) / 8.0
                    - (w * x * z) / 8.0
                    + (w * x * y**2 * z) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    x / 8.0
                    - w / 8.0
                    - z / 8.0
                    - (w * x) / 8.0
                    + (w * z) / 8.0
                    - (x * z) / 8.0
                    + (w * y**2) / 8.0
                    - (x * y**2) / 8.0
                    + (y**2 * z) / 8.0
                    - y**2 / 8.0
                    + (w * x * y**2) / 8.0
                    - (w * y**2 * z) / 8.0
                    + (x * y**2 * z) / 8.0
                    + (w * x * z) / 8.0
                    - (w * x * y**2 * z) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    (w**2 * x * y * z) / 16.0
                    - (w**2 * x * y) / 16.0
                    + (w**2 * x * z) / 16.0
                    - (w**2 * x) / 16.0
                    - (w**2 * y * z) / 16.0
                    + (w**2 * y) / 16.0
                    - (w**2 * z) / 16.0
                    + w**2 / 16.0
                    + (w * x**2 * y * z) / 16.0
                    - (w * x**2 * y) / 16.0
                    + (w * x**2 * z) / 16.0
                    - (w * x**2) / 16.0
                    - (w * x * y**2 * z) / 16.0
                    + (w * x * y**2) / 16.0
                    + (w * x * y * z**2) / 16.0
                    - (w * x * y * z) / 16.0
                    + (w * x * z**2) / 16.0
                    - (w * x) / 16.0
                    + (w * y**2 * z) / 16.0
                    - (w * y**2) / 16.0
                    - (w * y * z**2) / 16.0
                    + (w * y) / 16.0
                    - (w * z**2) / 16.0
                    - (w * z) / 16.0
                    + w / 8.0
                    - (x**2 * y * z) / 16.0
                    + (x**2 * y) / 16.0
                    - (x**2 * z) / 16.0
                    + x**2 / 16.0
                    + (x * y**2 * z) / 16.0
                    - (x * y**2) / 16.0
                    - (x * y * z**2) / 16.0
                    + (x * y) / 16.0
                    - (x * z**2) / 16.0
                    - (x * z) / 16.0
                    + x / 8.0
                    - (y**2 * z) / 16.0
                    + y**2 / 16.0
                    + (y * z**2) / 16.0
                    + (y * z) / 16.0
                    - y / 8.0
                    + z**2 / 16.0
                    + z / 8.0
                    - 3.0 / 16.0
                ],
                [
                    y / 8.0
                    - w / 8.0
                    - z / 8.0
                    - (w * y) / 8.0
                    + (w * z) / 8.0
                    - (y * z) / 8.0
                    + (w * x**2) / 8.0
                    - (x**2 * y) / 8.0
                    + (x**2 * z) / 8.0
                    - x**2 / 8.0
                    + (w * x**2 * y) / 8.0
                    - (w * x**2 * z) / 8.0
                    + (x**2 * y * z) / 8.0
                    + (w * y * z) / 8.0
                    - (w * x**2 * y * z) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    -(w**2 * x * y * z) / 16.0
                    + (w**2 * x * y) / 16.0
                    - (w**2 * x * z) / 16.0
                    + (w**2 * x) / 16.0
                    - (w**2 * y * z) / 16.0
                    + (w**2 * y) / 16.0
                    - (w**2 * z) / 16.0
                    + w**2 / 16.0
                    + (w * x**2 * y * z) / 16.0
                    - (w * x**2 * y) / 16.0
                    + (w * x**2 * z) / 16.0
                    - (w * x**2) / 16.0
                    + (w * x * y**2 * z) / 16.0
                    - (w * x * y**2) / 16.0
                    - (w * x * y * z**2) / 16.0
                    + (w * x * y * z) / 16.0
                    - (w * x * z**2) / 16.0
                    + (w * x) / 16.0
                    + (w * y**2 * z) / 16.0
                    - (w * y**2) / 16.0
                    - (w * y * z**2) / 16.0
                    + (w * y) / 16.0
                    - (w * z**2) / 16.0
                    - (w * z) / 16.0
                    + w / 8.0
                    - (x**2 * y * z) / 16.0
                    + (x**2 * y) / 16.0
                    - (x**2 * z) / 16.0
                    + x**2 / 16.0
                    - (x * y**2 * z) / 16.0
                    + (x * y**2) / 16.0
                    + (x * y * z**2) / 16.0
                    - (x * y) / 16.0
                    + (x * z**2) / 16.0
                    + (x * z) / 16.0
                    - x / 8.0
                    - (y**2 * z) / 16.0
                    + y**2 / 16.0
                    + (y * z**2) / 16.0
                    + (y * z) / 16.0
                    - y / 8.0
                    + z**2 / 16.0
                    + z / 8.0
                    - 3.0 / 16.0
                ],
                [
                    (w * x) / 8.0
                    - x / 8.0
                    - y / 8.0
                    - w / 8.0
                    + (w * y) / 8.0
                    + (x * y) / 8.0
                    + (w * z**2) / 8.0
                    + (x * z**2) / 8.0
                    + (y * z**2) / 8.0
                    - z**2 / 8.0
                    - (w * x * z**2) / 8.0
                    - (w * y * z**2) / 8.0
                    - (x * y * z**2) / 8.0
                    - (w * x * y) / 8.0
                    + (w * x * y * z**2) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    x / 8.0
                    - w / 8.0
                    - y / 8.0
                    - (w * x) / 8.0
                    + (w * y) / 8.0
                    - (x * y) / 8.0
                    + (w * z**2) / 8.0
                    - (x * z**2) / 8.0
                    + (y * z**2) / 8.0
                    - z**2 / 8.0
                    + (w * x * z**2) / 8.0
                    - (w * y * z**2) / 8.0
                    + (x * y * z**2) / 8.0
                    + (w * x * y) / 8.0
                    - (w * x * y * z**2) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    y / 8.0
                    - x / 8.0
                    - w / 8.0
                    + (w * x) / 8.0
                    - (w * y) / 8.0
                    - (x * y) / 8.0
                    + (w * z**2) / 8.0
                    + (x * z**2) / 8.0
                    - (y * z**2) / 8.0
                    - z**2 / 8.0
                    - (w * x * z**2) / 8.0
                    + (w * y * z**2) / 8.0
                    + (x * y * z**2) / 8.0
                    + (w * x * y) / 8.0
                    - (w * x * y * z**2) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    x / 8.0
                    - w / 8.0
                    + y / 8.0
                    - (w * x) / 8.0
                    - (w * y) / 8.0
                    + (x * y) / 8.0
                    + (w * z**2) / 8.0
                    - (x * z**2) / 8.0
                    - (y * z**2) / 8.0
                    - z**2 / 8.0
                    + (w * x * z**2) / 8.0
                    + (w * y * z**2) / 8.0
                    - (x * y * z**2) / 8.0
                    - (w * x * y) / 8.0
                    + (w * x * y * z**2) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    (w**2 * x * y * z) / 16.0
                    + (w**2 * x * y) / 16.0
                    - (w**2 * x * z) / 16.0
                    - (w**2 * x) / 16.0
                    - (w**2 * y * z) / 16.0
                    - (w**2 * y) / 16.0
                    + (w**2 * z) / 16.0
                    + w**2 / 16.0
                    + (w * x**2 * y * z) / 16.0
                    + (w * x**2 * y) / 16.0
                    - (w * x**2 * z) / 16.0
                    - (w * x**2) / 16.0
                    + (w * x * y**2 * z) / 16.0
                    + (w * x * y**2) / 16.0
                    - (w * x * y * z**2) / 16.0
                    - (w * x * y * z) / 16.0
                    + (w * x * z**2) / 16.0
                    - (w * x) / 16.0
                    - (w * y**2 * z) / 16.0
                    - (w * y**2) / 16.0
                    + (w * y * z**2) / 16.0
                    - (w * y) / 16.0
                    - (w * z**2) / 16.0
                    + (w * z) / 16.0
                    + w / 8.0
                    - (x**2 * y * z) / 16.0
                    - (x**2 * y) / 16.0
                    + (x**2 * z) / 16.0
                    + x**2 / 16.0
                    - (x * y**2 * z) / 16.0
                    - (x * y**2) / 16.0
                    + (x * y * z**2) / 16.0
                    - (x * y) / 16.0
                    - (x * z**2) / 16.0
                    + (x * z) / 16.0
                    + x / 8.0
                    + (y**2 * z) / 16.0
                    + y**2 / 16.0
                    - (y * z**2) / 16.0
                    + (y * z) / 16.0
                    + y / 8.0
                    + z**2 / 16.0
                    - z / 8.0
                    - 3.0 / 16.0
                ],
                [
                    z / 8.0
                    - y / 8.0
                    - w / 8.0
                    + (w * y) / 8.0
                    - (w * z) / 8.0
                    - (y * z) / 8.0
                    + (w * x**2) / 8.0
                    + (x**2 * y) / 8.0
                    - (x**2 * z) / 8.0
                    - x**2 / 8.0
                    - (w * x**2 * y) / 8.0
                    + (w * x**2 * z) / 8.0
                    + (x**2 * y * z) / 8.0
                    + (w * y * z) / 8.0
                    - (w * x**2 * y * z) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    -(w**2 * x * y * z) / 16.0
                    - (w**2 * x * y) / 16.0
                    + (w**2 * x * z) / 16.0
                    + (w**2 * x) / 16.0
                    - (w**2 * y * z) / 16.0
                    - (w**2 * y) / 16.0
                    + (w**2 * z) / 16.0
                    + w**2 / 16.0
                    + (w * x**2 * y * z) / 16.0
                    + (w * x**2 * y) / 16.0
                    - (w * x**2 * z) / 16.0
                    - (w * x**2) / 16.0
                    - (w * x * y**2 * z) / 16.0
                    - (w * x * y**2) / 16.0
                    + (w * x * y * z**2) / 16.0
                    + (w * x * y * z) / 16.0
                    - (w * x * z**2) / 16.0
                    + (w * x) / 16.0
                    - (w * y**2 * z) / 16.0
                    - (w * y**2) / 16.0
                    + (w * y * z**2) / 16.0
                    - (w * y) / 16.0
                    - (w * z**2) / 16.0
                    + (w * z) / 16.0
                    + w / 8.0
                    - (x**2 * y * z) / 16.0
                    - (x**2 * y) / 16.0
                    + (x**2 * z) / 16.0
                    + x**2 / 16.0
                    + (x * y**2 * z) / 16.0
                    + (x * y**2) / 16.0
                    - (x * y * z**2) / 16.0
                    + (x * y) / 16.0
                    + (x * z**2) / 16.0
                    - (x * z) / 16.0
                    - x / 8.0
                    + (y**2 * z) / 16.0
                    + y**2 / 16.0
                    - (y * z**2) / 16.0
                    + (y * z) / 16.0
                    + y / 8.0
                    + z**2 / 16.0
                    - z / 8.0
                    - 3.0 / 16.0
                ],
                [
                    z / 8.0
                    - x / 8.0
                    - w / 8.0
                    + (w * x) / 8.0
                    - (w * z) / 8.0
                    - (x * z) / 8.0
                    + (w * y**2) / 8.0
                    + (x * y**2) / 8.0
                    - (y**2 * z) / 8.0
                    - y**2 / 8.0
                    - (w * x * y**2) / 8.0
                    + (w * y**2 * z) / 8.0
                    + (x * y**2 * z) / 8.0
                    + (w * x * z) / 8.0
                    - (w * x * y**2 * z) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    x / 8.0
                    - w / 8.0
                    + z / 8.0
                    - (w * x) / 8.0
                    - (w * z) / 8.0
                    + (x * z) / 8.0
                    + (w * y**2) / 8.0
                    - (x * y**2) / 8.0
                    - (y**2 * z) / 8.0
                    - y**2 / 8.0
                    + (w * x * y**2) / 8.0
                    + (w * y**2 * z) / 8.0
                    - (x * y**2 * z) / 8.0
                    - (w * x * z) / 8.0
                    + (w * x * y**2 * z) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    -(w**2 * x * y * z) / 16.0
                    - (w**2 * x * y) / 16.0
                    - (w**2 * x * z) / 16.0
                    - (w**2 * x) / 16.0
                    + (w**2 * y * z) / 16.0
                    + (w**2 * y) / 16.0
                    + (w**2 * z) / 16.0
                    + w**2 / 16.0
                    - (w * x**2 * y * z) / 16.0
                    - (w * x**2 * y) / 16.0
                    - (w * x**2 * z) / 16.0
                    - (w * x**2) / 16.0
                    + (w * x * y**2 * z) / 16.0
                    + (w * x * y**2) / 16.0
                    + (w * x * y * z**2) / 16.0
                    + (w * x * y * z) / 16.0
                    + (w * x * z**2) / 16.0
                    - (w * x) / 16.0
                    - (w * y**2 * z) / 16.0
                    - (w * y**2) / 16.0
                    - (w * y * z**2) / 16.0
                    + (w * y) / 16.0
                    - (w * z**2) / 16.0
                    + (w * z) / 16.0
                    + w / 8.0
                    + (x**2 * y * z) / 16.0
                    + (x**2 * y) / 16.0
                    + (x**2 * z) / 16.0
                    + x**2 / 16.0
                    - (x * y**2 * z) / 16.0
                    - (x * y**2) / 16.0
                    - (x * y * z**2) / 16.0
                    + (x * y) / 16.0
                    - (x * z**2) / 16.0
                    + (x * z) / 16.0
                    + x / 8.0
                    + (y**2 * z) / 16.0
                    + y**2 / 16.0
                    + (y * z**2) / 16.0
                    - (y * z) / 16.0
                    - y / 8.0
                    + z**2 / 16.0
                    - z / 8.0
                    - 3.0 / 16.0
                ],
                [
                    y / 8.0
                    - w / 8.0
                    + z / 8.0
                    - (w * y) / 8.0
                    - (w * z) / 8.0
                    + (y * z) / 8.0
                    + (w * x**2) / 8.0
                    - (x**2 * y) / 8.0
                    - (x**2 * z) / 8.0
                    - x**2 / 8.0
                    + (w * x**2 * y) / 8.0
                    + (w * x**2 * z) / 8.0
                    - (x**2 * y * z) / 8.0
                    - (w * y * z) / 8.0
                    + (w * x**2 * y * z) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    (w**2 * x * y * z) / 16.0
                    + (w**2 * x * y) / 16.0
                    + (w**2 * x * z) / 16.0
                    + (w**2 * x) / 16.0
                    + (w**2 * y * z) / 16.0
                    + (w**2 * y) / 16.0
                    + (w**2 * z) / 16.0
                    + w**2 / 16.0
                    - (w * x**2 * y * z) / 16.0
                    - (w * x**2 * y) / 16.0
                    - (w * x**2 * z) / 16.0
                    - (w * x**2) / 16.0
                    - (w * x * y**2 * z) / 16.0
                    - (w * x * y**2) / 16.0
                    - (w * x * y * z**2) / 16.0
                    - (w * x * y * z) / 16.0
                    - (w * x * z**2) / 16.0
                    + (w * x) / 16.0
                    - (w * y**2 * z) / 16.0
                    - (w * y**2) / 16.0
                    - (w * y * z**2) / 16.0
                    + (w * y) / 16.0
                    - (w * z**2) / 16.0
                    + (w * z) / 16.0
                    + w / 8.0
                    + (x**2 * y * z) / 16.0
                    + (x**2 * y) / 16.0
                    + (x**2 * z) / 16.0
                    + x**2 / 16.0
                    + (x * y**2 * z) / 16.0
                    + (x * y**2) / 16.0
                    + (x * y * z**2) / 16.0
                    - (x * y) / 16.0
                    + (x * z**2) / 16.0
                    - (x * z) / 16.0
                    - x / 8.0
                    + (y**2 * z) / 16.0
                    + y**2 / 16.0
                    + (y * z**2) / 16.0
                    - (y * z) / 16.0
                    - y / 8.0
                    + z**2 / 16.0
                    - z / 8.0
                    - 3.0 / 16.0
                ],
                [
                    (x * y) / 8.0
                    - y / 8.0
                    - z / 8.0
                    - x / 8.0
                    + (x * z) / 8.0
                    + (y * z) / 8.0
                    + (w**2 * x) / 8.0
                    + (w**2 * y) / 8.0
                    + (w**2 * z) / 8.0
                    - w**2 / 8.0
                    - (w**2 * x * y) / 8.0
                    - (w**2 * x * z) / 8.0
                    - (w**2 * y * z) / 8.0
                    - (x * y * z) / 8.0
                    + (w**2 * x * y * z) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    x / 8.0
                    - y / 8.0
                    - z / 8.0
                    - (x * y) / 8.0
                    - (x * z) / 8.0
                    + (y * z) / 8.0
                    - (w**2 * x) / 8.0
                    + (w**2 * y) / 8.0
                    + (w**2 * z) / 8.0
                    - w**2 / 8.0
                    + (w**2 * x * y) / 8.0
                    + (w**2 * x * z) / 8.0
                    - (w**2 * y * z) / 8.0
                    + (x * y * z) / 8.0
                    - (w**2 * x * y * z) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    y / 8.0
                    - x / 8.0
                    - z / 8.0
                    - (x * y) / 8.0
                    + (x * z) / 8.0
                    - (y * z) / 8.0
                    + (w**2 * x) / 8.0
                    - (w**2 * y) / 8.0
                    + (w**2 * z) / 8.0
                    - w**2 / 8.0
                    + (w**2 * x * y) / 8.0
                    - (w**2 * x * z) / 8.0
                    + (w**2 * y * z) / 8.0
                    + (x * y * z) / 8.0
                    - (w**2 * x * y * z) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    x / 8.0
                    + y / 8.0
                    - z / 8.0
                    + (x * y) / 8.0
                    - (x * z) / 8.0
                    - (y * z) / 8.0
                    - (w**2 * x) / 8.0
                    - (w**2 * y) / 8.0
                    + (w**2 * z) / 8.0
                    - w**2 / 8.0
                    - (w**2 * x * y) / 8.0
                    + (w**2 * x * z) / 8.0
                    + (w**2 * y * z) / 8.0
                    - (x * y * z) / 8.0
                    + (w**2 * x * y * z) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    z / 8.0
                    - y / 8.0
                    - x / 8.0
                    + (x * y) / 8.0
                    - (x * z) / 8.0
                    - (y * z) / 8.0
                    + (w**2 * x) / 8.0
                    + (w**2 * y) / 8.0
                    - (w**2 * z) / 8.0
                    - w**2 / 8.0
                    - (w**2 * x * y) / 8.0
                    + (w**2 * x * z) / 8.0
                    + (w**2 * y * z) / 8.0
                    + (x * y * z) / 8.0
                    - (w**2 * x * y * z) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    x / 8.0
                    - y / 8.0
                    + z / 8.0
                    - (x * y) / 8.0
                    + (x * z) / 8.0
                    - (y * z) / 8.0
                    - (w**2 * x) / 8.0
                    + (w**2 * y) / 8.0
                    - (w**2 * z) / 8.0
                    - w**2 / 8.0
                    + (w**2 * x * y) / 8.0
                    - (w**2 * x * z) / 8.0
                    + (w**2 * y * z) / 8.0
                    - (x * y * z) / 8.0
                    + (w**2 * x * y * z) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    y / 8.0
                    - x / 8.0
                    + z / 8.0
                    - (x * y) / 8.0
                    - (x * z) / 8.0
                    + (y * z) / 8.0
                    + (w**2 * x) / 8.0
                    - (w**2 * y) / 8.0
                    - (w**2 * z) / 8.0
                    - w**2 / 8.0
                    + (w**2 * x * y) / 8.0
                    + (w**2 * x * z) / 8.0
                    - (w**2 * y * z) / 8.0
                    - (x * y * z) / 8.0
                    + (w**2 * x * y * z) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    x / 8.0
                    + y / 8.0
                    + z / 8.0
                    + (x * y) / 8.0
                    + (x * z) / 8.0
                    + (y * z) / 8.0
                    - (w**2 * x) / 8.0
                    - (w**2 * y) / 8.0
                    - (w**2 * z) / 8.0
                    - w**2 / 8.0
                    - (w**2 * x * y) / 8.0
                    - (w**2 * x * z) / 8.0
                    - (w**2 * y * z) / 8.0
                    + (x * y * z) / 8.0
                    - (w**2 * x * y * z) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    -(w**2 * x * y * z) / 16.0
                    + (w**2 * x * y) / 16.0
                    + (w**2 * x * z) / 16.0
                    - (w**2 * x) / 16.0
                    + (w**2 * y * z) / 16.0
                    - (w**2 * y) / 16.0
                    - (w**2 * z) / 16.0
                    + w**2 / 16.0
                    + (w * x**2 * y * z) / 16.0
                    - (w * x**2 * y) / 16.0
                    - (w * x**2 * z) / 16.0
                    + (w * x**2) / 16.0
                    + (w * x * y**2 * z) / 16.0
                    - (w * x * y**2) / 16.0
                    + (w * x * y * z**2) / 16.0
                    - (w * x * y * z) / 16.0
                    - (w * x * z**2) / 16.0
                    + (w * x) / 16.0
                    - (w * y**2 * z) / 16.0
                    + (w * y**2) / 16.0
                    - (w * y * z**2) / 16.0
                    + (w * y) / 16.0
                    + (w * z**2) / 16.0
                    + (w * z) / 16.0
                    - w / 8.0
                    + (x**2 * y * z) / 16.0
                    - (x**2 * y) / 16.0
                    - (x**2 * z) / 16.0
                    + x**2 / 16.0
                    + (x * y**2 * z) / 16.0
                    - (x * y**2) / 16.0
                    + (x * y * z**2) / 16.0
                    - (x * y) / 16.0
                    - (x * z**2) / 16.0
                    - (x * z) / 16.0
                    + x / 8.0
                    - (y**2 * z) / 16.0
                    + y**2 / 16.0
                    - (y * z**2) / 16.0
                    - (y * z) / 16.0
                    + y / 8.0
                    + z**2 / 16.0
                    + z / 8.0
                    - 3.0 / 16.0
                ],
                [
                    w / 8.0
                    - y / 8.0
                    - z / 8.0
                    - (w * y) / 8.0
                    - (w * z) / 8.0
                    + (y * z) / 8.0
                    - (w * x**2) / 8.0
                    + (x**2 * y) / 8.0
                    + (x**2 * z) / 8.0
                    - x**2 / 8.0
                    + (w * x**2 * y) / 8.0
                    + (w * x**2 * z) / 8.0
                    - (x**2 * y * z) / 8.0
                    + (w * y * z) / 8.0
                    - (w * x**2 * y * z) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    (w**2 * x * y * z) / 16.0
                    - (w**2 * x * y) / 16.0
                    - (w**2 * x * z) / 16.0
                    + (w**2 * x) / 16.0
                    + (w**2 * y * z) / 16.0
                    - (w**2 * y) / 16.0
                    - (w**2 * z) / 16.0
                    + w**2 / 16.0
                    + (w * x**2 * y * z) / 16.0
                    - (w * x**2 * y) / 16.0
                    - (w * x**2 * z) / 16.0
                    + (w * x**2) / 16.0
                    - (w * x * y**2 * z) / 16.0
                    + (w * x * y**2) / 16.0
                    - (w * x * y * z**2) / 16.0
                    + (w * x * y * z) / 16.0
                    + (w * x * z**2) / 16.0
                    - (w * x) / 16.0
                    - (w * y**2 * z) / 16.0
                    + (w * y**2) / 16.0
                    - (w * y * z**2) / 16.0
                    + (w * y) / 16.0
                    + (w * z**2) / 16.0
                    + (w * z) / 16.0
                    - w / 8.0
                    + (x**2 * y * z) / 16.0
                    - (x**2 * y) / 16.0
                    - (x**2 * z) / 16.0
                    + x**2 / 16.0
                    - (x * y**2 * z) / 16.0
                    + (x * y**2) / 16.0
                    - (x * y * z**2) / 16.0
                    + (x * y) / 16.0
                    + (x * z**2) / 16.0
                    + (x * z) / 16.0
                    - x / 8.0
                    - (y**2 * z) / 16.0
                    + y**2 / 16.0
                    - (y * z**2) / 16.0
                    - (y * z) / 16.0
                    + y / 8.0
                    + z**2 / 16.0
                    + z / 8.0
                    - 3.0 / 16.0
                ],
                [
                    w / 8.0
                    - x / 8.0
                    - z / 8.0
                    - (w * x) / 8.0
                    - (w * z) / 8.0
                    + (x * z) / 8.0
                    - (w * y**2) / 8.0
                    + (x * y**2) / 8.0
                    + (y**2 * z) / 8.0
                    - y**2 / 8.0
                    + (w * x * y**2) / 8.0
                    + (w * y**2 * z) / 8.0
                    - (x * y**2 * z) / 8.0
                    + (w * x * z) / 8.0
                    - (w * x * y**2 * z) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    w / 8.0
                    + x / 8.0
                    - z / 8.0
                    + (w * x) / 8.0
                    - (w * z) / 8.0
                    - (x * z) / 8.0
                    - (w * y**2) / 8.0
                    - (x * y**2) / 8.0
                    + (y**2 * z) / 8.0
                    - y**2 / 8.0
                    - (w * x * y**2) / 8.0
                    + (w * y**2 * z) / 8.0
                    + (x * y**2 * z) / 8.0
                    - (w * x * z) / 8.0
                    + (w * x * y**2 * z) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    (w**2 * x * y * z) / 16.0
                    - (w**2 * x * y) / 16.0
                    + (w**2 * x * z) / 16.0
                    - (w**2 * x) / 16.0
                    - (w**2 * y * z) / 16.0
                    + (w**2 * y) / 16.0
                    - (w**2 * z) / 16.0
                    + w**2 / 16.0
                    - (w * x**2 * y * z) / 16.0
                    + (w * x**2 * y) / 16.0
                    - (w * x**2 * z) / 16.0
                    + (w * x**2) / 16.0
                    + (w * x * y**2 * z) / 16.0
                    - (w * x * y**2) / 16.0
                    - (w * x * y * z**2) / 16.0
                    + (w * x * y * z) / 16.0
                    - (w * x * z**2) / 16.0
                    + (w * x) / 16.0
                    - (w * y**2 * z) / 16.0
                    + (w * y**2) / 16.0
                    + (w * y * z**2) / 16.0
                    - (w * y) / 16.0
                    + (w * z**2) / 16.0
                    + (w * z) / 16.0
                    - w / 8.0
                    - (x**2 * y * z) / 16.0
                    + (x**2 * y) / 16.0
                    - (x**2 * z) / 16.0
                    + x**2 / 16.0
                    + (x * y**2 * z) / 16.0
                    - (x * y**2) / 16.0
                    - (x * y * z**2) / 16.0
                    + (x * y) / 16.0
                    - (x * z**2) / 16.0
                    - (x * z) / 16.0
                    + x / 8.0
                    - (y**2 * z) / 16.0
                    + y**2 / 16.0
                    + (y * z**2) / 16.0
                    + (y * z) / 16.0
                    - y / 8.0
                    + z**2 / 16.0
                    + z / 8.0
                    - 3.0 / 16.0
                ],
                [
                    w / 8.0
                    + y / 8.0
                    - z / 8.0
                    + (w * y) / 8.0
                    - (w * z) / 8.0
                    - (y * z) / 8.0
                    - (w * x**2) / 8.0
                    - (x**2 * y) / 8.0
                    + (x**2 * z) / 8.0
                    - x**2 / 8.0
                    - (w * x**2 * y) / 8.0
                    + (w * x**2 * z) / 8.0
                    + (x**2 * y * z) / 8.0
                    - (w * y * z) / 8.0
                    + (w * x**2 * y * z) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    -(w**2 * x * y * z) / 16.0
                    + (w**2 * x * y) / 16.0
                    - (w**2 * x * z) / 16.0
                    + (w**2 * x) / 16.0
                    - (w**2 * y * z) / 16.0
                    + (w**2 * y) / 16.0
                    - (w**2 * z) / 16.0
                    + w**2 / 16.0
                    - (w * x**2 * y * z) / 16.0
                    + (w * x**2 * y) / 16.0
                    - (w * x**2 * z) / 16.0
                    + (w * x**2) / 16.0
                    - (w * x * y**2 * z) / 16.0
                    + (w * x * y**2) / 16.0
                    + (w * x * y * z**2) / 16.0
                    - (w * x * y * z) / 16.0
                    + (w * x * z**2) / 16.0
                    - (w * x) / 16.0
                    - (w * y**2 * z) / 16.0
                    + (w * y**2) / 16.0
                    + (w * y * z**2) / 16.0
                    - (w * y) / 16.0
                    + (w * z**2) / 16.0
                    + (w * z) / 16.0
                    - w / 8.0
                    - (x**2 * y * z) / 16.0
                    + (x**2 * y) / 16.0
                    - (x**2 * z) / 16.0
                    + x**2 / 16.0
                    - (x * y**2 * z) / 16.0
                    + (x * y**2) / 16.0
                    + (x * y * z**2) / 16.0
                    - (x * y) / 16.0
                    + (x * z**2) / 16.0
                    + (x * z) / 16.0
                    - x / 8.0
                    - (y**2 * z) / 16.0
                    + y**2 / 16.0
                    + (y * z**2) / 16.0
                    + (y * z) / 16.0
                    - y / 8.0
                    + z**2 / 16.0
                    + z / 8.0
                    - 3.0 / 16.0
                ],
                [
                    w / 8.0
                    - x / 8.0
                    - y / 8.0
                    - (w * x) / 8.0
                    - (w * y) / 8.0
                    + (x * y) / 8.0
                    - (w * z**2) / 8.0
                    + (x * z**2) / 8.0
                    + (y * z**2) / 8.0
                    - z**2 / 8.0
                    + (w * x * z**2) / 8.0
                    + (w * y * z**2) / 8.0
                    - (x * y * z**2) / 8.0
                    + (w * x * y) / 8.0
                    - (w * x * y * z**2) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    w / 8.0
                    + x / 8.0
                    - y / 8.0
                    + (w * x) / 8.0
                    - (w * y) / 8.0
                    - (x * y) / 8.0
                    - (w * z**2) / 8.0
                    - (x * z**2) / 8.0
                    + (y * z**2) / 8.0
                    - z**2 / 8.0
                    - (w * x * z**2) / 8.0
                    + (w * y * z**2) / 8.0
                    + (x * y * z**2) / 8.0
                    - (w * x * y) / 8.0
                    + (w * x * y * z**2) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    w / 8.0
                    - x / 8.0
                    + y / 8.0
                    - (w * x) / 8.0
                    + (w * y) / 8.0
                    - (x * y) / 8.0
                    - (w * z**2) / 8.0
                    + (x * z**2) / 8.0
                    - (y * z**2) / 8.0
                    - z**2 / 8.0
                    + (w * x * z**2) / 8.0
                    - (w * y * z**2) / 8.0
                    + (x * y * z**2) / 8.0
                    - (w * x * y) / 8.0
                    + (w * x * y * z**2) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    w / 8.0
                    + x / 8.0
                    + y / 8.0
                    + (w * x) / 8.0
                    + (w * y) / 8.0
                    + (x * y) / 8.0
                    - (w * z**2) / 8.0
                    - (x * z**2) / 8.0
                    - (y * z**2) / 8.0
                    - z**2 / 8.0
                    - (w * x * z**2) / 8.0
                    - (w * y * z**2) / 8.0
                    - (x * y * z**2) / 8.0
                    + (w * x * y) / 8.0
                    - (w * x * y * z**2) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    (w**2 * x * y * z) / 16.0
                    + (w**2 * x * y) / 16.0
                    - (w**2 * x * z) / 16.0
                    - (w**2 * x) / 16.0
                    - (w**2 * y * z) / 16.0
                    - (w**2 * y) / 16.0
                    + (w**2 * z) / 16.0
                    + w**2 / 16.0
                    - (w * x**2 * y * z) / 16.0
                    - (w * x**2 * y) / 16.0
                    + (w * x**2 * z) / 16.0
                    + (w * x**2) / 16.0
                    - (w * x * y**2 * z) / 16.0
                    - (w * x * y**2) / 16.0
                    + (w * x * y * z**2) / 16.0
                    + (w * x * y * z) / 16.0
                    - (w * x * z**2) / 16.0
                    + (w * x) / 16.0
                    + (w * y**2 * z) / 16.0
                    + (w * y**2) / 16.0
                    - (w * y * z**2) / 16.0
                    + (w * y) / 16.0
                    + (w * z**2) / 16.0
                    - (w * z) / 16.0
                    - w / 8.0
                    - (x**2 * y * z) / 16.0
                    - (x**2 * y) / 16.0
                    + (x**2 * z) / 16.0
                    + x**2 / 16.0
                    - (x * y**2 * z) / 16.0
                    - (x * y**2) / 16.0
                    + (x * y * z**2) / 16.0
                    - (x * y) / 16.0
                    - (x * z**2) / 16.0
                    + (x * z) / 16.0
                    + x / 8.0
                    + (y**2 * z) / 16.0
                    + y**2 / 16.0
                    - (y * z**2) / 16.0
                    + (y * z) / 16.0
                    + y / 8.0
                    + z**2 / 16.0
                    - z / 8.0
                    - 3.0 / 16.0
                ],
                [
                    w / 8.0
                    - y / 8.0
                    + z / 8.0
                    - (w * y) / 8.0
                    + (w * z) / 8.0
                    - (y * z) / 8.0
                    - (w * x**2) / 8.0
                    + (x**2 * y) / 8.0
                    - (x**2 * z) / 8.0
                    - x**2 / 8.0
                    + (w * x**2 * y) / 8.0
                    - (w * x**2 * z) / 8.0
                    + (x**2 * y * z) / 8.0
                    - (w * y * z) / 8.0
                    + (w * x**2 * y * z) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    -(w**2 * x * y * z) / 16.0
                    - (w**2 * x * y) / 16.0
                    + (w**2 * x * z) / 16.0
                    + (w**2 * x) / 16.0
                    - (w**2 * y * z) / 16.0
                    - (w**2 * y) / 16.0
                    + (w**2 * z) / 16.0
                    + w**2 / 16.0
                    - (w * x**2 * y * z) / 16.0
                    - (w * x**2 * y) / 16.0
                    + (w * x**2 * z) / 16.0
                    + (w * x**2) / 16.0
                    + (w * x * y**2 * z) / 16.0
                    + (w * x * y**2) / 16.0
                    - (w * x * y * z**2) / 16.0
                    - (w * x * y * z) / 16.0
                    + (w * x * z**2) / 16.0
                    - (w * x) / 16.0
                    + (w * y**2 * z) / 16.0
                    + (w * y**2) / 16.0
                    - (w * y * z**2) / 16.0
                    + (w * y) / 16.0
                    + (w * z**2) / 16.0
                    - (w * z) / 16.0
                    - w / 8.0
                    - (x**2 * y * z) / 16.0
                    - (x**2 * y) / 16.0
                    + (x**2 * z) / 16.0
                    + x**2 / 16.0
                    + (x * y**2 * z) / 16.0
                    + (x * y**2) / 16.0
                    - (x * y * z**2) / 16.0
                    + (x * y) / 16.0
                    + (x * z**2) / 16.0
                    - (x * z) / 16.0
                    - x / 8.0
                    + (y**2 * z) / 16.0
                    + y**2 / 16.0
                    - (y * z**2) / 16.0
                    + (y * z) / 16.0
                    + y / 8.0
                    + z**2 / 16.0
                    - z / 8.0
                    - 3.0 / 16.0
                ],
                [
                    w / 8.0
                    - x / 8.0
                    + z / 8.0
                    - (w * x) / 8.0
                    + (w * z) / 8.0
                    - (x * z) / 8.0
                    - (w * y**2) / 8.0
                    + (x * y**2) / 8.0
                    - (y**2 * z) / 8.0
                    - y**2 / 8.0
                    + (w * x * y**2) / 8.0
                    - (w * y**2 * z) / 8.0
                    + (x * y**2 * z) / 8.0
                    - (w * x * z) / 8.0
                    + (w * x * y**2 * z) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    w / 8.0
                    + x / 8.0
                    + z / 8.0
                    + (w * x) / 8.0
                    + (w * z) / 8.0
                    + (x * z) / 8.0
                    - (w * y**2) / 8.0
                    - (x * y**2) / 8.0
                    - (y**2 * z) / 8.0
                    - y**2 / 8.0
                    - (w * x * y**2) / 8.0
                    - (w * y**2 * z) / 8.0
                    - (x * y**2 * z) / 8.0
                    + (w * x * z) / 8.0
                    - (w * x * y**2 * z) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    -(w**2 * x * y * z) / 16.0
                    - (w**2 * x * y) / 16.0
                    - (w**2 * x * z) / 16.0
                    - (w**2 * x) / 16.0
                    + (w**2 * y * z) / 16.0
                    + (w**2 * y) / 16.0
                    + (w**2 * z) / 16.0
                    + w**2 / 16.0
                    + (w * x**2 * y * z) / 16.0
                    + (w * x**2 * y) / 16.0
                    + (w * x**2 * z) / 16.0
                    + (w * x**2) / 16.0
                    - (w * x * y**2 * z) / 16.0
                    - (w * x * y**2) / 16.0
                    - (w * x * y * z**2) / 16.0
                    - (w * x * y * z) / 16.0
                    - (w * x * z**2) / 16.0
                    + (w * x) / 16.0
                    + (w * y**2 * z) / 16.0
                    + (w * y**2) / 16.0
                    + (w * y * z**2) / 16.0
                    - (w * y) / 16.0
                    + (w * z**2) / 16.0
                    - (w * z) / 16.0
                    - w / 8.0
                    + (x**2 * y * z) / 16.0
                    + (x**2 * y) / 16.0
                    + (x**2 * z) / 16.0
                    + x**2 / 16.0
                    - (x * y**2 * z) / 16.0
                    - (x * y**2) / 16.0
                    - (x * y * z**2) / 16.0
                    + (x * y) / 16.0
                    - (x * z**2) / 16.0
                    + (x * z) / 16.0
                    + x / 8.0
                    + (y**2 * z) / 16.0
                    + y**2 / 16.0
                    + (y * z**2) / 16.0
                    - (y * z) / 16.0
                    - y / 8.0
                    + z**2 / 16.0
                    - z / 8.0
                    - 3.0 / 16.0
                ],
                [
                    w / 8.0
                    + y / 8.0
                    + z / 8.0
                    + (w * y) / 8.0
                    + (w * z) / 8.0
                    + (y * z) / 8.0
                    - (w * x**2) / 8.0
                    - (x**2 * y) / 8.0
                    - (x**2 * z) / 8.0
                    - x**2 / 8.0
                    - (w * x**2 * y) / 8.0
                    - (w * x**2 * z) / 8.0
                    - (x**2 * y * z) / 8.0
                    + (w * y * z) / 8.0
                    - (w * x**2 * y * z) / 8.0
                    + 1.0 / 8.0
                ],
                [
                    (w**2 * x * y * z) / 16.0
                    + (w**2 * x * y) / 16.0
                    + (w**2 * x * z) / 16.0
                    + (w**2 * x) / 16.0
                    + (w**2 * y * z) / 16.0
                    + (w**2 * y) / 16.0
                    + (w**2 * z) / 16.0
                    + w**2 / 16.0
                    + (w * x**2 * y * z) / 16.0
                    + (w * x**2 * y) / 16.0
                    + (w * x**2 * z) / 16.0
                    + (w * x**2) / 16.0
                    + (w * x * y**2 * z) / 16.0
                    + (w * x * y**2) / 16.0
                    + (w * x * y * z**2) / 16.0
                    + (w * x * y * z) / 16.0
                    + (w * x * z**2) / 16.0
                    - (w * x) / 16.0
                    + (w * y**2 * z) / 16.0
                    + (w * y**2) / 16.0
                    + (w * y * z**2) / 16.0
                    - (w * y) / 16.0
                    + (w * z**2) / 16.0
                    - (w * z) / 16.0
                    - w / 8.0
                    + (x**2 * y * z) / 16.0
                    + (x**2 * y) / 16.0
                    + (x**2 * z) / 16.0
                    + x**2 / 16.0
                    + (x * y**2 * z) / 16.0
                    + (x * y**2) / 16.0
                    + (x * y * z**2) / 16.0
                    - (x * y) / 16.0
                    + (x * z**2) / 16.0
                    - (x * z) / 16.0
                    - x / 8.0
                    + (y**2 * z) / 16.0
                    + y**2 / 16.0
                    + (y * z**2) / 16.0
                    - (y * z) / 16.0
                    - y / 8.0
                    + z**2 / 16.0
                    - z / 8.0
                    - 3.0 / 16.0
                ],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, interpList.shape[0]):
                for m in range(0, functionVector.shape[0]):
                  interpMatrix[
                      l
                      + k * interpList.shape[0]
                      + j * interpList.shape[0] * interpList.shape[0]
                      + i
                      * interpList.shape[0]
                      * interpList.shape[0]
                      * interpList.shape[0],
                      m,
                  ] = (
                      functionVector[m]
                      .subs(x, interpList[l])
                      .subs(y, interpList[k])
                      .subs(z, interpList[j])
                      .subs(w, interpList[i])
                  )
      else:
        raise NameError(
            "interpMatrix: Order {} is not supported!\nPolynomial order must be <3 for nodal Serendipity in 4D".format(
                order
            )
        )

    else:
      raise NameError(
          "interpMatrix: Basis {} is not supported!\nSupported basis are currently 'nodal Serendipity', 'modal Serendipity', and 'modal maximal order'".format(
              basis_type
          )
      )

  elif dim == 5:
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")
    w = Symbol("w")
    v = Symbol("v")
    if modal and basis_type == "maximal-order":
      if order == 1:
        functionVector = Matrix(
            [
                [0.1767766952966367],
                [0.3061862178478966 * x],
                [0.3061862178478966 * y],
                [0.3061862178478966 * z],
                [0.3061862178478966 * w],
                [0.3061862178478966 * v],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, interpList.shape[0]):
                for m in range(0, interpList.shape[0]):
                  for n in range(0, functionVector.shape[0]):
                    interpMatrix[
                        m
                        + l * interpList.shape[0]
                        + k * interpList.shape[0] * interpList.shape[0]
                        + j
                        * interpList.shape[0]
                        * interpList.shape[0]
                        * interpList.shape[0]
                        + i
                        * interpList.shape[0]
                        * interpList.shape[0]
                        * interpList.shape[0]
                        * interpList.shape[0],
                        n,
                    ] = (
                        functionVector[n]
                        .subs(x, interpList[m])
                        .subs(y, interpList[l])
                        .subs(z, interpList[k])
                        .subs(w, interpList[j])
                        .subs(v, interpList[i])
                    )

      elif order == 2:
        functionVector = Matrix(
            [
                [0.1767766952966367],
                [0.3061862178478966 * x],
                [0.3061862178478966 * y],
                [0.3061862178478966 * z],
                [0.3061862178478966 * w],
                [0.3061862178478966 * v],
                [0.5303300858899102 * x * y],
                [0.5303300858899102 * x * z],
                [0.5303300858899102 * y * z],
                [0.5303300858899102 * x * w],
                [0.5303300858899102 * y * w],
                [0.5303300858899102 * z * w],
                [0.5303300858899102 * x * v],
                [0.5303300858899102 * y * v],
                [0.5303300858899102 * z * v],
                [0.5303300858899102 * w * v],
                [0.592927061281571 * x**2 - 0.1976423537605237],
                [0.592927061281571 * y**2 - 0.1976423537605237],
                [0.592927061281571 * z**2 - 0.1976423537605237],
                [0.592927061281571 * w**2 - 0.1976423537605237],
                [0.592927061281571 * v**2 - 0.1976423537605237],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, interpList.shape[0]):
                for m in range(0, interpList.shape[0]):
                  for n in range(0, functionVector.shape[0]):
                    interpMatrix[
                        m
                        + l * interpList.shape[0]
                        + k * interpList.shape[0] * interpList.shape[0]
                        + j
                        * interpList.shape[0]
                        * interpList.shape[0]
                        * interpList.shape[0]
                        + i
                        * interpList.shape[0]
                        * interpList.shape[0]
                        * interpList.shape[0]
                        * interpList.shape[0],
                        n,
                    ] = (
                        functionVector[n]
                        .subs(x, interpList[m])
                        .subs(y, interpList[l])
                        .subs(z, interpList[k])
                        .subs(w, interpList[j])
                        .subs(v, interpList[i])
                    )

      elif order == 3:
        functionVector = Matrix(
            [
                [0.1767766952966367],
                [0.3061862178478966 * x],
                [0.3061862178478966 * y],
                [0.3061862178478966 * z],
                [0.3061862178478966 * w],
                [0.3061862178478966 * v],
                [0.5303300858899102 * x * y],
                [0.5303300858899102 * x * z],
                [0.5303300858899102 * y * z],
                [0.5303300858899102 * x * w],
                [0.5303300858899102 * y * w],
                [0.5303300858899102 * z * w],
                [0.5303300858899102 * x * v],
                [0.5303300858899102 * y * v],
                [0.5303300858899102 * z * v],
                [0.5303300858899102 * w * v],
                [0.592927061281571 * x**2 - 0.1976423537605237],
                [0.592927061281571 * y**2 - 0.1976423537605237],
                [0.592927061281571 * z**2 - 0.1976423537605237],
                [0.592927061281571 * w**2 - 0.1976423537605237],
                [0.592927061281571 * v**2 - 0.1976423537605237],
                [0.9185586535436896 * x * y * z],
                [0.9185586535436896 * x * y * w],
                [0.9185586535436896 * x * z * w],
                [0.9185586535436896 * y * z * w],
                [0.9185586535436896 * x * y * v],
                [0.9185586535436896 * x * z * v],
                [0.9185586535436896 * y * z * v],
                [0.9185586535436896 * x * w * v],
                [0.9185586535436896 * y * w * v],
                [0.9185586535436896 * z * w * v],
                [1.026979795322187 * x**2 * y - 0.3423265984407291 * y],
                [1.026979795322187 * x * y**2 - 0.3423265984407291 * x],
                [1.026979795322187 * x**2 * z - 0.3423265984407291 * z],
                [1.026979795322187 * y**2 * z - 0.3423265984407291 * z],
                [1.026979795322187 * x * z**2 - 0.3423265984407291 * x],
                [1.026979795322187 * y * z**2 - 0.3423265984407291 * y],
                [1.026979795322187 * x**2 * w - 0.3423265984407291 * w],
                [1.026979795322187 * y**2 * w - 0.3423265984407291 * w],
                [1.026979795322187 * z**2 * w - 0.3423265984407291 * w],
                [1.026979795322187 * x * w**2 - 0.3423265984407291 * x],
                [1.026979795322187 * y * w**2 - 0.3423265984407291 * y],
                [1.026979795322187 * z * w**2 - 0.3423265984407291 * z],
                [1.026979795322187 * x**2 * v - 0.3423265984407291 * v],
                [1.026979795322187 * y**2 * v - 0.3423265984407291 * v],
                [1.026979795322187 * z**2 * v - 0.3423265984407291 * v],
                [1.026979795322187 * w**2 * v - 0.3423265984407291 * v],
                [1.026979795322187 * x * v**2 - 0.3423265984407291 * x],
                [1.026979795322187 * y * v**2 - 0.3423265984407291 * y],
                [1.026979795322187 * z * v**2 - 0.3423265984407291 * z],
                [1.026979795322187 * w * v**2 - 0.3423265984407291 * w],
                [1.169267933366857 * x**3 - 0.701560760020114 * x],
                [1.169267933366857 * y**3 - 0.701560760020114 * y],
                [1.169267933366857 * z**3 - 0.701560760020114 * z],
                [1.169267933366857 * w**3 - 0.701560760020114 * w],
                [1.169267933366857 * v**3 - 0.701560760020114 * v],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, interpList.shape[0]):
                for m in range(0, interpList.shape[0]):
                  for n in range(0, functionVector.shape[0]):
                    interpMatrix[
                        m
                        + l * interpList.shape[0]
                        + k * interpList.shape[0] * interpList.shape[0]
                        + j
                        * interpList.shape[0]
                        * interpList.shape[0]
                        * interpList.shape[0]
                        + i
                        * interpList.shape[0]
                        * interpList.shape[0]
                        * interpList.shape[0]
                        * interpList.shape[0],
                        n,
                    ] = (
                        functionVector[n]
                        .subs(x, interpList[m])
                        .subs(y, interpList[l])
                        .subs(z, interpList[k])
                        .subs(w, interpList[j])
                        .subs(v, interpList[i])
                    )

      elif order == 4:
        functionVector = Matrix(
            [
                [0.1767766952966367],
                [0.3061862178478966 * x],
                [0.3061862178478966 * y],
                [0.3061862178478966 * z],
                [0.3061862178478966 * w],
                [0.3061862178478966 * v],
                [0.5303300858899102 * x * y],
                [0.5303300858899102 * x * z],
                [0.5303300858899102 * y * z],
                [0.5303300858899102 * x * w],
                [0.5303300858899102 * y * w],
                [0.5303300858899102 * z * w],
                [0.5303300858899102 * x * v],
                [0.5303300858899102 * y * v],
                [0.5303300858899102 * z * v],
                [0.5303300858899102 * w * v],
                [0.592927061281571 * x**2 - 0.1976423537605237],
                [0.592927061281571 * y**2 - 0.1976423537605237],
                [0.592927061281571 * z**2 - 0.1976423537605237],
                [0.592927061281571 * w**2 - 0.1976423537605237],
                [0.592927061281571 * v**2 - 0.1976423537605237],
                [0.9185586535436896 * x * y * z],
                [0.9185586535436896 * x * y * w],
                [0.9185586535436896 * x * z * w],
                [0.9185586535436896 * y * z * w],
                [0.9185586535436896 * x * y * v],
                [0.9185586535436896 * x * z * v],
                [0.9185586535436896 * y * z * v],
                [0.9185586535436896 * x * w * v],
                [0.9185586535436896 * y * w * v],
                [0.9185586535436896 * z * w * v],
                [1.026979795322187 * x**2 * y - 0.3423265984407291 * y],
                [1.026979795322187 * x * y**2 - 0.3423265984407291 * x],
                [1.026979795322187 * x**2 * z - 0.3423265984407291 * z],
                [1.026979795322187 * y**2 * z - 0.3423265984407291 * z],
                [1.026979795322187 * x * z**2 - 0.3423265984407291 * x],
                [1.026979795322187 * y * z**2 - 0.3423265984407291 * y],
                [1.026979795322187 * x**2 * w - 0.3423265984407291 * w],
                [1.026979795322187 * y**2 * w - 0.3423265984407291 * w],
                [1.026979795322187 * z**2 * w - 0.3423265984407291 * w],
                [1.026979795322187 * x * w**2 - 0.3423265984407291 * x],
                [1.026979795322187 * y * w**2 - 0.3423265984407291 * y],
                [1.026979795322187 * z * w**2 - 0.3423265984407291 * z],
                [1.026979795322187 * x**2 * v - 0.3423265984407291 * v],
                [1.026979795322187 * y**2 * v - 0.3423265984407291 * v],
                [1.026979795322187 * z**2 * v - 0.3423265984407291 * v],
                [1.026979795322187 * w**2 * v - 0.3423265984407291 * v],
                [1.026979795322187 * x * v**2 - 0.3423265984407291 * x],
                [1.026979795322187 * y * v**2 - 0.3423265984407291 * y],
                [1.026979795322187 * z * v**2 - 0.3423265984407291 * z],
                [1.026979795322187 * w * v**2 - 0.3423265984407291 * w],
                [1.169267933366857 * x**3 - 0.701560760020114 * x],
                [1.169267933366857 * y**3 - 0.701560760020114 * y],
                [1.169267933366857 * z**3 - 0.701560760020114 * z],
                [1.169267933366857 * w**3 - 0.701560760020114 * w],
                [1.169267933366857 * v**3 - 0.701560760020114 * v],
                [1.590990257669732 * x * y * z * w],
                [1.590990257669732 * x * y * z * v],
                [1.590990257669732 * x * y * w * v],
                [1.590990257669732 * x * z * w * v],
                [1.590990257669732 * y * z * w * v],
                [1.778781183844712 * x**2 * y * z - 0.5929270612815707 * y * z],
                [1.778781183844712 * x * y**2 * z - 0.5929270612815707 * x * z],
                [1.778781183844712 * x * y * z**2 - 0.5929270612815707 * x * y],
                [1.778781183844712 * x**2 * y * w - 0.5929270612815707 * y * w],
                [1.778781183844712 * x * y**2 * w - 0.5929270612815707 * x * w],
                [1.778781183844712 * x**2 * z * w - 0.5929270612815707 * z * w],
                [1.778781183844712 * y**2 * z * w - 0.5929270612815707 * z * w],
                [1.778781183844712 * x * z**2 * w - 0.5929270612815707 * x * w],
                [1.778781183844712 * y * z**2 * w - 0.5929270612815707 * y * w],
                [1.778781183844712 * x * y * w**2 - 0.5929270612815707 * x * y],
                [1.778781183844712 * x * z * w**2 - 0.5929270612815707 * x * z],
                [1.778781183844712 * y * z * w**2 - 0.5929270612815707 * y * z],
                [1.778781183844712 * x**2 * y * v - 0.5929270612815707 * y * v],
                [1.778781183844712 * x * y**2 * v - 0.5929270612815707 * x * v],
                [1.778781183844712 * x**2 * z * v - 0.5929270612815707 * z * v],
                [1.778781183844712 * y**2 * z * v - 0.5929270612815707 * z * v],
                [1.778781183844712 * x * z**2 * v - 0.5929270612815707 * x * v],
                [1.778781183844712 * y * z**2 * v - 0.5929270612815707 * y * v],
                [1.778781183844712 * x**2 * w * v - 0.5929270612815707 * w * v],
                [1.778781183844712 * y**2 * w * v - 0.5929270612815707 * w * v],
                [1.778781183844712 * z**2 * w * v - 0.5929270612815707 * w * v],
                [1.778781183844712 * x * w**2 * v - 0.5929270612815707 * x * v],
                [1.778781183844712 * y * w**2 * v - 0.5929270612815707 * y * v],
                [1.778781183844712 * z * w**2 * v - 0.5929270612815707 * z * v],
                [1.778781183844712 * x * y * v**2 - 0.5929270612815707 * x * y],
                [1.778781183844712 * x * z * v**2 - 0.5929270612815707 * x * z],
                [1.778781183844712 * y * z * v**2 - 0.5929270612815707 * y * z],
                [1.778781183844712 * x * w * v**2 - 0.5929270612815707 * x * w],
                [1.778781183844712 * y * w * v**2 - 0.5929270612815707 * y * w],
                [1.778781183844712 * z * w * v**2 - 0.5929270612815707 * z * w],
                [
                    1.988737822087165 * x**2 * y**2
                    - 0.6629126073623886 * y**2
                    - 0.6629126073623886 * x**2
                    + 0.2209708691207962
                ],
                [
                    1.988737822087165 * x**2 * z**2
                    - 0.6629126073623886 * z**2
                    - 0.6629126073623886 * x**2
                    + 0.2209708691207962
                ],
                [
                    1.988737822087165 * y**2 * z**2
                    - 0.6629126073623886 * z**2
                    - 0.6629126073623886 * y**2
                    + 0.2209708691207962
                ],
                [
                    1.988737822087165 * x**2 * w**2
                    - 0.6629126073623886 * w**2
                    - 0.6629126073623886 * x**2
                    + 0.2209708691207962
                ],
                [
                    1.988737822087165 * y**2 * w**2
                    - 0.6629126073623886 * w**2
                    - 0.6629126073623886 * y**2
                    + 0.2209708691207962
                ],
                [
                    1.988737822087165 * z**2 * w**2
                    - 0.6629126073623886 * w**2
                    - 0.6629126073623886 * z**2
                    + 0.2209708691207962
                ],
                [
                    1.988737822087165 * x**2 * v**2
                    - 0.6629126073623886 * v**2
                    - 0.6629126073623886 * x**2
                    + 0.2209708691207962
                ],
                [
                    1.988737822087165 * y**2 * v**2
                    - 0.6629126073623886 * v**2
                    - 0.6629126073623886 * y**2
                    + 0.2209708691207962
                ],
                [
                    1.988737822087165 * z**2 * v**2
                    - 0.6629126073623886 * v**2
                    - 0.6629126073623886 * z**2
                    + 0.2209708691207962
                ],
                [
                    1.988737822087165 * w**2 * v**2
                    - 0.6629126073623886 * v**2
                    - 0.6629126073623886 * w**2
                    + 0.2209708691207962
                ],
                [2.025231468252455 * x**3 * y - 1.215138880951473 * x * y],
                [2.025231468252455 * x * y**3 - 1.215138880951473 * x * y],
                [2.025231468252455 * x**3 * z - 1.215138880951473 * x * z],
                [2.025231468252455 * y**3 * z - 1.215138880951473 * y * z],
                [2.025231468252455 * x * z**3 - 1.215138880951473 * x * z],
                [2.025231468252455 * y * z**3 - 1.215138880951473 * y * z],
                [2.025231468252455 * x**3 * w - 1.215138880951473 * x * w],
                [2.025231468252455 * y**3 * w - 1.215138880951473 * y * w],
                [2.025231468252455 * z**3 * w - 1.215138880951473 * z * w],
                [2.025231468252455 * x * w**3 - 1.215138880951473 * x * w],
                [2.025231468252455 * y * w**3 - 1.215138880951473 * y * w],
                [2.025231468252455 * z * w**3 - 1.215138880951473 * z * w],
                [2.025231468252455 * x**3 * v - 1.215138880951473 * x * v],
                [2.025231468252455 * y**3 * v - 1.215138880951473 * y * v],
                [2.025231468252455 * z**3 * v - 1.215138880951473 * z * v],
                [2.025231468252455 * w**3 * v - 1.215138880951473 * w * v],
                [2.025231468252455 * x * v**3 - 1.215138880951473 * x * v],
                [2.025231468252455 * y * v**3 - 1.215138880951473 * y * v],
                [2.025231468252455 * z * v**3 - 1.215138880951473 * z * v],
                [2.025231468252455 * w * v**3 - 1.215138880951473 * w * v],
                [
                    2.320194125768356 * x**4
                    - 1.988737822087163 * x**2
                    + 0.1988737822087163
                ],
                [
                    2.320194125768356 * y**4
                    - 1.988737822087163 * y**2
                    + 0.1988737822087163
                ],
                [
                    2.320194125768356 * z**4
                    - 1.988737822087163 * z**2
                    + 0.1988737822087163
                ],
                [
                    2.320194125768356 * w**4
                    - 1.988737822087163 * w**2
                    + 0.1988737822087163
                ],
                [
                    2.320194125768356 * v**4
                    - 1.988737822087163 * v**2
                    + 0.1988737822087163
                ],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, interpList.shape[0]):
                for m in range(0, interpList.shape[0]):
                  for n in range(0, functionVector.shape[0]):
                    interpMatrix[
                        m
                        + l * interpList.shape[0]
                        + k * interpList.shape[0] * interpList.shape[0]
                        + j
                        * interpList.shape[0]
                        * interpList.shape[0]
                        * interpList.shape[0]
                        + i
                        * interpList.shape[0]
                        * interpList.shape[0]
                        * interpList.shape[0]
                        * interpList.shape[0],
                        n,
                    ] = (
                        functionVector[n]
                        .subs(x, interpList[m])
                        .subs(y, interpList[l])
                        .subs(z, interpList[k])
                        .subs(w, interpList[j])
                        .subs(v, interpList[i])
                    )
      else:
        raise NameError(
            "interpMatrix: Order {} is not supported!\nPolynomial order must be <5".format(
                order
            )
        )

    elif modal and basis_type == "serendipity":
      if order == 0:
        functionVector = Matrix([[0.1767766952966367]])
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, interpList.shape[0]):
                for m in range(0, interpList.shape[0]):
                  for n in range(0, functionVector.shape[0]):
                    interpMatrix[
                        m
                        + l * interpList.shape[0]
                        + k * interpList.shape[0] * interpList.shape[0]
                        + j
                        * interpList.shape[0]
                        * interpList.shape[0]
                        * interpList.shape[0]
                        + i
                        * interpList.shape[0]
                        * interpList.shape[0]
                        * interpList.shape[0]
                        * interpList.shape[0],
                        n,
                    ] = (
                        functionVector[n]
                        .subs(x, interpList[m])
                        .subs(y, interpList[l])
                        .subs(z, interpList[k])
                        .subs(w, interpList[j])
                        .subs(v, interpList[i])
                    )

      elif order == 1:
        functionVector = Matrix(
            [
                [0.1767766952966367],
                [0.3061862178478966 * x],
                [0.3061862178478966 * y],
                [0.3061862178478966 * z],
                [0.3061862178478966 * w],
                [0.3061862178478966 * v],
                [0.5303300858899102 * x * y],
                [0.5303300858899102 * x * z],
                [0.5303300858899102 * y * z],
                [0.5303300858899102 * x * w],
                [0.5303300858899102 * y * w],
                [0.5303300858899102 * z * w],
                [0.5303300858899102 * x * v],
                [0.5303300858899102 * y * v],
                [0.5303300858899102 * z * v],
                [0.5303300858899102 * w * v],
                [0.9185586535436896 * x * y * z],
                [0.9185586535436896 * x * y * w],
                [0.9185586535436896 * x * z * w],
                [0.9185586535436896 * y * z * w],
                [0.9185586535436896 * x * y * v],
                [0.9185586535436896 * x * z * v],
                [0.9185586535436896 * y * z * v],
                [0.9185586535436896 * x * w * v],
                [0.9185586535436896 * y * w * v],
                [0.9185586535436896 * z * w * v],
                [1.590990257669732 * x * y * z * w],
                [1.590990257669732 * x * y * z * v],
                [1.590990257669732 * x * y * w * v],
                [1.590990257669732 * x * z * w * v],
                [1.590990257669732 * y * z * w * v],
                [2.755675960631069 * x * y * z * w * v],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, interpList.shape[0]):
                for m in range(0, interpList.shape[0]):
                  for n in range(0, functionVector.shape[0]):
                    interpMatrix[
                        m
                        + l * interpList.shape[0]
                        + k * interpList.shape[0] * interpList.shape[0]
                        + j
                        * interpList.shape[0]
                        * interpList.shape[0]
                        * interpList.shape[0]
                        + i
                        * interpList.shape[0]
                        * interpList.shape[0]
                        * interpList.shape[0]
                        * interpList.shape[0],
                        n,
                    ] = (
                        functionVector[n]
                        .subs(x, interpList[m])
                        .subs(y, interpList[l])
                        .subs(z, interpList[k])
                        .subs(w, interpList[j])
                        .subs(v, interpList[i])
                    )

      elif order == 2:
        functionVector = Matrix(
            [
                [0.1767766952966367],
                [0.3061862178478966 * x],
                [0.3061862178478966 * y],
                [0.3061862178478966 * z],
                [0.3061862178478966 * w],
                [0.3061862178478966 * v],
                [0.5303300858899102 * x * y],
                [0.5303300858899102 * x * z],
                [0.5303300858899102 * y * z],
                [0.5303300858899102 * x * w],
                [0.5303300858899102 * y * w],
                [0.5303300858899102 * z * w],
                [0.5303300858899102 * x * v],
                [0.5303300858899102 * y * v],
                [0.5303300858899102 * z * v],
                [0.5303300858899102 * w * v],
                [0.592927061281571 * x**2 - 0.1976423537605237],
                [0.592927061281571 * y**2 - 0.1976423537605237],
                [0.592927061281571 * z**2 - 0.1976423537605237],
                [0.592927061281571 * w**2 - 0.1976423537605237],
                [0.592927061281571 * v**2 - 0.1976423537605237],
                [0.9185586535436896 * x * y * z],
                [0.9185586535436896 * x * y * w],
                [0.9185586535436896 * x * z * w],
                [0.9185586535436896 * y * z * w],
                [0.9185586535436896 * x * y * v],
                [0.9185586535436896 * x * z * v],
                [0.9185586535436896 * y * z * v],
                [0.9185586535436896 * x * w * v],
                [0.9185586535436896 * y * w * v],
                [0.9185586535436896 * z * w * v],
                [1.026979795322187 * x**2 * y - 0.3423265984407291 * y],
                [1.026979795322187 * x * y**2 - 0.3423265984407291 * x],
                [1.026979795322187 * x**2 * z - 0.3423265984407291 * z],
                [1.026979795322187 * y**2 * z - 0.3423265984407291 * z],
                [1.026979795322187 * x * z**2 - 0.3423265984407291 * x],
                [1.026979795322187 * y * z**2 - 0.3423265984407291 * y],
                [1.026979795322187 * x**2 * w - 0.3423265984407291 * w],
                [1.026979795322187 * y**2 * w - 0.3423265984407291 * w],
                [1.026979795322187 * z**2 * w - 0.3423265984407291 * w],
                [1.026979795322187 * x * w**2 - 0.3423265984407291 * x],
                [1.026979795322187 * y * w**2 - 0.3423265984407291 * y],
                [1.026979795322187 * z * w**2 - 0.3423265984407291 * z],
                [1.026979795322187 * x**2 * v - 0.3423265984407291 * v],
                [1.026979795322187 * y**2 * v - 0.3423265984407291 * v],
                [1.026979795322187 * z**2 * v - 0.3423265984407291 * v],
                [1.026979795322187 * w**2 * v - 0.3423265984407291 * v],
                [1.026979795322187 * x * v**2 - 0.3423265984407291 * x],
                [1.026979795322187 * y * v**2 - 0.3423265984407291 * y],
                [1.026979795322187 * z * v**2 - 0.3423265984407291 * z],
                [1.026979795322187 * w * v**2 - 0.3423265984407291 * w],
                [1.590990257669732 * x * y * z * w],
                [1.590990257669732 * x * y * z * v],
                [1.590990257669732 * x * y * w * v],
                [1.590990257669732 * x * z * w * v],
                [1.590990257669732 * y * z * w * v],
                [1.778781183844712 * x**2 * y * z - 0.5929270612815707 * y * z],
                [1.778781183844712 * x * y**2 * z - 0.5929270612815707 * x * z],
                [1.778781183844712 * x * y * z**2 - 0.5929270612815707 * x * y],
                [1.778781183844712 * x**2 * y * w - 0.5929270612815707 * y * w],
                [1.778781183844712 * x * y**2 * w - 0.5929270612815707 * x * w],
                [1.778781183844712 * x**2 * z * w - 0.5929270612815707 * z * w],
                [1.778781183844712 * y**2 * z * w - 0.5929270612815707 * z * w],
                [1.778781183844712 * x * z**2 * w - 0.5929270612815707 * x * w],
                [1.778781183844712 * y * z**2 * w - 0.5929270612815707 * y * w],
                [1.778781183844712 * x * y * w**2 - 0.5929270612815707 * x * y],
                [1.778781183844712 * x * z * w**2 - 0.5929270612815707 * x * z],
                [1.778781183844712 * y * z * w**2 - 0.5929270612815707 * y * z],
                [1.778781183844712 * x**2 * y * v - 0.5929270612815707 * y * v],
                [1.778781183844712 * x * y**2 * v - 0.5929270612815707 * x * v],
                [1.778781183844712 * x**2 * z * v - 0.5929270612815707 * z * v],
                [1.778781183844712 * y**2 * z * v - 0.5929270612815707 * z * v],
                [1.778781183844712 * x * z**2 * v - 0.5929270612815707 * x * v],
                [1.778781183844712 * y * z**2 * v - 0.5929270612815707 * y * v],
                [1.778781183844712 * x**2 * w * v - 0.5929270612815707 * w * v],
                [1.778781183844712 * y**2 * w * v - 0.5929270612815707 * w * v],
                [1.778781183844712 * z**2 * w * v - 0.5929270612815707 * w * v],
                [1.778781183844712 * x * w**2 * v - 0.5929270612815707 * x * v],
                [1.778781183844712 * y * w**2 * v - 0.5929270612815707 * y * v],
                [1.778781183844712 * z * w**2 * v - 0.5929270612815707 * z * v],
                [1.778781183844712 * x * y * v**2 - 0.5929270612815707 * x * y],
                [1.778781183844712 * x * z * v**2 - 0.5929270612815707 * x * z],
                [1.778781183844712 * y * z * v**2 - 0.5929270612815707 * y * z],
                [1.778781183844712 * x * w * v**2 - 0.5929270612815707 * x * w],
                [1.778781183844712 * y * w * v**2 - 0.5929270612815707 * y * w],
                [1.778781183844712 * z * w * v**2 - 0.5929270612815707 * z * w],
                [2.755675960631069 * x * y * z * w * v],
                [3.080939385966559 * x**2 * y * z * w - 1.026979795322186 * y * z * w],
                [3.080939385966559 * x * y**2 * z * w - 1.026979795322186 * x * z * w],
                [3.080939385966559 * x * y * z**2 * w - 1.026979795322186 * x * y * w],
                [3.080939385966559 * x * y * z * w**2 - 1.026979795322186 * x * y * z],
                [3.080939385966559 * x**2 * y * z * v - 1.026979795322186 * y * z * v],
                [3.080939385966559 * x * y**2 * z * v - 1.026979795322186 * x * z * v],
                [3.080939385966559 * x * y * z**2 * v - 1.026979795322186 * x * y * v],
                [3.080939385966559 * x**2 * y * w * v - 1.026979795322186 * y * w * v],
                [3.080939385966559 * x * y**2 * w * v - 1.026979795322186 * x * w * v],
                [3.080939385966559 * x**2 * z * w * v - 1.026979795322186 * z * w * v],
                [3.080939385966559 * y**2 * z * w * v - 1.026979795322186 * z * w * v],
                [3.080939385966559 * x * z**2 * w * v - 1.026979795322186 * x * w * v],
                [3.080939385966559 * y * z**2 * w * v - 1.026979795322186 * y * w * v],
                [3.080939385966559 * x * y * w**2 * v - 1.026979795322186 * x * y * v],
                [3.080939385966559 * x * z * w**2 * v - 1.026979795322186 * x * z * v],
                [3.080939385966559 * y * z * w**2 * v - 1.026979795322186 * y * z * v],
                [3.080939385966559 * x * y * z * v**2 - 1.026979795322186 * x * y * z],
                [3.080939385966559 * x * y * w * v**2 - 1.026979795322186 * x * y * w],
                [3.080939385966559 * x * z * w * v**2 - 1.026979795322186 * x * z * w],
                [3.080939385966559 * y * z * w * v**2 - 1.026979795322186 * y * z * w],
                [
                    5.336343551534144 * x**2 * y * z * w * v
                    - 1.778781183844715 * y * z * w * v
                ],
                [
                    5.336343551534144 * x * y**2 * z * w * v
                    - 1.778781183844715 * x * z * w * v
                ],
                [
                    5.336343551534144 * x * y * z**2 * w * v
                    - 1.778781183844715 * x * y * w * v
                ],
                [
                    5.336343551534144 * x * y * z * w**2 * v
                    - 1.778781183844715 * x * y * z * v
                ],
                [
                    5.336343551534144 * x * y * z * w * v**2
                    - 1.778781183844715 * x * y * z * w
                ],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, interpList.shape[0]):
                for m in range(0, interpList.shape[0]):
                  for n in range(0, functionVector.shape[0]):
                    interpMatrix[
                        m
                        + l * interpList.shape[0]
                        + k * interpList.shape[0] * interpList.shape[0]
                        + j
                        * interpList.shape[0]
                        * interpList.shape[0]
                        * interpList.shape[0]
                        + i
                        * interpList.shape[0]
                        * interpList.shape[0]
                        * interpList.shape[0]
                        * interpList.shape[0],
                        n,
                    ] = (
                        functionVector[n]
                        .subs(x, interpList[m])
                        .subs(y, interpList[l])
                        .subs(z, interpList[k])
                        .subs(w, interpList[j])
                        .subs(v, interpList[i])
                    )

      elif order == 3:
        functionVector = Matrix(
            [
                [0.1767766952966367],
                [0.3061862178478966 * x],
                [0.3061862178478966 * y],
                [0.3061862178478966 * z],
                [0.3061862178478966 * w],
                [0.3061862178478966 * v],
                [0.5303300858899102 * x * y],
                [0.5303300858899102 * x * z],
                [0.5303300858899102 * y * z],
                [0.5303300858899102 * x * w],
                [0.5303300858899102 * y * w],
                [0.5303300858899102 * z * w],
                [0.5303300858899102 * x * v],
                [0.5303300858899102 * y * v],
                [0.5303300858899102 * z * v],
                [0.5303300858899102 * w * v],
                [0.592927061281571 * x**2 - 0.1976423537605237],
                [0.592927061281571 * y**2 - 0.1976423537605237],
                [0.592927061281571 * z**2 - 0.1976423537605237],
                [0.592927061281571 * w**2 - 0.1976423537605237],
                [0.592927061281571 * v**2 - 0.1976423537605237],
                [0.9185586535436896 * x * y * z],
                [0.9185586535436896 * x * y * w],
                [0.9185586535436896 * x * z * w],
                [0.9185586535436896 * y * z * w],
                [0.9185586535436896 * x * y * v],
                [0.9185586535436896 * x * z * v],
                [0.9185586535436896 * y * z * v],
                [0.9185586535436896 * x * w * v],
                [0.9185586535436896 * y * w * v],
                [0.9185586535436896 * z * w * v],
                [1.026979795322187 * x**2 * y - 0.3423265984407291 * y],
                [1.026979795322187 * x * y**2 - 0.3423265984407291 * x],
                [1.026979795322187 * x**2 * z - 0.3423265984407291 * z],
                [1.026979795322187 * y**2 * z - 0.3423265984407291 * z],
                [1.026979795322187 * x * z**2 - 0.3423265984407291 * x],
                [1.026979795322187 * y * z**2 - 0.3423265984407291 * y],
                [1.026979795322187 * x**2 * w - 0.3423265984407291 * w],
                [1.026979795322187 * y**2 * w - 0.3423265984407291 * w],
                [1.026979795322187 * z**2 * w - 0.3423265984407291 * w],
                [1.026979795322187 * x * w**2 - 0.3423265984407291 * x],
                [1.026979795322187 * y * w**2 - 0.3423265984407291 * y],
                [1.026979795322187 * z * w**2 - 0.3423265984407291 * z],
                [1.026979795322187 * x**2 * v - 0.3423265984407291 * v],
                [1.026979795322187 * y**2 * v - 0.3423265984407291 * v],
                [1.026979795322187 * z**2 * v - 0.3423265984407291 * v],
                [1.026979795322187 * w**2 * v - 0.3423265984407291 * v],
                [1.026979795322187 * x * v**2 - 0.3423265984407291 * x],
                [1.026979795322187 * y * v**2 - 0.3423265984407291 * y],
                [1.026979795322187 * z * v**2 - 0.3423265984407291 * z],
                [1.026979795322187 * w * v**2 - 0.3423265984407291 * w],
                [1.169267933366857 * x**3 - 0.701560760020114 * x],
                [1.169267933366857 * y**3 - 0.701560760020114 * y],
                [1.169267933366857 * z**3 - 0.701560760020114 * z],
                [1.169267933366857 * w**3 - 0.701560760020114 * w],
                [1.169267933366857 * v**3 - 0.701560760020114 * v],
                [1.590990257669732 * x * y * z * w],
                [1.590990257669732 * x * y * z * v],
                [1.590990257669732 * x * y * w * v],
                [1.590990257669732 * x * z * w * v],
                [1.590990257669732 * y * z * w * v],
                [1.778781183844712 * x**2 * y * z - 0.5929270612815707 * y * z],
                [1.778781183844712 * x * y**2 * z - 0.5929270612815707 * x * z],
                [1.778781183844712 * x * y * z**2 - 0.5929270612815707 * x * y],
                [1.778781183844712 * x**2 * y * w - 0.5929270612815707 * y * w],
                [1.778781183844712 * x * y**2 * w - 0.5929270612815707 * x * w],
                [1.778781183844712 * x**2 * z * w - 0.5929270612815707 * z * w],
                [1.778781183844712 * y**2 * z * w - 0.5929270612815707 * z * w],
                [1.778781183844712 * x * z**2 * w - 0.5929270612815707 * x * w],
                [1.778781183844712 * y * z**2 * w - 0.5929270612815707 * y * w],
                [1.778781183844712 * x * y * w**2 - 0.5929270612815707 * x * y],
                [1.778781183844712 * x * z * w**2 - 0.5929270612815707 * x * z],
                [1.778781183844712 * y * z * w**2 - 0.5929270612815707 * y * z],
                [1.778781183844712 * x**2 * y * v - 0.5929270612815707 * y * v],
                [1.778781183844712 * x * y**2 * v - 0.5929270612815707 * x * v],
                [1.778781183844712 * x**2 * z * v - 0.5929270612815707 * z * v],
                [1.778781183844712 * y**2 * z * v - 0.5929270612815707 * z * v],
                [1.778781183844712 * x * z**2 * v - 0.5929270612815707 * x * v],
                [1.778781183844712 * y * z**2 * v - 0.5929270612815707 * y * v],
                [1.778781183844712 * x**2 * w * v - 0.5929270612815707 * w * v],
                [1.778781183844712 * y**2 * w * v - 0.5929270612815707 * w * v],
                [1.778781183844712 * z**2 * w * v - 0.5929270612815707 * w * v],
                [1.778781183844712 * x * w**2 * v - 0.5929270612815707 * x * v],
                [1.778781183844712 * y * w**2 * v - 0.5929270612815707 * y * v],
                [1.778781183844712 * z * w**2 * v - 0.5929270612815707 * z * v],
                [1.778781183844712 * x * y * v**2 - 0.5929270612815707 * x * y],
                [1.778781183844712 * x * z * v**2 - 0.5929270612815707 * x * z],
                [1.778781183844712 * y * z * v**2 - 0.5929270612815707 * y * z],
                [1.778781183844712 * x * w * v**2 - 0.5929270612815707 * x * w],
                [1.778781183844712 * y * w * v**2 - 0.5929270612815707 * y * w],
                [1.778781183844712 * z * w * v**2 - 0.5929270612815707 * z * w],
                [2.025231468252455 * x**3 * y - 1.215138880951473 * x * y],
                [2.025231468252455 * x * y**3 - 1.215138880951473 * x * y],
                [2.025231468252455 * x**3 * z - 1.215138880951473 * x * z],
                [2.025231468252455 * y**3 * z - 1.215138880951473 * y * z],
                [2.025231468252455 * x * z**3 - 1.215138880951473 * x * z],
                [2.025231468252455 * y * z**3 - 1.215138880951473 * y * z],
                [2.025231468252455 * x**3 * w - 1.215138880951473 * x * w],
                [2.025231468252455 * y**3 * w - 1.215138880951473 * y * w],
                [2.025231468252455 * z**3 * w - 1.215138880951473 * z * w],
                [2.025231468252455 * x * w**3 - 1.215138880951473 * x * w],
                [2.025231468252455 * y * w**3 - 1.215138880951473 * y * w],
                [2.025231468252455 * z * w**3 - 1.215138880951473 * z * w],
                [2.025231468252455 * x**3 * v - 1.215138880951473 * x * v],
                [2.025231468252455 * y**3 * v - 1.215138880951473 * y * v],
                [2.025231468252455 * z**3 * v - 1.215138880951473 * z * v],
                [2.025231468252455 * w**3 * v - 1.215138880951473 * w * v],
                [2.025231468252455 * x * v**3 - 1.215138880951473 * x * v],
                [2.025231468252455 * y * v**3 - 1.215138880951473 * y * v],
                [2.025231468252455 * z * v**3 - 1.215138880951473 * z * v],
                [2.025231468252455 * w * v**3 - 1.215138880951473 * w * v],
                [2.755675960631069 * x * y * z * w * v],
                [3.080939385966559 * x**2 * y * z * w - 1.026979795322186 * y * z * w],
                [3.080939385966559 * x * y**2 * z * w - 1.026979795322186 * x * z * w],
                [3.080939385966559 * x * y * z**2 * w - 1.026979795322186 * x * y * w],
                [3.080939385966559 * x * y * z * w**2 - 1.026979795322186 * x * y * z],
                [3.080939385966559 * x**2 * y * z * v - 1.026979795322186 * y * z * v],
                [3.080939385966559 * x * y**2 * z * v - 1.026979795322186 * x * z * v],
                [3.080939385966559 * x * y * z**2 * v - 1.026979795322186 * x * y * v],
                [3.080939385966559 * x**2 * y * w * v - 1.026979795322186 * y * w * v],
                [3.080939385966559 * x * y**2 * w * v - 1.026979795322186 * x * w * v],
                [3.080939385966559 * x**2 * z * w * v - 1.026979795322186 * z * w * v],
                [3.080939385966559 * y**2 * z * w * v - 1.026979795322186 * z * w * v],
                [3.080939385966559 * x * z**2 * w * v - 1.026979795322186 * x * w * v],
                [3.080939385966559 * y * z**2 * w * v - 1.026979795322186 * y * w * v],
                [3.080939385966559 * x * y * w**2 * v - 1.026979795322186 * x * y * v],
                [3.080939385966559 * x * z * w**2 * v - 1.026979795322186 * x * z * v],
                [3.080939385966559 * y * z * w**2 * v - 1.026979795322186 * y * z * v],
                [3.080939385966559 * x * y * z * v**2 - 1.026979795322186 * x * y * z],
                [3.080939385966559 * x * y * w * v**2 - 1.026979795322186 * x * y * w],
                [3.080939385966559 * x * z * w * v**2 - 1.026979795322186 * x * z * w],
                [3.080939385966559 * y * z * w * v**2 - 1.026979795322186 * y * z * w],
                [3.507803800100568 * x**3 * y * z - 2.104682280060341 * x * y * z],
                [3.507803800100568 * x * y**3 * z - 2.104682280060341 * x * y * z],
                [3.507803800100568 * x * y * z**3 - 2.104682280060341 * x * y * z],
                [3.507803800100568 * x**3 * y * w - 2.104682280060341 * x * y * w],
                [3.507803800100568 * x * y**3 * w - 2.104682280060341 * x * y * w],
                [3.507803800100568 * x**3 * z * w - 2.104682280060341 * x * z * w],
                [3.507803800100568 * y**3 * z * w - 2.104682280060341 * y * z * w],
                [3.507803800100568 * x * z**3 * w - 2.104682280060341 * x * z * w],
                [3.507803800100568 * y * z**3 * w - 2.104682280060341 * y * z * w],
                [3.507803800100568 * x * y * w**3 - 2.104682280060341 * x * y * w],
                [3.507803800100568 * x * z * w**3 - 2.104682280060341 * x * z * w],
                [3.507803800100568 * y * z * w**3 - 2.104682280060341 * y * z * w],
                [3.507803800100568 * x**3 * y * v - 2.104682280060341 * x * y * v],
                [3.507803800100568 * x * y**3 * v - 2.104682280060341 * x * y * v],
                [3.507803800100568 * x**3 * z * v - 2.104682280060341 * x * z * v],
                [3.507803800100568 * y**3 * z * v - 2.104682280060341 * y * z * v],
                [3.507803800100568 * x * z**3 * v - 2.104682280060341 * x * z * v],
                [3.507803800100568 * y * z**3 * v - 2.104682280060341 * y * z * v],
                [3.507803800100568 * x**3 * w * v - 2.104682280060341 * x * w * v],
                [3.507803800100568 * y**3 * w * v - 2.104682280060341 * y * w * v],
                [3.507803800100568 * z**3 * w * v - 2.104682280060341 * z * w * v],
                [3.507803800100568 * x * w**3 * v - 2.104682280060341 * x * w * v],
                [3.507803800100568 * y * w**3 * v - 2.104682280060341 * y * w * v],
                [3.507803800100568 * z * w**3 * v - 2.104682280060341 * z * w * v],
                [3.507803800100568 * x * y * v**3 - 2.104682280060341 * x * y * v],
                [3.507803800100568 * x * z * v**3 - 2.104682280060341 * x * z * v],
                [3.507803800100568 * y * z * v**3 - 2.104682280060341 * y * z * v],
                [3.507803800100568 * x * w * v**3 - 2.104682280060341 * x * w * v],
                [3.507803800100568 * y * w * v**3 - 2.104682280060341 * y * w * v],
                [3.507803800100568 * z * w * v**3 - 2.104682280060341 * z * w * v],
                [
                    5.336343551534144 * x**2 * y * z * w * v
                    - 1.778781183844715 * y * z * w * v
                ],
                [
                    5.336343551534144 * x * y**2 * z * w * v
                    - 1.778781183844715 * x * z * w * v
                ],
                [
                    5.336343551534144 * x * y * z**2 * w * v
                    - 1.778781183844715 * x * y * w * v
                ],
                [
                    5.336343551534144 * x * y * z * w**2 * v
                    - 1.778781183844715 * x * y * z * v
                ],
                [
                    5.336343551534144 * x * y * z * w * v**2
                    - 1.778781183844715 * x * y * z * w
                ],
                [
                    6.075694404757367 * x**3 * y * z * w
                    - 3.64541664285442 * x * y * z * w
                ],
                [
                    6.075694404757367 * x * y**3 * z * w
                    - 3.64541664285442 * x * y * z * w
                ],
                [
                    6.075694404757367 * x * y * z**3 * w
                    - 3.64541664285442 * x * y * z * w
                ],
                [
                    6.075694404757367 * x * y * z * w**3
                    - 3.64541664285442 * x * y * z * w
                ],
                [
                    6.075694404757367 * x**3 * y * z * v
                    - 3.64541664285442 * x * y * z * v
                ],
                [
                    6.075694404757367 * x * y**3 * z * v
                    - 3.64541664285442 * x * y * z * v
                ],
                [
                    6.075694404757367 * x * y * z**3 * v
                    - 3.64541664285442 * x * y * z * v
                ],
                [
                    6.075694404757367 * x**3 * y * w * v
                    - 3.64541664285442 * x * y * w * v
                ],
                [
                    6.075694404757367 * x * y**3 * w * v
                    - 3.64541664285442 * x * y * w * v
                ],
                [
                    6.075694404757367 * x**3 * z * w * v
                    - 3.64541664285442 * x * z * w * v
                ],
                [
                    6.075694404757367 * y**3 * z * w * v
                    - 3.64541664285442 * y * z * w * v
                ],
                [
                    6.075694404757367 * x * z**3 * w * v
                    - 3.64541664285442 * x * z * w * v
                ],
                [
                    6.075694404757367 * y * z**3 * w * v
                    - 3.64541664285442 * y * z * w * v
                ],
                [
                    6.075694404757367 * x * y * w**3 * v
                    - 3.64541664285442 * x * y * w * v
                ],
                [
                    6.075694404757367 * x * z * w**3 * v
                    - 3.64541664285442 * x * z * w * v
                ],
                [
                    6.075694404757367 * y * z * w**3 * v
                    - 3.64541664285442 * y * z * w * v
                ],
                [
                    6.075694404757367 * x * y * z * v**3
                    - 3.64541664285442 * x * y * z * v
                ],
                [
                    6.075694404757367 * x * y * w * v**3
                    - 3.64541664285442 * x * y * w * v
                ],
                [
                    6.075694404757367 * x * z * w * v**3
                    - 3.64541664285442 * x * z * w * v
                ],
                [
                    6.075694404757367 * y * z * w * v**3
                    - 3.64541664285442 * y * z * w * v
                ],
                [
                    10.52341140030171 * x**3 * y * z * w * v
                    - 6.314046840181025 * x * y * z * w * v
                ],
                [
                    10.52341140030171 * x * y**3 * z * w * v
                    - 6.314046840181025 * x * y * z * w * v
                ],
                [
                    10.52341140030171 * x * y * z**3 * w * v
                    - 6.314046840181025 * x * y * z * w * v
                ],
                [
                    10.52341140030171 * x * y * z * w**3 * v
                    - 6.314046840181025 * x * y * z * w * v
                ],
                [
                    10.52341140030171 * x * y * z * w * v**3
                    - 6.314046840181025 * x * y * z * w * v
                ],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, interpList.shape[0]):
                for m in range(0, interpList.shape[0]):
                  for n in range(0, functionVector.shape[0]):
                    interpMatrix[
                        m
                        + l * interpList.shape[0]
                        + k * interpList.shape[0] * interpList.shape[0]
                        + j
                        * interpList.shape[0]
                        * interpList.shape[0]
                        * interpList.shape[0]
                        + i
                        * interpList.shape[0]
                        * interpList.shape[0]
                        * interpList.shape[0]
                        * interpList.shape[0],
                        n,
                    ] = (
                        functionVector[n]
                        .subs(x, interpList[m])
                        .subs(y, interpList[l])
                        .subs(z, interpList[k])
                        .subs(w, interpList[j])
                        .subs(v, interpList[i])
                    )

      elif order == 4:
        functionVector = Matrix(
            [
                [0.1767766952966367],
                [0.3061862178478966 * x],
                [0.3061862178478966 * y],
                [0.3061862178478966 * z],
                [0.3061862178478966 * w],
                [0.3061862178478966 * v],
                [0.5303300858899102 * x * y],
                [0.5303300858899102 * x * z],
                [0.5303300858899102 * y * z],
                [0.5303300858899102 * x * w],
                [0.5303300858899102 * y * w],
                [0.5303300858899102 * z * w],
                [0.5303300858899102 * x * v],
                [0.5303300858899102 * y * v],
                [0.5303300858899102 * z * v],
                [0.5303300858899102 * w * v],
                [0.592927061281571 * x**2 - 0.1976423537605237],
                [0.592927061281571 * y**2 - 0.1976423537605237],
                [0.592927061281571 * z**2 - 0.1976423537605237],
                [0.592927061281571 * w**2 - 0.1976423537605237],
                [0.592927061281571 * v**2 - 0.1976423537605237],
                [0.9185586535436896 * x * y * z],
                [0.9185586535436896 * x * y * w],
                [0.9185586535436896 * x * z * w],
                [0.9185586535436896 * y * z * w],
                [0.9185586535436896 * x * y * v],
                [0.9185586535436896 * x * z * v],
                [0.9185586535436896 * y * z * v],
                [0.9185586535436896 * x * w * v],
                [0.9185586535436896 * y * w * v],
                [0.9185586535436896 * z * w * v],
                [1.026979795322187 * x**2 * y - 0.3423265984407291 * y],
                [1.026979795322187 * x * y**2 - 0.3423265984407291 * x],
                [1.026979795322187 * x**2 * z - 0.3423265984407291 * z],
                [1.026979795322187 * y**2 * z - 0.3423265984407291 * z],
                [1.026979795322187 * x * z**2 - 0.3423265984407291 * x],
                [1.026979795322187 * y * z**2 - 0.3423265984407291 * y],
                [1.026979795322187 * x**2 * w - 0.3423265984407291 * w],
                [1.026979795322187 * y**2 * w - 0.3423265984407291 * w],
                [1.026979795322187 * z**2 * w - 0.3423265984407291 * w],
                [1.026979795322187 * x * w**2 - 0.3423265984407291 * x],
                [1.026979795322187 * y * w**2 - 0.3423265984407291 * y],
                [1.026979795322187 * z * w**2 - 0.3423265984407291 * z],
                [1.026979795322187 * x**2 * v - 0.3423265984407291 * v],
                [1.026979795322187 * y**2 * v - 0.3423265984407291 * v],
                [1.026979795322187 * z**2 * v - 0.3423265984407291 * v],
                [1.026979795322187 * w**2 * v - 0.3423265984407291 * v],
                [1.026979795322187 * x * v**2 - 0.3423265984407291 * x],
                [1.026979795322187 * y * v**2 - 0.3423265984407291 * y],
                [1.026979795322187 * z * v**2 - 0.3423265984407291 * z],
                [1.026979795322187 * w * v**2 - 0.3423265984407291 * w],
                [1.169267933366857 * x**3 - 0.701560760020114 * x],
                [1.169267933366857 * y**3 - 0.701560760020114 * y],
                [1.169267933366857 * z**3 - 0.701560760020114 * z],
                [1.169267933366857 * w**3 - 0.701560760020114 * w],
                [1.169267933366857 * v**3 - 0.701560760020114 * v],
                [1.590990257669732 * x * y * z * w],
                [1.590990257669732 * x * y * z * v],
                [1.590990257669732 * x * y * w * v],
                [1.590990257669732 * x * z * w * v],
                [1.590990257669732 * y * z * w * v],
                [1.778781183844712 * x**2 * y * z - 0.5929270612815707 * y * z],
                [1.778781183844712 * x * y**2 * z - 0.5929270612815707 * x * z],
                [1.778781183844712 * x * y * z**2 - 0.5929270612815707 * x * y],
                [1.778781183844712 * x**2 * y * w - 0.5929270612815707 * y * w],
                [1.778781183844712 * x * y**2 * w - 0.5929270612815707 * x * w],
                [1.778781183844712 * x**2 * z * w - 0.5929270612815707 * z * w],
                [1.778781183844712 * y**2 * z * w - 0.5929270612815707 * z * w],
                [1.778781183844712 * x * z**2 * w - 0.5929270612815707 * x * w],
                [1.778781183844712 * y * z**2 * w - 0.5929270612815707 * y * w],
                [1.778781183844712 * x * y * w**2 - 0.5929270612815707 * x * y],
                [1.778781183844712 * x * z * w**2 - 0.5929270612815707 * x * z],
                [1.778781183844712 * y * z * w**2 - 0.5929270612815707 * y * z],
                [1.778781183844712 * x**2 * y * v - 0.5929270612815707 * y * v],
                [1.778781183844712 * x * y**2 * v - 0.5929270612815707 * x * v],
                [1.778781183844712 * x**2 * z * v - 0.5929270612815707 * z * v],
                [1.778781183844712 * y**2 * z * v - 0.5929270612815707 * z * v],
                [1.778781183844712 * x * z**2 * v - 0.5929270612815707 * x * v],
                [1.778781183844712 * y * z**2 * v - 0.5929270612815707 * y * v],
                [1.778781183844712 * x**2 * w * v - 0.5929270612815707 * w * v],
                [1.778781183844712 * y**2 * w * v - 0.5929270612815707 * w * v],
                [1.778781183844712 * z**2 * w * v - 0.5929270612815707 * w * v],
                [1.778781183844712 * x * w**2 * v - 0.5929270612815707 * x * v],
                [1.778781183844712 * y * w**2 * v - 0.5929270612815707 * y * v],
                [1.778781183844712 * z * w**2 * v - 0.5929270612815707 * z * v],
                [1.778781183844712 * x * y * v**2 - 0.5929270612815707 * x * y],
                [1.778781183844712 * x * z * v**2 - 0.5929270612815707 * x * z],
                [1.778781183844712 * y * z * v**2 - 0.5929270612815707 * y * z],
                [1.778781183844712 * x * w * v**2 - 0.5929270612815707 * x * w],
                [1.778781183844712 * y * w * v**2 - 0.5929270612815707 * y * w],
                [1.778781183844712 * z * w * v**2 - 0.5929270612815707 * z * w],
                [
                    1.988737822087165 * x**2 * y**2
                    - 0.6629126073623886 * y**2
                    - 0.6629126073623886 * x**2
                    + 0.2209708691207962
                ],
                [
                    1.988737822087165 * x**2 * z**2
                    - 0.6629126073623886 * z**2
                    - 0.6629126073623886 * x**2
                    + 0.2209708691207962
                ],
                [
                    1.988737822087165 * y**2 * z**2
                    - 0.6629126073623886 * z**2
                    - 0.6629126073623886 * y**2
                    + 0.2209708691207962
                ],
                [
                    1.988737822087165 * x**2 * w**2
                    - 0.6629126073623886 * w**2
                    - 0.6629126073623886 * x**2
                    + 0.2209708691207962
                ],
                [
                    1.988737822087165 * y**2 * w**2
                    - 0.6629126073623886 * w**2
                    - 0.6629126073623886 * y**2
                    + 0.2209708691207962
                ],
                [
                    1.988737822087165 * z**2 * w**2
                    - 0.6629126073623886 * w**2
                    - 0.6629126073623886 * z**2
                    + 0.2209708691207962
                ],
                [
                    1.988737822087165 * x**2 * v**2
                    - 0.6629126073623886 * v**2
                    - 0.6629126073623886 * x**2
                    + 0.2209708691207962
                ],
                [
                    1.988737822087165 * y**2 * v**2
                    - 0.6629126073623886 * v**2
                    - 0.6629126073623886 * y**2
                    + 0.2209708691207962
                ],
                [
                    1.988737822087165 * z**2 * v**2
                    - 0.6629126073623886 * v**2
                    - 0.6629126073623886 * z**2
                    + 0.2209708691207962
                ],
                [
                    1.988737822087165 * w**2 * v**2
                    - 0.6629126073623886 * v**2
                    - 0.6629126073623886 * w**2
                    + 0.2209708691207962
                ],
                [2.025231468252455 * x**3 * y - 1.215138880951473 * x * y],
                [2.025231468252455 * x * y**3 - 1.215138880951473 * x * y],
                [2.025231468252455 * x**3 * z - 1.215138880951473 * x * z],
                [2.025231468252455 * y**3 * z - 1.215138880951473 * y * z],
                [2.025231468252455 * x * z**3 - 1.215138880951473 * x * z],
                [2.025231468252455 * y * z**3 - 1.215138880951473 * y * z],
                [2.025231468252455 * x**3 * w - 1.215138880951473 * x * w],
                [2.025231468252455 * y**3 * w - 1.215138880951473 * y * w],
                [2.025231468252455 * z**3 * w - 1.215138880951473 * z * w],
                [2.025231468252455 * x * w**3 - 1.215138880951473 * x * w],
                [2.025231468252455 * y * w**3 - 1.215138880951473 * y * w],
                [2.025231468252455 * z * w**3 - 1.215138880951473 * z * w],
                [2.025231468252455 * x**3 * v - 1.215138880951473 * x * v],
                [2.025231468252455 * y**3 * v - 1.215138880951473 * y * v],
                [2.025231468252455 * z**3 * v - 1.215138880951473 * z * v],
                [2.025231468252455 * w**3 * v - 1.215138880951473 * w * v],
                [2.025231468252455 * x * v**3 - 1.215138880951473 * x * v],
                [2.025231468252455 * y * v**3 - 1.215138880951473 * y * v],
                [2.025231468252455 * z * v**3 - 1.215138880951473 * z * v],
                [2.025231468252455 * w * v**3 - 1.215138880951473 * w * v],
                [
                    2.320194125768356 * x**4
                    - 1.988737822087163 * x**2
                    + 0.1988737822087163
                ],
                [
                    2.320194125768356 * y**4
                    - 1.988737822087163 * y**2
                    + 0.1988737822087163
                ],
                [
                    2.320194125768356 * z**4
                    - 1.988737822087163 * z**2
                    + 0.1988737822087163
                ],
                [
                    2.320194125768356 * w**4
                    - 1.988737822087163 * w**2
                    + 0.1988737822087163
                ],
                [
                    2.320194125768356 * v**4
                    - 1.988737822087163 * v**2
                    + 0.1988737822087163
                ],
                [2.755675960631069 * x * y * z * w * v],
                [3.080939385966559 * x**2 * y * z * w - 1.026979795322186 * y * z * w],
                [3.080939385966559 * x * y**2 * z * w - 1.026979795322186 * x * z * w],
                [3.080939385966559 * x * y * z**2 * w - 1.026979795322186 * x * y * w],
                [3.080939385966559 * x * y * z * w**2 - 1.026979795322186 * x * y * z],
                [3.080939385966559 * x**2 * y * z * v - 1.026979795322186 * y * z * v],
                [3.080939385966559 * x * y**2 * z * v - 1.026979795322186 * x * z * v],
                [3.080939385966559 * x * y * z**2 * v - 1.026979795322186 * x * y * v],
                [3.080939385966559 * x**2 * y * w * v - 1.026979795322186 * y * w * v],
                [3.080939385966559 * x * y**2 * w * v - 1.026979795322186 * x * w * v],
                [3.080939385966559 * x**2 * z * w * v - 1.026979795322186 * z * w * v],
                [3.080939385966559 * y**2 * z * w * v - 1.026979795322186 * z * w * v],
                [3.080939385966559 * x * z**2 * w * v - 1.026979795322186 * x * w * v],
                [3.080939385966559 * y * z**2 * w * v - 1.026979795322186 * y * w * v],
                [3.080939385966559 * x * y * w**2 * v - 1.026979795322186 * x * y * v],
                [3.080939385966559 * x * z * w**2 * v - 1.026979795322186 * x * z * v],
                [3.080939385966559 * y * z * w**2 * v - 1.026979795322186 * y * z * v],
                [3.080939385966559 * x * y * z * v**2 - 1.026979795322186 * x * y * z],
                [3.080939385966559 * x * y * w * v**2 - 1.026979795322186 * x * y * w],
                [3.080939385966559 * x * z * w * v**2 - 1.026979795322186 * x * z * w],
                [3.080939385966559 * y * z * w * v**2 - 1.026979795322186 * y * z * w],
                [
                    3.444594950788842 * x**2 * y**2 * z
                    - 1.148198316929614 * y**2 * z
                    - 1.148198316929614 * x**2 * z
                    + 0.3827327723098713 * z
                ],
                [
                    3.444594950788842 * x**2 * y * z**2
                    - 1.148198316929614 * y * z**2
                    - 1.148198316929614 * x**2 * y
                    + 0.3827327723098713 * y
                ],
                [
                    3.444594950788842 * x * y**2 * z**2
                    - 1.148198316929614 * x * z**2
                    - 1.148198316929614 * x * y**2
                    + 0.3827327723098713 * x
                ],
                [
                    3.444594950788842 * x**2 * y**2 * w
                    - 1.148198316929614 * y**2 * w
                    - 1.148198316929614 * x**2 * w
                    + 0.3827327723098713 * w
                ],
                [
                    3.444594950788842 * x**2 * z**2 * w
                    - 1.148198316929614 * z**2 * w
                    - 1.148198316929614 * x**2 * w
                    + 0.3827327723098713 * w
                ],
                [
                    3.444594950788842 * y**2 * z**2 * w
                    - 1.148198316929614 * z**2 * w
                    - 1.148198316929614 * y**2 * w
                    + 0.3827327723098713 * w
                ],
                [
                    3.444594950788842 * x**2 * y * w**2
                    - 1.148198316929614 * y * w**2
                    - 1.148198316929614 * x**2 * y
                    + 0.3827327723098713 * y
                ],
                [
                    3.444594950788842 * x * y**2 * w**2
                    - 1.148198316929614 * x * w**2
                    - 1.148198316929614 * x * y**2
                    + 0.3827327723098713 * x
                ],
                [
                    3.444594950788842 * x**2 * z * w**2
                    - 1.148198316929614 * z * w**2
                    - 1.148198316929614 * x**2 * z
                    + 0.3827327723098713 * z
                ],
                [
                    3.444594950788842 * y**2 * z * w**2
                    - 1.148198316929614 * z * w**2
                    - 1.148198316929614 * y**2 * z
                    + 0.3827327723098713 * z
                ],
                [
                    3.444594950788842 * x * z**2 * w**2
                    - 1.148198316929614 * x * w**2
                    - 1.148198316929614 * x * z**2
                    + 0.3827327723098713 * x
                ],
                [
                    3.444594950788842 * y * z**2 * w**2
                    - 1.148198316929614 * y * w**2
                    - 1.148198316929614 * y * z**2
                    + 0.3827327723098713 * y
                ],
                [
                    3.444594950788842 * x**2 * y**2 * v
                    - 1.148198316929614 * y**2 * v
                    - 1.148198316929614 * x**2 * v
                    + 0.3827327723098713 * v
                ],
                [
                    3.444594950788842 * x**2 * z**2 * v
                    - 1.148198316929614 * z**2 * v
                    - 1.148198316929614 * x**2 * v
                    + 0.3827327723098713 * v
                ],
                [
                    3.444594950788842 * y**2 * z**2 * v
                    - 1.148198316929614 * z**2 * v
                    - 1.148198316929614 * y**2 * v
                    + 0.3827327723098713 * v
                ],
                [
                    3.444594950788842 * x**2 * w**2 * v
                    - 1.148198316929614 * w**2 * v
                    - 1.148198316929614 * x**2 * v
                    + 0.3827327723098713 * v
                ],
                [
                    3.444594950788842 * y**2 * w**2 * v
                    - 1.148198316929614 * w**2 * v
                    - 1.148198316929614 * y**2 * v
                    + 0.3827327723098713 * v
                ],
                [
                    3.444594950788842 * z**2 * w**2 * v
                    - 1.148198316929614 * w**2 * v
                    - 1.148198316929614 * z**2 * v
                    + 0.3827327723098713 * v
                ],
                [
                    3.444594950788842 * x**2 * y * v**2
                    - 1.148198316929614 * y * v**2
                    - 1.148198316929614 * x**2 * y
                    + 0.3827327723098713 * y
                ],
                [
                    3.444594950788842 * x * y**2 * v**2
                    - 1.148198316929614 * x * v**2
                    - 1.148198316929614 * x * y**2
                    + 0.3827327723098713 * x
                ],
                [
                    3.444594950788842 * x**2 * z * v**2
                    - 1.148198316929614 * z * v**2
                    - 1.148198316929614 * x**2 * z
                    + 0.3827327723098713 * z
                ],
                [
                    3.444594950788842 * y**2 * z * v**2
                    - 1.148198316929614 * z * v**2
                    - 1.148198316929614 * y**2 * z
                    + 0.3827327723098713 * z
                ],
                [
                    3.444594950788842 * x * z**2 * v**2
                    - 1.148198316929614 * x * v**2
                    - 1.148198316929614 * x * z**2
                    + 0.3827327723098713 * x
                ],
                [
                    3.444594950788842 * y * z**2 * v**2
                    - 1.148198316929614 * y * v**2
                    - 1.148198316929614 * y * z**2
                    + 0.3827327723098713 * y
                ],
                [
                    3.444594950788842 * x**2 * w * v**2
                    - 1.148198316929614 * w * v**2
                    - 1.148198316929614 * x**2 * w
                    + 0.3827327723098713 * w
                ],
                [
                    3.444594950788842 * y**2 * w * v**2
                    - 1.148198316929614 * w * v**2
                    - 1.148198316929614 * y**2 * w
                    + 0.3827327723098713 * w
                ],
                [
                    3.444594950788842 * z**2 * w * v**2
                    - 1.148198316929614 * w * v**2
                    - 1.148198316929614 * z**2 * w
                    + 0.3827327723098713 * w
                ],
                [
                    3.444594950788842 * x * w**2 * v**2
                    - 1.148198316929614 * x * v**2
                    - 1.148198316929614 * x * w**2
                    + 0.3827327723098713 * x
                ],
                [
                    3.444594950788842 * y * w**2 * v**2
                    - 1.148198316929614 * y * v**2
                    - 1.148198316929614 * y * w**2
                    + 0.3827327723098713 * y
                ],
                [
                    3.444594950788842 * z * w**2 * v**2
                    - 1.148198316929614 * z * v**2
                    - 1.148198316929614 * z * w**2
                    + 0.3827327723098713 * z
                ],
                [3.507803800100568 * x**3 * y * z - 2.104682280060341 * x * y * z],
                [3.507803800100568 * x * y**3 * z - 2.104682280060341 * x * y * z],
                [3.507803800100568 * x * y * z**3 - 2.104682280060341 * x * y * z],
                [3.507803800100568 * x**3 * y * w - 2.104682280060341 * x * y * w],
                [3.507803800100568 * x * y**3 * w - 2.104682280060341 * x * y * w],
                [3.507803800100568 * x**3 * z * w - 2.104682280060341 * x * z * w],
                [3.507803800100568 * y**3 * z * w - 2.104682280060341 * y * z * w],
                [3.507803800100568 * x * z**3 * w - 2.104682280060341 * x * z * w],
                [3.507803800100568 * y * z**3 * w - 2.104682280060341 * y * z * w],
                [3.507803800100568 * x * y * w**3 - 2.104682280060341 * x * y * w],
                [3.507803800100568 * x * z * w**3 - 2.104682280060341 * x * z * w],
                [3.507803800100568 * y * z * w**3 - 2.104682280060341 * y * z * w],
                [3.507803800100568 * x**3 * y * v - 2.104682280060341 * x * y * v],
                [3.507803800100568 * x * y**3 * v - 2.104682280060341 * x * y * v],
                [3.507803800100568 * x**3 * z * v - 2.104682280060341 * x * z * v],
                [3.507803800100568 * y**3 * z * v - 2.104682280060341 * y * z * v],
                [3.507803800100568 * x * z**3 * v - 2.104682280060341 * x * z * v],
                [3.507803800100568 * y * z**3 * v - 2.104682280060341 * y * z * v],
                [3.507803800100568 * x**3 * w * v - 2.104682280060341 * x * w * v],
                [3.507803800100568 * y**3 * w * v - 2.104682280060341 * y * w * v],
                [3.507803800100568 * z**3 * w * v - 2.104682280060341 * z * w * v],
                [3.507803800100568 * x * w**3 * v - 2.104682280060341 * x * w * v],
                [3.507803800100568 * y * w**3 * v - 2.104682280060341 * y * w * v],
                [3.507803800100568 * z * w**3 * v - 2.104682280060341 * z * w * v],
                [3.507803800100568 * x * y * v**3 - 2.104682280060341 * x * y * v],
                [3.507803800100568 * x * z * v**3 - 2.104682280060341 * x * z * v],
                [3.507803800100568 * y * z * v**3 - 2.104682280060341 * y * z * v],
                [3.507803800100568 * x * w * v**3 - 2.104682280060341 * x * w * v],
                [3.507803800100568 * y * w * v**3 - 2.104682280060341 * y * w * v],
                [3.507803800100568 * z * w * v**3 - 2.104682280060341 * z * w * v],
                [
                    4.018694109253645 * x**4 * y
                    - 3.444594950788839 * x**2 * y
                    + 0.3444594950788838 * y
                ],
                [
                    4.018694109253645 * x * y**4
                    - 3.444594950788839 * x * y**2
                    + 0.3444594950788838 * x
                ],
                [
                    4.018694109253645 * x**4 * z
                    - 3.444594950788839 * x**2 * z
                    + 0.3444594950788838 * z
                ],
                [
                    4.018694109253645 * y**4 * z
                    - 3.444594950788839 * y**2 * z
                    + 0.3444594950788838 * z
                ],
                [
                    4.018694109253645 * x * z**4
                    - 3.444594950788839 * x * z**2
                    + 0.3444594950788838 * x
                ],
                [
                    4.018694109253645 * y * z**4
                    - 3.444594950788839 * y * z**2
                    + 0.3444594950788838 * y
                ],
                [
                    4.018694109253645 * x**4 * w
                    - 3.444594950788839 * x**2 * w
                    + 0.3444594950788838 * w
                ],
                [
                    4.018694109253645 * y**4 * w
                    - 3.444594950788839 * y**2 * w
                    + 0.3444594950788838 * w
                ],
                [
                    4.018694109253645 * z**4 * w
                    - 3.444594950788839 * z**2 * w
                    + 0.3444594950788838 * w
                ],
                [
                    4.018694109253645 * x * w**4
                    - 3.444594950788839 * x * w**2
                    + 0.3444594950788838 * x
                ],
                [
                    4.018694109253645 * y * w**4
                    - 3.444594950788839 * y * w**2
                    + 0.3444594950788838 * y
                ],
                [
                    4.018694109253645 * z * w**4
                    - 3.444594950788839 * z * w**2
                    + 0.3444594950788838 * z
                ],
                [
                    4.018694109253645 * x**4 * v
                    - 3.444594950788839 * x**2 * v
                    + 0.3444594950788838 * v
                ],
                [
                    4.018694109253645 * y**4 * v
                    - 3.444594950788839 * y**2 * v
                    + 0.3444594950788838 * v
                ],
                [
                    4.018694109253645 * z**4 * v
                    - 3.444594950788839 * z**2 * v
                    + 0.3444594950788838 * v
                ],
                [
                    4.018694109253645 * w**4 * v
                    - 3.444594950788839 * w**2 * v
                    + 0.3444594950788838 * v
                ],
                [
                    4.018694109253645 * x * v**4
                    - 3.444594950788839 * x * v**2
                    + 0.3444594950788838 * x
                ],
                [
                    4.018694109253645 * y * v**4
                    - 3.444594950788839 * y * v**2
                    + 0.3444594950788838 * y
                ],
                [
                    4.018694109253645 * z * v**4
                    - 3.444594950788839 * z * v**2
                    + 0.3444594950788838 * z
                ],
                [
                    4.018694109253645 * w * v**4
                    - 3.444594950788839 * w * v**2
                    + 0.3444594950788838 * w
                ],
                [
                    5.336343551534144 * x**2 * y * z * w * v
                    - 1.778781183844715 * y * z * w * v
                ],
                [
                    5.336343551534144 * x * y**2 * z * w * v
                    - 1.778781183844715 * x * z * w * v
                ],
                [
                    5.336343551534144 * x * y * z**2 * w * v
                    - 1.778781183844715 * x * y * w * v
                ],
                [
                    5.336343551534144 * x * y * z * w**2 * v
                    - 1.778781183844715 * x * y * z * v
                ],
                [
                    5.336343551534144 * x * y * z * w * v**2
                    - 1.778781183844715 * x * y * z * w
                ],
                [
                    5.966213466261497 * x**2 * y**2 * z * w
                    - 1.988737822087165 * y**2 * z * w
                    - 1.988737822087165 * x**2 * z * w
                    + 0.6629126073623886 * z * w
                ],
                [
                    5.966213466261497 * x**2 * y * z**2 * w
                    - 1.988737822087165 * y * z**2 * w
                    - 1.988737822087165 * x**2 * y * w
                    + 0.6629126073623886 * y * w
                ],
                [
                    5.966213466261497 * x * y**2 * z**2 * w
                    - 1.988737822087165 * x * z**2 * w
                    - 1.988737822087165 * x * y**2 * w
                    + 0.6629126073623886 * x * w
                ],
                [
                    5.966213466261497 * x**2 * y * z * w**2
                    - 1.988737822087165 * y * z * w**2
                    - 1.988737822087165 * x**2 * y * z
                    + 0.6629126073623886 * y * z
                ],
                [
                    5.966213466261497 * x * y**2 * z * w**2
                    - 1.988737822087165 * x * z * w**2
                    - 1.988737822087165 * x * y**2 * z
                    + 0.6629126073623886 * x * z
                ],
                [
                    5.966213466261497 * x * y * z**2 * w**2
                    - 1.988737822087165 * x * y * w**2
                    - 1.988737822087165 * x * y * z**2
                    + 0.6629126073623886 * x * y
                ],
                [
                    5.966213466261497 * x**2 * y**2 * z * v
                    - 1.988737822087165 * y**2 * z * v
                    - 1.988737822087165 * x**2 * z * v
                    + 0.6629126073623886 * z * v
                ],
                [
                    5.966213466261497 * x**2 * y * z**2 * v
                    - 1.988737822087165 * y * z**2 * v
                    - 1.988737822087165 * x**2 * y * v
                    + 0.6629126073623886 * y * v
                ],
                [
                    5.966213466261497 * x * y**2 * z**2 * v
                    - 1.988737822087165 * x * z**2 * v
                    - 1.988737822087165 * x * y**2 * v
                    + 0.6629126073623886 * x * v
                ],
                [
                    5.966213466261497 * x**2 * y**2 * w * v
                    - 1.988737822087165 * y**2 * w * v
                    - 1.988737822087165 * x**2 * w * v
                    + 0.6629126073623886 * w * v
                ],
                [
                    5.966213466261497 * x**2 * z**2 * w * v
                    - 1.988737822087165 * z**2 * w * v
                    - 1.988737822087165 * x**2 * w * v
                    + 0.6629126073623886 * w * v
                ],
                [
                    5.966213466261497 * y**2 * z**2 * w * v
                    - 1.988737822087165 * z**2 * w * v
                    - 1.988737822087165 * y**2 * w * v
                    + 0.6629126073623886 * w * v
                ],
                [
                    5.966213466261497 * x**2 * y * w**2 * v
                    - 1.988737822087165 * y * w**2 * v
                    - 1.988737822087165 * x**2 * y * v
                    + 0.6629126073623886 * y * v
                ],
                [
                    5.966213466261497 * x * y**2 * w**2 * v
                    - 1.988737822087165 * x * w**2 * v
                    - 1.988737822087165 * x * y**2 * v
                    + 0.6629126073623886 * x * v
                ],
                [
                    5.966213466261497 * x**2 * z * w**2 * v
                    - 1.988737822087165 * z * w**2 * v
                    - 1.988737822087165 * x**2 * z * v
                    + 0.6629126073623886 * z * v
                ],
                [
                    5.966213466261497 * y**2 * z * w**2 * v
                    - 1.988737822087165 * z * w**2 * v
                    - 1.988737822087165 * y**2 * z * v
                    + 0.6629126073623886 * z * v
                ],
                [
                    5.966213466261497 * x * z**2 * w**2 * v
                    - 1.988737822087165 * x * w**2 * v
                    - 1.988737822087165 * x * z**2 * v
                    + 0.6629126073623886 * x * v
                ],
                [
                    5.966213466261497 * y * z**2 * w**2 * v
                    - 1.988737822087165 * y * w**2 * v
                    - 1.988737822087165 * y * z**2 * v
                    + 0.6629126073623886 * y * v
                ],
                [
                    5.966213466261497 * x**2 * y * z * v**2
                    - 1.988737822087165 * y * z * v**2
                    - 1.988737822087165 * x**2 * y * z
                    + 0.6629126073623886 * y * z
                ],
                [
                    5.966213466261497 * x * y**2 * z * v**2
                    - 1.988737822087165 * x * z * v**2
                    - 1.988737822087165 * x * y**2 * z
                    + 0.6629126073623886 * x * z
                ],
                [
                    5.966213466261497 * x * y * z**2 * v**2
                    - 1.988737822087165 * x * y * v**2
                    - 1.988737822087165 * x * y * z**2
                    + 0.6629126073623886 * x * y
                ],
                [
                    5.966213466261497 * x**2 * y * w * v**2
                    - 1.988737822087165 * y * w * v**2
                    - 1.988737822087165 * x**2 * y * w
                    + 0.6629126073623886 * y * w
                ],
                [
                    5.966213466261497 * x * y**2 * w * v**2
                    - 1.988737822087165 * x * w * v**2
                    - 1.988737822087165 * x * y**2 * w
                    + 0.6629126073623886 * x * w
                ],
                [
                    5.966213466261497 * x**2 * z * w * v**2
                    - 1.988737822087165 * z * w * v**2
                    - 1.988737822087165 * x**2 * z * w
                    + 0.6629126073623886 * z * w
                ],
                [
                    5.966213466261497 * y**2 * z * w * v**2
                    - 1.988737822087165 * z * w * v**2
                    - 1.988737822087165 * y**2 * z * w
                    + 0.6629126073623886 * z * w
                ],
                [
                    5.966213466261497 * x * z**2 * w * v**2
                    - 1.988737822087165 * x * w * v**2
                    - 1.988737822087165 * x * z**2 * w
                    + 0.6629126073623886 * x * w
                ],
                [
                    5.966213466261497 * y * z**2 * w * v**2
                    - 1.988737822087165 * y * w * v**2
                    - 1.988737822087165 * y * z**2 * w
                    + 0.6629126073623886 * y * w
                ],
                [
                    5.966213466261497 * x * y * w**2 * v**2
                    - 1.988737822087165 * x * y * v**2
                    - 1.988737822087165 * x * y * w**2
                    + 0.6629126073623886 * x * y
                ],
                [
                    5.966213466261497 * x * z * w**2 * v**2
                    - 1.988737822087165 * x * z * v**2
                    - 1.988737822087165 * x * z * w**2
                    + 0.6629126073623886 * x * z
                ],
                [
                    5.966213466261497 * y * z * w**2 * v**2
                    - 1.988737822087165 * y * z * v**2
                    - 1.988737822087165 * y * z * w**2
                    + 0.6629126073623886 * y * z
                ],
                [
                    6.075694404757367 * x**3 * y * z * w
                    - 3.64541664285442 * x * y * z * w
                ],
                [
                    6.075694404757367 * x * y**3 * z * w
                    - 3.64541664285442 * x * y * z * w
                ],
                [
                    6.075694404757367 * x * y * z**3 * w
                    - 3.64541664285442 * x * y * z * w
                ],
                [
                    6.075694404757367 * x * y * z * w**3
                    - 3.64541664285442 * x * y * z * w
                ],
                [
                    6.075694404757367 * x**3 * y * z * v
                    - 3.64541664285442 * x * y * z * v
                ],
                [
                    6.075694404757367 * x * y**3 * z * v
                    - 3.64541664285442 * x * y * z * v
                ],
                [
                    6.075694404757367 * x * y * z**3 * v
                    - 3.64541664285442 * x * y * z * v
                ],
                [
                    6.075694404757367 * x**3 * y * w * v
                    - 3.64541664285442 * x * y * w * v
                ],
                [
                    6.075694404757367 * x * y**3 * w * v
                    - 3.64541664285442 * x * y * w * v
                ],
                [
                    6.075694404757367 * x**3 * z * w * v
                    - 3.64541664285442 * x * z * w * v
                ],
                [
                    6.075694404757367 * y**3 * z * w * v
                    - 3.64541664285442 * y * z * w * v
                ],
                [
                    6.075694404757367 * x * z**3 * w * v
                    - 3.64541664285442 * x * z * w * v
                ],
                [
                    6.075694404757367 * y * z**3 * w * v
                    - 3.64541664285442 * y * z * w * v
                ],
                [
                    6.075694404757367 * x * y * w**3 * v
                    - 3.64541664285442 * x * y * w * v
                ],
                [
                    6.075694404757367 * x * z * w**3 * v
                    - 3.64541664285442 * x * z * w * v
                ],
                [
                    6.075694404757367 * y * z * w**3 * v
                    - 3.64541664285442 * y * z * w * v
                ],
                [
                    6.075694404757367 * x * y * z * v**3
                    - 3.64541664285442 * x * y * z * v
                ],
                [
                    6.075694404757367 * x * y * w * v**3
                    - 3.64541664285442 * x * y * w * v
                ],
                [
                    6.075694404757367 * x * z * w * v**3
                    - 3.64541664285442 * x * z * w * v
                ],
                [
                    6.075694404757367 * y * z * w * v**3
                    - 3.64541664285442 * y * z * w * v
                ],
                [
                    6.960582377305069 * x**4 * y * z
                    - 5.966213466261488 * x**2 * y * z
                    + 0.5966213466261489 * y * z
                ],
                [
                    6.960582377305069 * x * y**4 * z
                    - 5.966213466261488 * x * y**2 * z
                    + 0.5966213466261489 * x * z
                ],
                [
                    6.960582377305069 * x * y * z**4
                    - 5.966213466261488 * x * y * z**2
                    + 0.5966213466261489 * x * y
                ],
                [
                    6.960582377305069 * x**4 * y * w
                    - 5.966213466261488 * x**2 * y * w
                    + 0.5966213466261489 * y * w
                ],
                [
                    6.960582377305069 * x * y**4 * w
                    - 5.966213466261488 * x * y**2 * w
                    + 0.5966213466261489 * x * w
                ],
                [
                    6.960582377305069 * x**4 * z * w
                    - 5.966213466261488 * x**2 * z * w
                    + 0.5966213466261489 * z * w
                ],
                [
                    6.960582377305069 * y**4 * z * w
                    - 5.966213466261488 * y**2 * z * w
                    + 0.5966213466261489 * z * w
                ],
                [
                    6.960582377305069 * x * z**4 * w
                    - 5.966213466261488 * x * z**2 * w
                    + 0.5966213466261489 * x * w
                ],
                [
                    6.960582377305069 * y * z**4 * w
                    - 5.966213466261488 * y * z**2 * w
                    + 0.5966213466261489 * y * w
                ],
                [
                    6.960582377305069 * x * y * w**4
                    - 5.966213466261488 * x * y * w**2
                    + 0.5966213466261489 * x * y
                ],
                [
                    6.960582377305069 * x * z * w**4
                    - 5.966213466261488 * x * z * w**2
                    + 0.5966213466261489 * x * z
                ],
                [
                    6.960582377305069 * y * z * w**4
                    - 5.966213466261488 * y * z * w**2
                    + 0.5966213466261489 * y * z
                ],
                [
                    6.960582377305069 * x**4 * y * v
                    - 5.966213466261488 * x**2 * y * v
                    + 0.5966213466261489 * y * v
                ],
                [
                    6.960582377305069 * x * y**4 * v
                    - 5.966213466261488 * x * y**2 * v
                    + 0.5966213466261489 * x * v
                ],
                [
                    6.960582377305069 * x**4 * z * v
                    - 5.966213466261488 * x**2 * z * v
                    + 0.5966213466261489 * z * v
                ],
                [
                    6.960582377305069 * y**4 * z * v
                    - 5.966213466261488 * y**2 * z * v
                    + 0.5966213466261489 * z * v
                ],
                [
                    6.960582377305069 * x * z**4 * v
                    - 5.966213466261488 * x * z**2 * v
                    + 0.5966213466261489 * x * v
                ],
                [
                    6.960582377305069 * y * z**4 * v
                    - 5.966213466261488 * y * z**2 * v
                    + 0.5966213466261489 * y * v
                ],
                [
                    6.960582377305069 * x**4 * w * v
                    - 5.966213466261488 * x**2 * w * v
                    + 0.5966213466261489 * w * v
                ],
                [
                    6.960582377305069 * y**4 * w * v
                    - 5.966213466261488 * y**2 * w * v
                    + 0.5966213466261489 * w * v
                ],
                [
                    6.960582377305069 * z**4 * w * v
                    - 5.966213466261488 * z**2 * w * v
                    + 0.5966213466261489 * w * v
                ],
                [
                    6.960582377305069 * x * w**4 * v
                    - 5.966213466261488 * x * w**2 * v
                    + 0.5966213466261489 * x * v
                ],
                [
                    6.960582377305069 * y * w**4 * v
                    - 5.966213466261488 * y * w**2 * v
                    + 0.5966213466261489 * y * v
                ],
                [
                    6.960582377305069 * z * w**4 * v
                    - 5.966213466261488 * z * w**2 * v
                    + 0.5966213466261489 * z * v
                ],
                [
                    6.960582377305069 * x * y * v**4
                    - 5.966213466261488 * x * y * v**2
                    + 0.5966213466261489 * x * y
                ],
                [
                    6.960582377305069 * x * z * v**4
                    - 5.966213466261488 * x * z * v**2
                    + 0.5966213466261489 * x * z
                ],
                [
                    6.960582377305069 * y * z * v**4
                    - 5.966213466261488 * y * z * v**2
                    + 0.5966213466261489 * y * z
                ],
                [
                    6.960582377305069 * x * w * v**4
                    - 5.966213466261488 * x * w * v**2
                    + 0.5966213466261489 * x * w
                ],
                [
                    6.960582377305069 * y * w * v**4
                    - 5.966213466261488 * y * w * v**2
                    + 0.5966213466261489 * y * w
                ],
                [
                    6.960582377305069 * z * w * v**4
                    - 5.966213466261488 * z * w * v**2
                    + 0.5966213466261489 * z * w
                ],
                [
                    10.33378485236653 * x**2 * y**2 * z * w * v
                    - 3.444594950788842 * y**2 * z * w * v
                    - 3.444594950788842 * x**2 * z * w * v
                    + 1.148198316929614 * z * w * v
                ],
                [
                    10.33378485236653 * x**2 * y * z**2 * w * v
                    - 3.444594950788842 * y * z**2 * w * v
                    - 3.444594950788842 * x**2 * y * w * v
                    + 1.148198316929614 * y * w * v
                ],
                [
                    10.33378485236653 * x * y**2 * z**2 * w * v
                    - 3.444594950788842 * x * z**2 * w * v
                    - 3.444594950788842 * x * y**2 * w * v
                    + 1.148198316929614 * x * w * v
                ],
                [
                    10.33378485236653 * x**2 * y * z * w**2 * v
                    - 3.444594950788842 * y * z * w**2 * v
                    - 3.444594950788842 * x**2 * y * z * v
                    + 1.148198316929614 * y * z * v
                ],
                [
                    10.33378485236653 * x * y**2 * z * w**2 * v
                    - 3.444594950788842 * x * z * w**2 * v
                    - 3.444594950788842 * x * y**2 * z * v
                    + 1.148198316929614 * x * z * v
                ],
                [
                    10.33378485236653 * x * y * z**2 * w**2 * v
                    - 3.444594950788842 * x * y * w**2 * v
                    - 3.444594950788842 * x * y * z**2 * v
                    + 1.148198316929614 * x * y * v
                ],
                [
                    10.33378485236653 * x**2 * y * z * w * v**2
                    - 3.444594950788842 * y * z * w * v**2
                    - 3.444594950788842 * x**2 * y * z * w
                    + 1.148198316929614 * y * z * w
                ],
                [
                    10.33378485236653 * x * y**2 * z * w * v**2
                    - 3.444594950788842 * x * z * w * v**2
                    - 3.444594950788842 * x * y**2 * z * w
                    + 1.148198316929614 * x * z * w
                ],
                [
                    10.33378485236653 * x * y * z**2 * w * v**2
                    - 3.444594950788842 * x * y * w * v**2
                    - 3.444594950788842 * x * y * z**2 * w
                    + 1.148198316929614 * x * y * w
                ],
                [
                    10.33378485236653 * x * y * z * w**2 * v**2
                    - 3.444594950788842 * x * y * z * v**2
                    - 3.444594950788842 * x * y * z * w**2
                    + 1.148198316929614 * x * y * z
                ],
                [
                    10.52341140030171 * x**3 * y * z * w * v
                    - 6.314046840181025 * x * y * z * w * v
                ],
                [
                    10.52341140030171 * x * y**3 * z * w * v
                    - 6.314046840181025 * x * y * z * w * v
                ],
                [
                    10.52341140030171 * x * y * z**3 * w * v
                    - 6.314046840181025 * x * y * z * w * v
                ],
                [
                    10.52341140030171 * x * y * z * w**3 * v
                    - 6.314046840181025 * x * y * z * w * v
                ],
                [
                    10.52341140030171 * x * y * z * w * v**3
                    - 6.314046840181025 * x * y * z * w * v
                ],
                [
                    12.05608232776096 * x**4 * y * z * w
                    - 10.33378485236654 * x**2 * y * z * w
                    + 1.033378485236654 * y * z * w
                ],
                [
                    12.05608232776096 * x * y**4 * z * w
                    - 10.33378485236654 * x * y**2 * z * w
                    + 1.033378485236654 * x * z * w
                ],
                [
                    12.05608232776096 * x * y * z**4 * w
                    - 10.33378485236654 * x * y * z**2 * w
                    + 1.033378485236654 * x * y * w
                ],
                [
                    12.05608232776096 * x * y * z * w**4
                    - 10.33378485236654 * x * y * z * w**2
                    + 1.033378485236654 * x * y * z
                ],
                [
                    12.05608232776096 * x**4 * y * z * v
                    - 10.33378485236654 * x**2 * y * z * v
                    + 1.033378485236654 * y * z * v
                ],
                [
                    12.05608232776096 * x * y**4 * z * v
                    - 10.33378485236654 * x * y**2 * z * v
                    + 1.033378485236654 * x * z * v
                ],
                [
                    12.05608232776096 * x * y * z**4 * v
                    - 10.33378485236654 * x * y * z**2 * v
                    + 1.033378485236654 * x * y * v
                ],
                [
                    12.05608232776096 * x**4 * y * w * v
                    - 10.33378485236654 * x**2 * y * w * v
                    + 1.033378485236654 * y * w * v
                ],
                [
                    12.05608232776096 * x * y**4 * w * v
                    - 10.33378485236654 * x * y**2 * w * v
                    + 1.033378485236654 * x * w * v
                ],
                [
                    12.05608232776096 * x**4 * z * w * v
                    - 10.33378485236654 * x**2 * z * w * v
                    + 1.033378485236654 * z * w * v
                ],
                [
                    12.05608232776096 * y**4 * z * w * v
                    - 10.33378485236654 * y**2 * z * w * v
                    + 1.033378485236654 * z * w * v
                ],
                [
                    12.05608232776096 * x * z**4 * w * v
                    - 10.33378485236654 * x * z**2 * w * v
                    + 1.033378485236654 * x * w * v
                ],
                [
                    12.05608232776096 * y * z**4 * w * v
                    - 10.33378485236654 * y * z**2 * w * v
                    + 1.033378485236654 * y * w * v
                ],
                [
                    12.05608232776096 * x * y * w**4 * v
                    - 10.33378485236654 * x * y * w**2 * v
                    + 1.033378485236654 * x * y * v
                ],
                [
                    12.05608232776096 * x * z * w**4 * v
                    - 10.33378485236654 * x * z * w**2 * v
                    + 1.033378485236654 * x * z * v
                ],
                [
                    12.05608232776096 * y * z * w**4 * v
                    - 10.33378485236654 * y * z * w**2 * v
                    + 1.033378485236654 * y * z * v
                ],
                [
                    12.05608232776096 * x * y * z * v**4
                    - 10.33378485236654 * x * y * z * v**2
                    + 1.033378485236654 * x * y * z
                ],
                [
                    12.05608232776096 * x * y * w * v**4
                    - 10.33378485236654 * x * y * w * v**2
                    + 1.033378485236654 * x * y * w
                ],
                [
                    12.05608232776096 * x * z * w * v**4
                    - 10.33378485236654 * x * z * w * v**2
                    + 1.033378485236654 * x * z * w
                ],
                [
                    12.05608232776096 * y * z * w * v**4
                    - 10.33378485236654 * y * z * w * v**2
                    + 1.033378485236654 * y * z * w
                ],
                [
                    20.88174713191521 * x**4 * y * z * w * v
                    - 17.89864039878447 * x**2 * y * z * w * v
                    + 1.789864039878446 * y * z * w * v
                ],
                [
                    20.88174713191521 * x * y**4 * z * w * v
                    - 17.89864039878447 * x * y**2 * z * w * v
                    + 1.789864039878446 * x * z * w * v
                ],
                [
                    20.88174713191521 * x * y * z**4 * w * v
                    - 17.89864039878447 * x * y * z**2 * w * v
                    + 1.789864039878446 * x * y * w * v
                ],
                [
                    20.88174713191521 * x * y * z * w**4 * v
                    - 17.89864039878447 * x * y * z * w**2 * v
                    + 1.789864039878446 * x * y * z * v
                ],
                [
                    20.88174713191521 * x * y * z * w * v**4
                    - 17.89864039878447 * x * y * z * w * v**2
                    + 1.789864039878446 * x * y * z * w
                ],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, interpList.shape[0]):
                for m in range(0, interpList.shape[0]):
                  for n in range(0, functionVector.shape[0]):
                    interpMatrix[
                        m
                        + l * interpList.shape[0]
                        + k * interpList.shape[0] * interpList.shape[0]
                        + j
                        * interpList.shape[0]
                        * interpList.shape[0]
                        * interpList.shape[0]
                        + i
                        * interpList.shape[0]
                        * interpList.shape[0]
                        * interpList.shape[0]
                        * interpList.shape[0],
                        n,
                    ] = (
                        functionVector[n]
                        .subs(x, interpList[m])
                        .subs(y, interpList[l])
                        .subs(z, interpList[k])
                        .subs(w, interpList[j])
                        .subs(v, interpList[i])
                    )
      else:
        raise NameError(
            "interpMatrix: Order {} is not supported!\nPolynomial order must be <5".format(
                order
            )
        )

    elif modal and basis_type == "gkhybrid":
      if order == 1:
        functionVector = Matrix(
            [
                [0.1767766952966368],
                [0.3061862178478971 * x],
                [0.3061862178478971 * y],
                [0.3061862178478971 * z],
                [0.3061862178478971 * w],
                [0.3061862178478971 * v],
                [0.5303300858899105 * x * y],
                [0.5303300858899105 * x * z],
                [0.5303300858899105 * y * z],
                [0.5303300858899105 * w * x],
                [0.5303300858899105 * w * y],
                [0.5303300858899105 * w * z],
                [0.5303300858899105 * v * x],
                [0.5303300858899105 * v * y],
                [0.5303300858899105 * v * z],
                [0.5303300858899105 * v * w],
                [0.9185586535436913 * x * y * z],
                [0.9185586535436913 * w * x * y],
                [0.9185586535436913 * w * x * z],
                [0.9185586535436913 * w * y * z],
                [0.9185586535436913 * v * x * y],
                [0.9185586535436913 * v * x * z],
                [0.9185586535436913 * v * y * z],
                [0.9185586535436913 * v * w * x],
                [0.9185586535436913 * v * w * y],
                [0.9185586535436913 * v * w * z],
                [1.590990257669731 * w * x * y * z],
                [1.590990257669731 * v * x * y * z],
                [1.590990257669731 * v * w * x * y],
                [1.590990257669731 * v * w * x * z],
                [1.590990257669731 * v * w * y * z],
                [2.755675960631073 * v * w * x * y * z],
                [0.592927061281571 * (w**2 - 0.3333333333333333)],
                [1.026979795322186 * (w**2 * x - 0.3333333333333333 * x)],
                [1.026979795322186 * (w**2 * y - 0.3333333333333333 * y)],
                [1.026979795322186 * (w**2 * z - 0.3333333333333333 * z)],
                [1.026979795322186 * (v * w**2 - 0.3333333333333333 * v)],
                [1.778781183844713 * (w**2 * x * y - 0.3333333333333333 * x * y)],
                [1.778781183844713 * (w**2 * x * z - 0.3333333333333333 * x * z)],
                [1.778781183844713 * (w**2 * y * z - 0.3333333333333333 * y * z)],
                [1.778781183844713 * (v * w**2 * x - 0.3333333333333333 * v * x)],
                [1.778781183844713 * (v * w**2 * y - 0.3333333333333333 * v * y)],
                [1.778781183844713 * (v * w**2 * z - 0.3333333333333333 * v * z)],
                [
                    3.080939385966558
                    * (w**2 * x * y * z - 0.3333333333333333 * x * y * z)
                ],
                [
                    3.080939385966558
                    * (v * w**2 * x * y - 0.3333333333333333 * v * x * y)
                ],
                [
                    3.080939385966558
                    * (v * w**2 * x * z - 0.3333333333333333 * v * x * z)
                ],
                [
                    3.080939385966558
                    * (v * w**2 * y * z - 0.3333333333333333 * v * y * z)
                ],
                [
                    5.336343551534138
                    * (v * w**2 * x * y * z - 0.3333333333333333 * v * x * y * z)
                ],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpListND[0].shape[0]
                * interpListND[1].shape[0]
                * interpListND[2].shape[0]
                * interpListND[3].shape[0]
                * interpListND[4].shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpListND[4].shape[0]):
          for j in range(0, interpListND[3].shape[0]):
            for k in range(0, interpListND[2].shape[0]):
              for l in range(0, interpListND[1].shape[0]):
                for m in range(0, interpListND[0].shape[0]):
                  for n in range(0, functionVector.shape[0]):
                    interpMatrix[
                        m
                        + l * interpListND[0].shape[0]
                        + k * interpListND[0].shape[0] * interpListND[1].shape[0]
                        + j
                        * interpListND[0].shape[0]
                        * interpListND[1].shape[0]
                        * interpListND[2].shape[0]
                        + i
                        * interpListND[0].shape[0]
                        * interpListND[1].shape[0]
                        * interpListND[2].shape[0]
                        * interpListND[3].shape[0],
                        n,
                    ] = (
                        functionVector[n]
                        .subs(x, interpListND[0][m])
                        .subs(y, interpListND[1][l])
                        .subs(z, interpListND[2][k])
                        .subs(w, interpListND[3][j])
                        .subs(v, interpListND[4][i])
                    )

      else:
        raise NameError(
            "interpMatrix: Order {} is not supported!\nPolynomial order must be =1".format(
                order
            )
        )

    elif modal == False and basis_type == "serendipity":
      if order == 1:
        functionVector = Matrix(
            [
                [
                    (v * w) / 32.0
                    - w / 32.0
                    - x / 32.0
                    - y / 32.0
                    - z / 32.0
                    - v / 32.0
                    + (v * x) / 32.0
                    + (v * y) / 32.0
                    + (w * x) / 32.0
                    + (v * z) / 32.0
                    + (w * y) / 32.0
                    + (w * z) / 32.0
                    + (x * y) / 32.0
                    + (x * z) / 32.0
                    + (y * z) / 32.0
                    - (v * w * x) / 32.0
                    - (v * w * y) / 32.0
                    - (v * w * z) / 32.0
                    - (v * x * y) / 32.0
                    - (v * x * z) / 32.0
                    - (w * x * y) / 32.0
                    - (v * y * z) / 32.0
                    - (w * x * z) / 32.0
                    - (w * y * z) / 32.0
                    - (x * y * z) / 32.0
                    + (v * w * x * y) / 32.0
                    + (v * w * x * z) / 32.0
                    + (v * w * y * z) / 32.0
                    + (v * x * y * z) / 32.0
                    + (w * x * y * z) / 32.0
                    - (v * w * x * y * z) / 32.0
                    + 1.0 / 32.0
                ],
                [
                    x / 32.0
                    - w / 32.0
                    - v / 32.0
                    - y / 32.0
                    - z / 32.0
                    + (v * w) / 32.0
                    - (v * x) / 32.0
                    + (v * y) / 32.0
                    - (w * x) / 32.0
                    + (v * z) / 32.0
                    + (w * y) / 32.0
                    + (w * z) / 32.0
                    - (x * y) / 32.0
                    - (x * z) / 32.0
                    + (y * z) / 32.0
                    + (v * w * x) / 32.0
                    - (v * w * y) / 32.0
                    - (v * w * z) / 32.0
                    + (v * x * y) / 32.0
                    + (v * x * z) / 32.0
                    + (w * x * y) / 32.0
                    - (v * y * z) / 32.0
                    + (w * x * z) / 32.0
                    - (w * y * z) / 32.0
                    + (x * y * z) / 32.0
                    - (v * w * x * y) / 32.0
                    - (v * w * x * z) / 32.0
                    + (v * w * y * z) / 32.0
                    - (v * x * y * z) / 32.0
                    - (w * x * y * z) / 32.0
                    + (v * w * x * y * z) / 32.0
                    + 1.0 / 32.0
                ],
                [
                    y / 32.0
                    - w / 32.0
                    - x / 32.0
                    - v / 32.0
                    - z / 32.0
                    + (v * w) / 32.0
                    + (v * x) / 32.0
                    - (v * y) / 32.0
                    + (w * x) / 32.0
                    + (v * z) / 32.0
                    - (w * y) / 32.0
                    + (w * z) / 32.0
                    - (x * y) / 32.0
                    + (x * z) / 32.0
                    - (y * z) / 32.0
                    - (v * w * x) / 32.0
                    + (v * w * y) / 32.0
                    - (v * w * z) / 32.0
                    + (v * x * y) / 32.0
                    - (v * x * z) / 32.0
                    + (w * x * y) / 32.0
                    + (v * y * z) / 32.0
                    - (w * x * z) / 32.0
                    + (w * y * z) / 32.0
                    + (x * y * z) / 32.0
                    - (v * w * x * y) / 32.0
                    + (v * w * x * z) / 32.0
                    - (v * w * y * z) / 32.0
                    - (v * x * y * z) / 32.0
                    - (w * x * y * z) / 32.0
                    + (v * w * x * y * z) / 32.0
                    + 1.0 / 32.0
                ],
                [
                    x / 32.0
                    - w / 32.0
                    - v / 32.0
                    + y / 32.0
                    - z / 32.0
                    + (v * w) / 32.0
                    - (v * x) / 32.0
                    - (v * y) / 32.0
                    - (w * x) / 32.0
                    + (v * z) / 32.0
                    - (w * y) / 32.0
                    + (w * z) / 32.0
                    + (x * y) / 32.0
                    - (x * z) / 32.0
                    - (y * z) / 32.0
                    + (v * w * x) / 32.0
                    + (v * w * y) / 32.0
                    - (v * w * z) / 32.0
                    - (v * x * y) / 32.0
                    + (v * x * z) / 32.0
                    - (w * x * y) / 32.0
                    + (v * y * z) / 32.0
                    + (w * x * z) / 32.0
                    + (w * y * z) / 32.0
                    - (x * y * z) / 32.0
                    + (v * w * x * y) / 32.0
                    - (v * w * x * z) / 32.0
                    - (v * w * y * z) / 32.0
                    + (v * x * y * z) / 32.0
                    + (w * x * y * z) / 32.0
                    - (v * w * x * y * z) / 32.0
                    + 1.0 / 32.0
                ],
                [
                    z / 32.0
                    - w / 32.0
                    - x / 32.0
                    - y / 32.0
                    - v / 32.0
                    + (v * w) / 32.0
                    + (v * x) / 32.0
                    + (v * y) / 32.0
                    + (w * x) / 32.0
                    - (v * z) / 32.0
                    + (w * y) / 32.0
                    - (w * z) / 32.0
                    + (x * y) / 32.0
                    - (x * z) / 32.0
                    - (y * z) / 32.0
                    - (v * w * x) / 32.0
                    - (v * w * y) / 32.0
                    + (v * w * z) / 32.0
                    - (v * x * y) / 32.0
                    + (v * x * z) / 32.0
                    - (w * x * y) / 32.0
                    + (v * y * z) / 32.0
                    + (w * x * z) / 32.0
                    + (w * y * z) / 32.0
                    + (x * y * z) / 32.0
                    + (v * w * x * y) / 32.0
                    - (v * w * x * z) / 32.0
                    - (v * w * y * z) / 32.0
                    - (v * x * y * z) / 32.0
                    - (w * x * y * z) / 32.0
                    + (v * w * x * y * z) / 32.0
                    + 1.0 / 32.0
                ],
                [
                    x / 32.0
                    - w / 32.0
                    - v / 32.0
                    - y / 32.0
                    + z / 32.0
                    + (v * w) / 32.0
                    - (v * x) / 32.0
                    + (v * y) / 32.0
                    - (w * x) / 32.0
                    - (v * z) / 32.0
                    + (w * y) / 32.0
                    - (w * z) / 32.0
                    - (x * y) / 32.0
                    + (x * z) / 32.0
                    - (y * z) / 32.0
                    + (v * w * x) / 32.0
                    - (v * w * y) / 32.0
                    + (v * w * z) / 32.0
                    + (v * x * y) / 32.0
                    - (v * x * z) / 32.0
                    + (w * x * y) / 32.0
                    + (v * y * z) / 32.0
                    - (w * x * z) / 32.0
                    + (w * y * z) / 32.0
                    - (x * y * z) / 32.0
                    - (v * w * x * y) / 32.0
                    + (v * w * x * z) / 32.0
                    - (v * w * y * z) / 32.0
                    + (v * x * y * z) / 32.0
                    + (w * x * y * z) / 32.0
                    - (v * w * x * y * z) / 32.0
                    + 1.0 / 32.0
                ],
                [
                    y / 32.0
                    - w / 32.0
                    - x / 32.0
                    - v / 32.0
                    + z / 32.0
                    + (v * w) / 32.0
                    + (v * x) / 32.0
                    - (v * y) / 32.0
                    + (w * x) / 32.0
                    - (v * z) / 32.0
                    - (w * y) / 32.0
                    - (w * z) / 32.0
                    - (x * y) / 32.0
                    - (x * z) / 32.0
                    + (y * z) / 32.0
                    - (v * w * x) / 32.0
                    + (v * w * y) / 32.0
                    + (v * w * z) / 32.0
                    + (v * x * y) / 32.0
                    + (v * x * z) / 32.0
                    + (w * x * y) / 32.0
                    - (v * y * z) / 32.0
                    + (w * x * z) / 32.0
                    - (w * y * z) / 32.0
                    - (x * y * z) / 32.0
                    - (v * w * x * y) / 32.0
                    - (v * w * x * z) / 32.0
                    + (v * w * y * z) / 32.0
                    + (v * x * y * z) / 32.0
                    + (w * x * y * z) / 32.0
                    - (v * w * x * y * z) / 32.0
                    + 1.0 / 32.0
                ],
                [
                    x / 32.0
                    - w / 32.0
                    - v / 32.0
                    + y / 32.0
                    + z / 32.0
                    + (v * w) / 32.0
                    - (v * x) / 32.0
                    - (v * y) / 32.0
                    - (w * x) / 32.0
                    - (v * z) / 32.0
                    - (w * y) / 32.0
                    - (w * z) / 32.0
                    + (x * y) / 32.0
                    + (x * z) / 32.0
                    + (y * z) / 32.0
                    + (v * w * x) / 32.0
                    + (v * w * y) / 32.0
                    + (v * w * z) / 32.0
                    - (v * x * y) / 32.0
                    - (v * x * z) / 32.0
                    - (w * x * y) / 32.0
                    - (v * y * z) / 32.0
                    - (w * x * z) / 32.0
                    - (w * y * z) / 32.0
                    + (x * y * z) / 32.0
                    + (v * w * x * y) / 32.0
                    + (v * w * x * z) / 32.0
                    + (v * w * y * z) / 32.0
                    - (v * x * y * z) / 32.0
                    - (w * x * y * z) / 32.0
                    + (v * w * x * y * z) / 32.0
                    + 1.0 / 32.0
                ],
                [
                    w / 32.0
                    - v / 32.0
                    - x / 32.0
                    - y / 32.0
                    - z / 32.0
                    - (v * w) / 32.0
                    + (v * x) / 32.0
                    + (v * y) / 32.0
                    - (w * x) / 32.0
                    + (v * z) / 32.0
                    - (w * y) / 32.0
                    - (w * z) / 32.0
                    + (x * y) / 32.0
                    + (x * z) / 32.0
                    + (y * z) / 32.0
                    + (v * w * x) / 32.0
                    + (v * w * y) / 32.0
                    + (v * w * z) / 32.0
                    - (v * x * y) / 32.0
                    - (v * x * z) / 32.0
                    + (w * x * y) / 32.0
                    - (v * y * z) / 32.0
                    + (w * x * z) / 32.0
                    + (w * y * z) / 32.0
                    - (x * y * z) / 32.0
                    - (v * w * x * y) / 32.0
                    - (v * w * x * z) / 32.0
                    - (v * w * y * z) / 32.0
                    + (v * x * y * z) / 32.0
                    - (w * x * y * z) / 32.0
                    + (v * w * x * y * z) / 32.0
                    + 1.0 / 32.0
                ],
                [
                    w / 32.0
                    - v / 32.0
                    + x / 32.0
                    - y / 32.0
                    - z / 32.0
                    - (v * w) / 32.0
                    - (v * x) / 32.0
                    + (v * y) / 32.0
                    + (w * x) / 32.0
                    + (v * z) / 32.0
                    - (w * y) / 32.0
                    - (w * z) / 32.0
                    - (x * y) / 32.0
                    - (x * z) / 32.0
                    + (y * z) / 32.0
                    - (v * w * x) / 32.0
                    + (v * w * y) / 32.0
                    + (v * w * z) / 32.0
                    + (v * x * y) / 32.0
                    + (v * x * z) / 32.0
                    - (w * x * y) / 32.0
                    - (v * y * z) / 32.0
                    - (w * x * z) / 32.0
                    + (w * y * z) / 32.0
                    + (x * y * z) / 32.0
                    + (v * w * x * y) / 32.0
                    + (v * w * x * z) / 32.0
                    - (v * w * y * z) / 32.0
                    - (v * x * y * z) / 32.0
                    + (w * x * y * z) / 32.0
                    - (v * w * x * y * z) / 32.0
                    + 1.0 / 32.0
                ],
                [
                    w / 32.0
                    - v / 32.0
                    - x / 32.0
                    + y / 32.0
                    - z / 32.0
                    - (v * w) / 32.0
                    + (v * x) / 32.0
                    - (v * y) / 32.0
                    - (w * x) / 32.0
                    + (v * z) / 32.0
                    + (w * y) / 32.0
                    - (w * z) / 32.0
                    - (x * y) / 32.0
                    + (x * z) / 32.0
                    - (y * z) / 32.0
                    + (v * w * x) / 32.0
                    - (v * w * y) / 32.0
                    + (v * w * z) / 32.0
                    + (v * x * y) / 32.0
                    - (v * x * z) / 32.0
                    - (w * x * y) / 32.0
                    + (v * y * z) / 32.0
                    + (w * x * z) / 32.0
                    - (w * y * z) / 32.0
                    + (x * y * z) / 32.0
                    + (v * w * x * y) / 32.0
                    - (v * w * x * z) / 32.0
                    + (v * w * y * z) / 32.0
                    - (v * x * y * z) / 32.0
                    + (w * x * y * z) / 32.0
                    - (v * w * x * y * z) / 32.0
                    + 1.0 / 32.0
                ],
                [
                    w / 32.0
                    - v / 32.0
                    + x / 32.0
                    + y / 32.0
                    - z / 32.0
                    - (v * w) / 32.0
                    - (v * x) / 32.0
                    - (v * y) / 32.0
                    + (w * x) / 32.0
                    + (v * z) / 32.0
                    + (w * y) / 32.0
                    - (w * z) / 32.0
                    + (x * y) / 32.0
                    - (x * z) / 32.0
                    - (y * z) / 32.0
                    - (v * w * x) / 32.0
                    - (v * w * y) / 32.0
                    + (v * w * z) / 32.0
                    - (v * x * y) / 32.0
                    + (v * x * z) / 32.0
                    + (w * x * y) / 32.0
                    + (v * y * z) / 32.0
                    - (w * x * z) / 32.0
                    - (w * y * z) / 32.0
                    - (x * y * z) / 32.0
                    - (v * w * x * y) / 32.0
                    + (v * w * x * z) / 32.0
                    + (v * w * y * z) / 32.0
                    + (v * x * y * z) / 32.0
                    - (w * x * y * z) / 32.0
                    + (v * w * x * y * z) / 32.0
                    + 1.0 / 32.0
                ],
                [
                    w / 32.0
                    - v / 32.0
                    - x / 32.0
                    - y / 32.0
                    + z / 32.0
                    - (v * w) / 32.0
                    + (v * x) / 32.0
                    + (v * y) / 32.0
                    - (w * x) / 32.0
                    - (v * z) / 32.0
                    - (w * y) / 32.0
                    + (w * z) / 32.0
                    + (x * y) / 32.0
                    - (x * z) / 32.0
                    - (y * z) / 32.0
                    + (v * w * x) / 32.0
                    + (v * w * y) / 32.0
                    - (v * w * z) / 32.0
                    - (v * x * y) / 32.0
                    + (v * x * z) / 32.0
                    + (w * x * y) / 32.0
                    + (v * y * z) / 32.0
                    - (w * x * z) / 32.0
                    - (w * y * z) / 32.0
                    + (x * y * z) / 32.0
                    - (v * w * x * y) / 32.0
                    + (v * w * x * z) / 32.0
                    + (v * w * y * z) / 32.0
                    - (v * x * y * z) / 32.0
                    + (w * x * y * z) / 32.0
                    - (v * w * x * y * z) / 32.0
                    + 1.0 / 32.0
                ],
                [
                    w / 32.0
                    - v / 32.0
                    + x / 32.0
                    - y / 32.0
                    + z / 32.0
                    - (v * w) / 32.0
                    - (v * x) / 32.0
                    + (v * y) / 32.0
                    + (w * x) / 32.0
                    - (v * z) / 32.0
                    - (w * y) / 32.0
                    + (w * z) / 32.0
                    - (x * y) / 32.0
                    + (x * z) / 32.0
                    - (y * z) / 32.0
                    - (v * w * x) / 32.0
                    + (v * w * y) / 32.0
                    - (v * w * z) / 32.0
                    + (v * x * y) / 32.0
                    - (v * x * z) / 32.0
                    - (w * x * y) / 32.0
                    + (v * y * z) / 32.0
                    + (w * x * z) / 32.0
                    - (w * y * z) / 32.0
                    - (x * y * z) / 32.0
                    + (v * w * x * y) / 32.0
                    - (v * w * x * z) / 32.0
                    + (v * w * y * z) / 32.0
                    + (v * x * y * z) / 32.0
                    - (w * x * y * z) / 32.0
                    + (v * w * x * y * z) / 32.0
                    + 1.0 / 32.0
                ],
                [
                    w / 32.0
                    - v / 32.0
                    - x / 32.0
                    + y / 32.0
                    + z / 32.0
                    - (v * w) / 32.0
                    + (v * x) / 32.0
                    - (v * y) / 32.0
                    - (w * x) / 32.0
                    - (v * z) / 32.0
                    + (w * y) / 32.0
                    + (w * z) / 32.0
                    - (x * y) / 32.0
                    - (x * z) / 32.0
                    + (y * z) / 32.0
                    + (v * w * x) / 32.0
                    - (v * w * y) / 32.0
                    - (v * w * z) / 32.0
                    + (v * x * y) / 32.0
                    + (v * x * z) / 32.0
                    - (w * x * y) / 32.0
                    - (v * y * z) / 32.0
                    - (w * x * z) / 32.0
                    + (w * y * z) / 32.0
                    - (x * y * z) / 32.0
                    + (v * w * x * y) / 32.0
                    + (v * w * x * z) / 32.0
                    - (v * w * y * z) / 32.0
                    + (v * x * y * z) / 32.0
                    - (w * x * y * z) / 32.0
                    + (v * w * x * y * z) / 32.0
                    + 1.0 / 32.0
                ],
                [
                    w / 32.0
                    - v / 32.0
                    + x / 32.0
                    + y / 32.0
                    + z / 32.0
                    - (v * w) / 32.0
                    - (v * x) / 32.0
                    - (v * y) / 32.0
                    + (w * x) / 32.0
                    - (v * z) / 32.0
                    + (w * y) / 32.0
                    + (w * z) / 32.0
                    + (x * y) / 32.0
                    + (x * z) / 32.0
                    + (y * z) / 32.0
                    - (v * w * x) / 32.0
                    - (v * w * y) / 32.0
                    - (v * w * z) / 32.0
                    - (v * x * y) / 32.0
                    - (v * x * z) / 32.0
                    + (w * x * y) / 32.0
                    - (v * y * z) / 32.0
                    + (w * x * z) / 32.0
                    + (w * y * z) / 32.0
                    + (x * y * z) / 32.0
                    - (v * w * x * y) / 32.0
                    - (v * w * x * z) / 32.0
                    - (v * w * y * z) / 32.0
                    - (v * x * y * z) / 32.0
                    + (w * x * y * z) / 32.0
                    - (v * w * x * y * z) / 32.0
                    + 1.0 / 32.0
                ],
                [
                    v / 32.0
                    - w / 32.0
                    - x / 32.0
                    - y / 32.0
                    - z / 32.0
                    - (v * w) / 32.0
                    - (v * x) / 32.0
                    - (v * y) / 32.0
                    + (w * x) / 32.0
                    - (v * z) / 32.0
                    + (w * y) / 32.0
                    + (w * z) / 32.0
                    + (x * y) / 32.0
                    + (x * z) / 32.0
                    + (y * z) / 32.0
                    + (v * w * x) / 32.0
                    + (v * w * y) / 32.0
                    + (v * w * z) / 32.0
                    + (v * x * y) / 32.0
                    + (v * x * z) / 32.0
                    - (w * x * y) / 32.0
                    + (v * y * z) / 32.0
                    - (w * x * z) / 32.0
                    - (w * y * z) / 32.0
                    - (x * y * z) / 32.0
                    - (v * w * x * y) / 32.0
                    - (v * w * x * z) / 32.0
                    - (v * w * y * z) / 32.0
                    - (v * x * y * z) / 32.0
                    + (w * x * y * z) / 32.0
                    + (v * w * x * y * z) / 32.0
                    + 1.0 / 32.0
                ],
                [
                    v / 32.0
                    - w / 32.0
                    + x / 32.0
                    - y / 32.0
                    - z / 32.0
                    - (v * w) / 32.0
                    + (v * x) / 32.0
                    - (v * y) / 32.0
                    - (w * x) / 32.0
                    - (v * z) / 32.0
                    + (w * y) / 32.0
                    + (w * z) / 32.0
                    - (x * y) / 32.0
                    - (x * z) / 32.0
                    + (y * z) / 32.0
                    - (v * w * x) / 32.0
                    + (v * w * y) / 32.0
                    + (v * w * z) / 32.0
                    - (v * x * y) / 32.0
                    - (v * x * z) / 32.0
                    + (w * x * y) / 32.0
                    + (v * y * z) / 32.0
                    + (w * x * z) / 32.0
                    - (w * y * z) / 32.0
                    + (x * y * z) / 32.0
                    + (v * w * x * y) / 32.0
                    + (v * w * x * z) / 32.0
                    - (v * w * y * z) / 32.0
                    + (v * x * y * z) / 32.0
                    - (w * x * y * z) / 32.0
                    - (v * w * x * y * z) / 32.0
                    + 1.0 / 32.0
                ],
                [
                    v / 32.0
                    - w / 32.0
                    - x / 32.0
                    + y / 32.0
                    - z / 32.0
                    - (v * w) / 32.0
                    - (v * x) / 32.0
                    + (v * y) / 32.0
                    + (w * x) / 32.0
                    - (v * z) / 32.0
                    - (w * y) / 32.0
                    + (w * z) / 32.0
                    - (x * y) / 32.0
                    + (x * z) / 32.0
                    - (y * z) / 32.0
                    + (v * w * x) / 32.0
                    - (v * w * y) / 32.0
                    + (v * w * z) / 32.0
                    - (v * x * y) / 32.0
                    + (v * x * z) / 32.0
                    + (w * x * y) / 32.0
                    - (v * y * z) / 32.0
                    - (w * x * z) / 32.0
                    + (w * y * z) / 32.0
                    + (x * y * z) / 32.0
                    + (v * w * x * y) / 32.0
                    - (v * w * x * z) / 32.0
                    + (v * w * y * z) / 32.0
                    + (v * x * y * z) / 32.0
                    - (w * x * y * z) / 32.0
                    - (v * w * x * y * z) / 32.0
                    + 1.0 / 32.0
                ],
                [
                    v / 32.0
                    - w / 32.0
                    + x / 32.0
                    + y / 32.0
                    - z / 32.0
                    - (v * w) / 32.0
                    + (v * x) / 32.0
                    + (v * y) / 32.0
                    - (w * x) / 32.0
                    - (v * z) / 32.0
                    - (w * y) / 32.0
                    + (w * z) / 32.0
                    + (x * y) / 32.0
                    - (x * z) / 32.0
                    - (y * z) / 32.0
                    - (v * w * x) / 32.0
                    - (v * w * y) / 32.0
                    + (v * w * z) / 32.0
                    + (v * x * y) / 32.0
                    - (v * x * z) / 32.0
                    - (w * x * y) / 32.0
                    - (v * y * z) / 32.0
                    + (w * x * z) / 32.0
                    + (w * y * z) / 32.0
                    - (x * y * z) / 32.0
                    - (v * w * x * y) / 32.0
                    + (v * w * x * z) / 32.0
                    + (v * w * y * z) / 32.0
                    - (v * x * y * z) / 32.0
                    + (w * x * y * z) / 32.0
                    + (v * w * x * y * z) / 32.0
                    + 1.0 / 32.0
                ],
                [
                    v / 32.0
                    - w / 32.0
                    - x / 32.0
                    - y / 32.0
                    + z / 32.0
                    - (v * w) / 32.0
                    - (v * x) / 32.0
                    - (v * y) / 32.0
                    + (w * x) / 32.0
                    + (v * z) / 32.0
                    + (w * y) / 32.0
                    - (w * z) / 32.0
                    + (x * y) / 32.0
                    - (x * z) / 32.0
                    - (y * z) / 32.0
                    + (v * w * x) / 32.0
                    + (v * w * y) / 32.0
                    - (v * w * z) / 32.0
                    + (v * x * y) / 32.0
                    - (v * x * z) / 32.0
                    - (w * x * y) / 32.0
                    - (v * y * z) / 32.0
                    + (w * x * z) / 32.0
                    + (w * y * z) / 32.0
                    + (x * y * z) / 32.0
                    - (v * w * x * y) / 32.0
                    + (v * w * x * z) / 32.0
                    + (v * w * y * z) / 32.0
                    + (v * x * y * z) / 32.0
                    - (w * x * y * z) / 32.0
                    - (v * w * x * y * z) / 32.0
                    + 1.0 / 32.0
                ],
                [
                    v / 32.0
                    - w / 32.0
                    + x / 32.0
                    - y / 32.0
                    + z / 32.0
                    - (v * w) / 32.0
                    + (v * x) / 32.0
                    - (v * y) / 32.0
                    - (w * x) / 32.0
                    + (v * z) / 32.0
                    + (w * y) / 32.0
                    - (w * z) / 32.0
                    - (x * y) / 32.0
                    + (x * z) / 32.0
                    - (y * z) / 32.0
                    - (v * w * x) / 32.0
                    + (v * w * y) / 32.0
                    - (v * w * z) / 32.0
                    - (v * x * y) / 32.0
                    + (v * x * z) / 32.0
                    + (w * x * y) / 32.0
                    - (v * y * z) / 32.0
                    - (w * x * z) / 32.0
                    + (w * y * z) / 32.0
                    - (x * y * z) / 32.0
                    + (v * w * x * y) / 32.0
                    - (v * w * x * z) / 32.0
                    + (v * w * y * z) / 32.0
                    - (v * x * y * z) / 32.0
                    + (w * x * y * z) / 32.0
                    + (v * w * x * y * z) / 32.0
                    + 1.0 / 32.0
                ],
                [
                    v / 32.0
                    - w / 32.0
                    - x / 32.0
                    + y / 32.0
                    + z / 32.0
                    - (v * w) / 32.0
                    - (v * x) / 32.0
                    + (v * y) / 32.0
                    + (w * x) / 32.0
                    + (v * z) / 32.0
                    - (w * y) / 32.0
                    - (w * z) / 32.0
                    - (x * y) / 32.0
                    - (x * z) / 32.0
                    + (y * z) / 32.0
                    + (v * w * x) / 32.0
                    - (v * w * y) / 32.0
                    - (v * w * z) / 32.0
                    - (v * x * y) / 32.0
                    - (v * x * z) / 32.0
                    + (w * x * y) / 32.0
                    + (v * y * z) / 32.0
                    + (w * x * z) / 32.0
                    - (w * y * z) / 32.0
                    - (x * y * z) / 32.0
                    + (v * w * x * y) / 32.0
                    + (v * w * x * z) / 32.0
                    - (v * w * y * z) / 32.0
                    - (v * x * y * z) / 32.0
                    + (w * x * y * z) / 32.0
                    + (v * w * x * y * z) / 32.0
                    + 1.0 / 32.0
                ],
                [
                    v / 32.0
                    - w / 32.0
                    + x / 32.0
                    + y / 32.0
                    + z / 32.0
                    - (v * w) / 32.0
                    + (v * x) / 32.0
                    + (v * y) / 32.0
                    - (w * x) / 32.0
                    + (v * z) / 32.0
                    - (w * y) / 32.0
                    - (w * z) / 32.0
                    + (x * y) / 32.0
                    + (x * z) / 32.0
                    + (y * z) / 32.0
                    - (v * w * x) / 32.0
                    - (v * w * y) / 32.0
                    - (v * w * z) / 32.0
                    + (v * x * y) / 32.0
                    + (v * x * z) / 32.0
                    - (w * x * y) / 32.0
                    + (v * y * z) / 32.0
                    - (w * x * z) / 32.0
                    - (w * y * z) / 32.0
                    + (x * y * z) / 32.0
                    - (v * w * x * y) / 32.0
                    - (v * w * x * z) / 32.0
                    - (v * w * y * z) / 32.0
                    + (v * x * y * z) / 32.0
                    - (w * x * y * z) / 32.0
                    - (v * w * x * y * z) / 32.0
                    + 1.0 / 32.0
                ],
                [
                    v / 32.0
                    + w / 32.0
                    - x / 32.0
                    - y / 32.0
                    - z / 32.0
                    + (v * w) / 32.0
                    - (v * x) / 32.0
                    - (v * y) / 32.0
                    - (w * x) / 32.0
                    - (v * z) / 32.0
                    - (w * y) / 32.0
                    - (w * z) / 32.0
                    + (x * y) / 32.0
                    + (x * z) / 32.0
                    + (y * z) / 32.0
                    - (v * w * x) / 32.0
                    - (v * w * y) / 32.0
                    - (v * w * z) / 32.0
                    + (v * x * y) / 32.0
                    + (v * x * z) / 32.0
                    + (w * x * y) / 32.0
                    + (v * y * z) / 32.0
                    + (w * x * z) / 32.0
                    + (w * y * z) / 32.0
                    - (x * y * z) / 32.0
                    + (v * w * x * y) / 32.0
                    + (v * w * x * z) / 32.0
                    + (v * w * y * z) / 32.0
                    - (v * x * y * z) / 32.0
                    - (w * x * y * z) / 32.0
                    - (v * w * x * y * z) / 32.0
                    + 1.0 / 32.0
                ],
                [
                    v / 32.0
                    + w / 32.0
                    + x / 32.0
                    - y / 32.0
                    - z / 32.0
                    + (v * w) / 32.0
                    + (v * x) / 32.0
                    - (v * y) / 32.0
                    + (w * x) / 32.0
                    - (v * z) / 32.0
                    - (w * y) / 32.0
                    - (w * z) / 32.0
                    - (x * y) / 32.0
                    - (x * z) / 32.0
                    + (y * z) / 32.0
                    + (v * w * x) / 32.0
                    - (v * w * y) / 32.0
                    - (v * w * z) / 32.0
                    - (v * x * y) / 32.0
                    - (v * x * z) / 32.0
                    - (w * x * y) / 32.0
                    + (v * y * z) / 32.0
                    - (w * x * z) / 32.0
                    + (w * y * z) / 32.0
                    + (x * y * z) / 32.0
                    - (v * w * x * y) / 32.0
                    - (v * w * x * z) / 32.0
                    + (v * w * y * z) / 32.0
                    + (v * x * y * z) / 32.0
                    + (w * x * y * z) / 32.0
                    + (v * w * x * y * z) / 32.0
                    + 1.0 / 32.0
                ],
                [
                    v / 32.0
                    + w / 32.0
                    - x / 32.0
                    + y / 32.0
                    - z / 32.0
                    + (v * w) / 32.0
                    - (v * x) / 32.0
                    + (v * y) / 32.0
                    - (w * x) / 32.0
                    - (v * z) / 32.0
                    + (w * y) / 32.0
                    - (w * z) / 32.0
                    - (x * y) / 32.0
                    + (x * z) / 32.0
                    - (y * z) / 32.0
                    - (v * w * x) / 32.0
                    + (v * w * y) / 32.0
                    - (v * w * z) / 32.0
                    - (v * x * y) / 32.0
                    + (v * x * z) / 32.0
                    - (w * x * y) / 32.0
                    - (v * y * z) / 32.0
                    + (w * x * z) / 32.0
                    - (w * y * z) / 32.0
                    + (x * y * z) / 32.0
                    - (v * w * x * y) / 32.0
                    + (v * w * x * z) / 32.0
                    - (v * w * y * z) / 32.0
                    + (v * x * y * z) / 32.0
                    + (w * x * y * z) / 32.0
                    + (v * w * x * y * z) / 32.0
                    + 1.0 / 32.0
                ],
                [
                    v / 32.0
                    + w / 32.0
                    + x / 32.0
                    + y / 32.0
                    - z / 32.0
                    + (v * w) / 32.0
                    + (v * x) / 32.0
                    + (v * y) / 32.0
                    + (w * x) / 32.0
                    - (v * z) / 32.0
                    + (w * y) / 32.0
                    - (w * z) / 32.0
                    + (x * y) / 32.0
                    - (x * z) / 32.0
                    - (y * z) / 32.0
                    + (v * w * x) / 32.0
                    + (v * w * y) / 32.0
                    - (v * w * z) / 32.0
                    + (v * x * y) / 32.0
                    - (v * x * z) / 32.0
                    + (w * x * y) / 32.0
                    - (v * y * z) / 32.0
                    - (w * x * z) / 32.0
                    - (w * y * z) / 32.0
                    - (x * y * z) / 32.0
                    + (v * w * x * y) / 32.0
                    - (v * w * x * z) / 32.0
                    - (v * w * y * z) / 32.0
                    - (v * x * y * z) / 32.0
                    - (w * x * y * z) / 32.0
                    - (v * w * x * y * z) / 32.0
                    + 1.0 / 32.0
                ],
                [
                    v / 32.0
                    + w / 32.0
                    - x / 32.0
                    - y / 32.0
                    + z / 32.0
                    + (v * w) / 32.0
                    - (v * x) / 32.0
                    - (v * y) / 32.0
                    - (w * x) / 32.0
                    + (v * z) / 32.0
                    - (w * y) / 32.0
                    + (w * z) / 32.0
                    + (x * y) / 32.0
                    - (x * z) / 32.0
                    - (y * z) / 32.0
                    - (v * w * x) / 32.0
                    - (v * w * y) / 32.0
                    + (v * w * z) / 32.0
                    + (v * x * y) / 32.0
                    - (v * x * z) / 32.0
                    + (w * x * y) / 32.0
                    - (v * y * z) / 32.0
                    - (w * x * z) / 32.0
                    - (w * y * z) / 32.0
                    + (x * y * z) / 32.0
                    + (v * w * x * y) / 32.0
                    - (v * w * x * z) / 32.0
                    - (v * w * y * z) / 32.0
                    + (v * x * y * z) / 32.0
                    + (w * x * y * z) / 32.0
                    + (v * w * x * y * z) / 32.0
                    + 1.0 / 32.0
                ],
                [
                    v / 32.0
                    + w / 32.0
                    + x / 32.0
                    - y / 32.0
                    + z / 32.0
                    + (v * w) / 32.0
                    + (v * x) / 32.0
                    - (v * y) / 32.0
                    + (w * x) / 32.0
                    + (v * z) / 32.0
                    - (w * y) / 32.0
                    + (w * z) / 32.0
                    - (x * y) / 32.0
                    + (x * z) / 32.0
                    - (y * z) / 32.0
                    + (v * w * x) / 32.0
                    - (v * w * y) / 32.0
                    + (v * w * z) / 32.0
                    - (v * x * y) / 32.0
                    + (v * x * z) / 32.0
                    - (w * x * y) / 32.0
                    - (v * y * z) / 32.0
                    + (w * x * z) / 32.0
                    - (w * y * z) / 32.0
                    - (x * y * z) / 32.0
                    - (v * w * x * y) / 32.0
                    + (v * w * x * z) / 32.0
                    - (v * w * y * z) / 32.0
                    - (v * x * y * z) / 32.0
                    - (w * x * y * z) / 32.0
                    - (v * w * x * y * z) / 32.0
                    + 1.0 / 32.0
                ],
                [
                    v / 32.0
                    + w / 32.0
                    - x / 32.0
                    + y / 32.0
                    + z / 32.0
                    + (v * w) / 32.0
                    - (v * x) / 32.0
                    + (v * y) / 32.0
                    - (w * x) / 32.0
                    + (v * z) / 32.0
                    + (w * y) / 32.0
                    + (w * z) / 32.0
                    - (x * y) / 32.0
                    - (x * z) / 32.0
                    + (y * z) / 32.0
                    - (v * w * x) / 32.0
                    + (v * w * y) / 32.0
                    + (v * w * z) / 32.0
                    - (v * x * y) / 32.0
                    - (v * x * z) / 32.0
                    - (w * x * y) / 32.0
                    + (v * y * z) / 32.0
                    - (w * x * z) / 32.0
                    + (w * y * z) / 32.0
                    - (x * y * z) / 32.0
                    - (v * w * x * y) / 32.0
                    - (v * w * x * z) / 32.0
                    + (v * w * y * z) / 32.0
                    - (v * x * y * z) / 32.0
                    - (w * x * y * z) / 32.0
                    - (v * w * x * y * z) / 32.0
                    + 1.0 / 32.0
                ],
                [
                    v / 32.0
                    + w / 32.0
                    + x / 32.0
                    + y / 32.0
                    + z / 32.0
                    + (v * w) / 32.0
                    + (v * x) / 32.0
                    + (v * y) / 32.0
                    + (w * x) / 32.0
                    + (v * z) / 32.0
                    + (w * y) / 32.0
                    + (w * z) / 32.0
                    + (x * y) / 32.0
                    + (x * z) / 32.0
                    + (y * z) / 32.0
                    + (v * w * x) / 32.0
                    + (v * w * y) / 32.0
                    + (v * w * z) / 32.0
                    + (v * x * y) / 32.0
                    + (v * x * z) / 32.0
                    + (w * x * y) / 32.0
                    + (v * y * z) / 32.0
                    + (w * x * z) / 32.0
                    + (w * y * z) / 32.0
                    + (x * y * z) / 32.0
                    + (v * w * x * y) / 32.0
                    + (v * w * x * z) / 32.0
                    + (v * w * y * z) / 32.0
                    + (v * x * y * z) / 32.0
                    + (w * x * y * z) / 32.0
                    + (v * w * x * y * z) / 32.0
                    + 1.0 / 32.0
                ],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, interpList.shape[0]):
                for m in range(0, interpList.shape[0]):
                  for n in range(0, functionVector.shape[0]):
                    interpMatrix[
                        m
                        + l * interpList.shape[0]
                        + k * interpList.shape[0] * interpList.shape[0]
                        + j
                        * interpList.shape[0]
                        * interpList.shape[0]
                        * interpList.shape[0]
                        + i
                        * interpList.shape[0]
                        * interpList.shape[0]
                        * interpList.shape[0]
                        * interpList.shape[0],
                        n,
                    ] = (
                        functionVector[n]
                        .subs(x, interpList[m])
                        .subs(y, interpList[l])
                        .subs(z, interpList[k])
                        .subs(w, interpList[j])
                        .subs(v, interpList[i])
                    )
      else:
        raise NameError(
            "interpMatrix: Order {} is not supported!\nPolynomial order must be 1 for nodal Serendipity in 5D".format(
                order
            )
        )

    else:
      raise NameError(
          "interpMatrix: Basis {} is not supported!\nSupported basis are currently 'nodal Serendipity', 'modal Serendipity', and 'modal maximal order'".format(
              basis_type
          )
      )

  elif dim == 6:
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")
    w = Symbol("w")
    v = Symbol("v")
    u = Symbol("u")
    if modal and basis_type == "serendipity":
      if order == 0:
        functionVector = Matrix([[0.125]])
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, interpList.shape[0]):
                for m in range(0, interpList.shape[0]):
                  for n in range(0, interpList.shape[0]):
                    for o in range(0, functionVector.shape[0]):
                      interpMatrix[
                          n
                          + m * interpList.shape[0]
                          + l * interpList.shape[0] * interpList.shape[0]
                          + k
                          * interpList.shape[0]
                          * interpList.shape[0]
                          * interpList.shape[0]
                          + j
                          * interpList.shape[0]
                          * interpList.shape[0]
                          * interpList.shape[0]
                          * interpList.shape[0]
                          + i
                          * interpList.shape[0]
                          * interpList.shape[0]
                          * interpList.shape[0]
                          * interpList.shape[0]
                          * interpList.shape[0],
                          o,
                      ] = (
                          functionVector[o]
                          .subs(x, interpList[n])
                          .subs(y, interpList[m])
                          .subs(z, interpList[l])
                          .subs(w, interpList[k])
                          .subs(v, interpList[j])
                          .subs(u, interpList[i])
                      )
      elif order == 1:
        functionVector = Matrix(
            [
                [0.125],
                [0.2165063509461096 * x],
                [0.2165063509461096 * y],
                [0.2165063509461096 * z],
                [0.2165063509461096 * w],
                [0.2165063509461096 * v],
                [0.2165063509461096 * u],
                [0.375 * x * y],
                [0.375 * x * z],
                [0.375 * y * z],
                [0.375 * x * w],
                [0.375 * y * w],
                [0.375 * z * w],
                [0.375 * x * v],
                [0.375 * y * v],
                [0.375 * z * v],
                [0.375 * w * v],
                [0.375 * x * u],
                [0.375 * y * u],
                [0.375 * z * u],
                [0.375 * w * u],
                [0.375 * v * u],
                [0.6495190528383289 * x * y * z],
                [0.6495190528383289 * x * y * w],
                [0.6495190528383289 * x * z * w],
                [0.6495190528383289 * y * z * w],
                [0.6495190528383289 * x * y * v],
                [0.6495190528383289 * x * z * v],
                [0.6495190528383289 * y * z * v],
                [0.6495190528383289 * x * w * v],
                [0.6495190528383289 * y * w * v],
                [0.6495190528383289 * z * w * v],
                [0.6495190528383289 * x * y * u],
                [0.6495190528383289 * x * z * u],
                [0.6495190528383289 * y * z * u],
                [0.6495190528383289 * x * w * u],
                [0.6495190528383289 * y * w * u],
                [0.6495190528383289 * z * w * u],
                [0.6495190528383289 * x * v * u],
                [0.6495190528383289 * y * v * u],
                [0.6495190528383289 * z * v * u],
                [0.6495190528383289 * w * v * u],
                [1.125 * x * y * z * w],
                [1.125 * x * y * z * v],
                [1.125 * x * y * w * v],
                [1.125 * x * z * w * v],
                [1.125 * y * z * w * v],
                [1.125 * x * y * z * u],
                [1.125 * x * y * w * u],
                [1.125 * x * z * w * u],
                [1.125 * y * z * w * u],
                [1.125 * x * y * v * u],
                [1.125 * x * z * v * u],
                [1.125 * y * z * v * u],
                [1.125 * x * w * v * u],
                [1.125 * y * w * v * u],
                [1.125 * z * w * v * u],
                [1.948557158514986 * x * y * z * w * v],
                [1.948557158514986 * x * y * z * w * u],
                [1.948557158514986 * x * y * z * v * u],
                [1.948557158514986 * x * y * w * v * u],
                [1.948557158514986 * x * z * w * v * u],
                [1.948557158514986 * y * z * w * v * u],
                [3.375 * x * y * z * w * v * u],
            ]
        )
        interpMatrix = numpy.zeros(
            (
                interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0]
                * interpList.shape[0],
                functionVector.shape[0],
            )
        )
        for i in range(0, interpList.shape[0]):
          for j in range(0, interpList.shape[0]):
            for k in range(0, interpList.shape[0]):
              for l in range(0, interpList.shape[0]):
                for m in range(0, interpList.shape[0]):
                  for n in range(0, interpList.shape[0]):
                    for o in range(0, functionVector.shape[0]):
                      interpMatrix[
                          n
                          + m * interpList.shape[0]
                          + l * interpList.shape[0] * interpList.shape[0]
                          + k
                          * interpList.shape[0]
                          * interpList.shape[0]
                          * interpList.shape[0]
                          + j
                          * interpList.shape[0]
                          * interpList.shape[0]
                          * interpList.shape[0]
                          * interpList.shape[0]
                          + i
                          * interpList.shape[0]
                          * interpList.shape[0]
                          * interpList.shape[0]
                          * interpList.shape[0]
                          * interpList.shape[0],
                          o,
                      ] = (
                          functionVector[o]
                          .subs(x, interpList[n])
                          .subs(y, interpList[m])
                          .subs(z, interpList[l])
                          .subs(w, interpList[k])
                          .subs(v, interpList[j])
                          .subs(u, interpList[i])
                      )
      else:
        raise NameError(
            "interpMatrix: Order {} is not supported!\nPolynomial order must be 1 for modal Serendipity in 6D".format(
                order
            )
        )

    else:
      raise NameError(
          "interpMatrix: Basis {} is not supported!\nSupported basis are currently 'modal Serendipity' in 6D".format(
              basis_type
          )
      )

  else:
    raise NameError("interpMatrix: Dimension {} is not supported.".format(dim))

  return interpMatrix


if __name__ == "__main__":
  import tables
  # set command line options
  parser = OptionParser()
  parser.add_option(
      "-d", "--dimension", action="store", dest="dim", help="specified dimension"
  )
  parser.add_option(
      "-o", "--order", action="store", dest="order", help="specified polynomial order"
  )
  parser.add_option(
      "-b", "--basis", action="store", dest="basis", help="specified basis set"
  )
  parser.add_option(
      "-i",
      "--interp",
      action="store",
      dest="interp",
      help="specified number of interpolation points",
  )
  parser.add_option(
      "-m",
      "--modal",
      action="store",
      dest="modal",
      help="set to True for modal basis set",
  )

  (options, args) = parser.parse_args()

  dim = int(options.dim)
  order = int(options.order)
  basis_type = options.basis
  modal = options.modal
  interp = int(options.interp)

  interpMatrix = createInterpMatrix(dim, order, basis_type, interp, modal)
  fh = tables.open_file("interpMatrix.h5", mode="w")
  fh.create_array("/", "interpolation_matrix", interpMatrix)
  fh.close()
