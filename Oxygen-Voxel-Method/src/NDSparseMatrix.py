import numpy as np
import scipy.sparse as sp

class NDSparseMatrix:
  def __init__(self, shape : tuple, defaultValue=0):
    self.elements = {}
    self.defaultValue = defaultValue
    self.shape = shape
    self.dim  = len(shape)

    self._count = 0 # Used to iter through the array

  def addValue(self, tuple, value):
    if value==self.defaultValue:
      return
    self.elements[tuple] = value

  def readValue(self, tuple):
    try:
      value = self.elements[tuple]
    except KeyError:
      # could also be 0.0 if using floats...
      value = self.defaultValue
    return value

  def __getitem__(self, item : tuple):
    return self.readValue(item)

  def EmptyMatrix(self):
    self.elements.clear

  @property
  def nonzeros(self):
    return len(self.elements)
  
  @property
  def shape(self):
    return self._shape
  @shape.setter
  def shape(self, shape : tuple):
    assert len(shape)>=1, "Number of dimensions cannot be 0."
    self._shape = shape
    return

  @property
  def dim(self):
    return self._dim
  @dim.setter
  def dim(self, newDim):
    self._dim = int(newDim)
    return

  def __len__(self):
    size = 1
    for d in self.shape:
      size *=d
    return size
  
  def __iter__(self):
    for k in range(self.shape[2]):
      for j in range(self.shape[1]):
        for i in range(self.shape[0]):
          yield str(self.readValue((i,j,k)))
  
class SparseRowIndexer:
    def __init__(self, csr_matrix):
        data = []
        indices = []
        indptr = []

        # Iterating over the rows this way is significantly more efficient
        # than csr_matrix[row_index,:] and csr_matrix.getrow(row_index)
        for row_start, row_end in zip(csr_matrix.indptr[:-1], csr_matrix.indptr[1:]):
             data.append(csr_matrix.data[row_start:row_end])
             indices.append(csr_matrix.indices[row_start:row_end])
             indptr.append(row_end-row_start) # nnz of the row

        self.data = np.array(data)
        self.indices = np.array(indices)
        self.indptr = np.array(indptr)
        self.n_columns = csr_matrix.shape[1]

    def __getitem__(self, row_selector):
        data = np.concatenate(self.data[row_selector])
        indices = np.concatenate(self.indices[row_selector])
        indptr = np.append(0, np.cumsum(self.indptr[row_selector]))

        shape = [indptr.shape[0]-1, self.n_columns]

        return sparse.csr_matrix((data, indices, indptr), shape=shape)
