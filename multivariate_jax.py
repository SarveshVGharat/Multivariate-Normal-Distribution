import jax.numpy as jnp
import jax
from jax import random
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.linalg as spla
import pandas as pd
import os
import traceback

def box_muller(n = 1000, d = 2):
  """ Computing independent normal distributions with 0 mean.
      Parameters:
      n: number of samples, default value is 1000
      d: number of distributions, default value is 2
      Note:
      Values given in floating point will be rounded off to nearest integer
  """
  try:
    n = (d*n)/2
    n = int(n)
    key = random.PRNGKey(42)
    key1 = random.PRNGKey(43)
    u1 = random.uniform(key, shape=(1,n)) 
    u2 = random.uniform(key1, shape=(1,n))
    z1 = jnp.sqrt(-2 * jnp.log(u1)) * jnp.cos(2 * jnp.pi * u2) 
    z2 = jnp.sqrt(-2 * jnp.log(u1)) * jnp.sin(2 * jnp.pi * u2)
    return z1,z2

  except TypeError:
    traceback.print_exc()
    

def cholesky_decomposition(n = 1000, d  = 2, mean = jax.numpy.array([[0], [0]]), covariance = jax.numpy.array([[1, 0], [0, 1]])):
  """ Computing Cholesky Decomposition to extract samples from random distribution
      Parameters:
      n: number of samples, default value is 1000
      d: number of distributions, default value is 2
      mean: mean of the distributions, default value is [0,0]
      covariance: covariance matrix for required distributions, default value is [[1,0],[0,1]]
  """
  try:
    z1, z2 = box_muller(n, d)
    L = jnp.linalg.cholesky(covariance)
    temp = jnp.concatenate((z1,z2)).reshape(n,d)
    Y = L.dot(temp.T) + mean
    return Y 

  except TypeError:
    traceback.print_exc()
  except ValueError:
    traceback.print_exc()
  except np.linalg.LinAlgError:
    traceback.print_exc()
    
    
def plot_mean_cov_diff(n = 1000, d  = 2, mean = jnp.array([[0], [0]]), covariance =jnp.array([[1, 0], [0, 1]])):
  """ Computing and Plotting Mean and Covariance differences. 
      Parameters:
      n: number of samples, default value is 1000
      d: number of distributions, default value is 2
      mean: mean of the distributions, default value is [0,0]
      covariance: covariance matrix for required distributions, default value is [[1,0],[0,1]]  
  """
  try:
    Y = cholesky_decomposition(n, d, mean, covariance)
    mean_diff = jnp.mean(Y.T, axis=0) - mean.T
    cov_diff = jnp.cov(Y) - covariance

    plt.title("Difference between Mean")
    sns.heatmap(mean_diff, yticklabels=False, xticklabels=False)    
    plt.show()

    plt.title("Difference between Covariance Matrix")
    sns.heatmap(cov_diff, yticklabels=False, xticklabels=False)
    plt.show()

  except AttributeError:
    traceback.print_exc()
  except TypeError:
    traceback.print_exc()
  except ValueError:
    traceback.print_exc()
  except np.linalg.LinAlgError:
    traceback.print_exc()
    
def mean_cov_diff(n = 1000, d  = 2, mean = np.array([[0], [0]]), covariance = np.array([[1, 0], [0, 1]])):
  """ Computing and Returning Mean and Covariance differences. 
      Parameters:
      n: number of samples, default value is 1000
      d: number of distributions, default value is 2
      mean: mean of the distributions, default value is [0,0]
      covariance: covariance matrix for required distributions, default value is [[1,0],[0,1]]  
  """
  try:
    Y = cholesky_decomposition(n, d, mean, covariance)
    mean_diff = jnp.mean(Y.T, axis=0) - mean.T
    cov_diff = jnp.cov(Y) - covariance
    return mean_diff, cov_diff

  except AttributeError:
    traceback.print_exc()
  except TypeError:
    traceback.print_exc()
  except ValueError:
    traceback.print_exc()
  except np.linalg.LinAlgError:
    traceback.print_exc()

def plot_bivariate(n = 10000,  mean = jnp.array([[0], [0]]), covariance = jnp.array([[1, 0], [0, 1]])):
  """Computing and Plotting Bivariate distribution
     Parameters:
     n: number of samples, default value is 1000
     mean: mean of the distributions, default value is [0,0]
     covariance: covariance matrix for required distributions, default value is [[1,0],[0,1]]  
  """
  try:
    Y = cholesky_decomposition(n, 2, mean, covariance)
    Y = np.array(Y)
    me = jnp.array(np.mean(Y.T, axis=0))
    cov = jnp.cov(Y) 
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    temp1 = (eigenvectors[0,0]**2 + eigenvectors[0,1]**2)**0.5
    temp2 = (eigenvectors[1,0]**2 + eigenvectors[1,1]**2)**0.5
    d1 = eigenvalues[0]**0.5 
    d2 = eigenvalues[1]**0.5
    m1 = eigenvectors[0,1]/eigenvectors[0,0]
    m2 = eigenvectors[1,1]/eigenvectors[1,0]
    x1 = d1*(eigenvectors[0,1]/temp1)
    y1 = d1*(eigenvectors[0,0]/temp1)
    x2 = d2*(eigenvectors[1,1]/temp2)
    y2 = d2*(eigenvectors[1,0]/temp2)

    plt.arrow(me[0], me[1], x1, y1, head_width = 0.2, width = 0.01, ec = 'Black')
    plt.arrow(me[0], me[1], x2, y2, head_width = 0.2,  width = 0.01, ec = 'Black')
    plt.scatter([Y[0,:]], [Y[1,:]], color = 'Grey', s = 0.25)
    plt.grid(linestyle = '--', linewidth = 0.5)  
    plt.show()

  except AttributeError:
    traceback.print_exc()
  except TypeError:
    traceback.print_exc()
  except ValueError:
    traceback.print_exc()
  except np.linalg.LinAlgError:
    traceback.print_exc()