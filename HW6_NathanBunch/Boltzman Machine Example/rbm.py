from __future__ import print_function
import numpy as np

class RBM:
  
  def __init__(self, num_visible, num_hidden):
    self.num_hidden = num_hidden
    self.num_visible = num_visible
    self.debug_print = True

    # Initialize a weight matrix, of dimensions (num_visible x num_hidden), using
    # a uniform distribution between -sqrt(6. / (num_hidden + num_visible))
    # and sqrt(6. / (num_hidden + num_visible)). One could vary the 
    # standard deviation by multiplying the interval with appropriate value.
    # Here we initialize the weights with mean 0 and standard deviation 0.1. 
    # Reference: Understanding the difficulty of training deep feedforward 
    # neural networks by Xavier Glorot and Yoshua Bengio
    np_rng = np.random.RandomState(1234)

    self.weights = np.asarray(np_rng.uniform(
			low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
      high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                       	size=(num_visible, num_hidden)))

    print(self.weights)
    # Insert weights for the bias units into the first row and first column.
    self.weights = np.insert(self.weights, 0, 0, axis = 0)
    self.weights = np.insert(self.weights, 0, 0, axis = 1)
    print(self.weights)

  def train(self, data, max_epochs = 1000, learning_rate = 0.1):
    """
    Train the machine.

    Parameters
    ----------
    data: A matrix where each row is a training example consisting of the states of visible units.    
    """

    num_examples = data.shape[0]
    #print(data)
    #print(num_examples)

    # Insert bias units of 1 into the first column.
    # here is when the intercept is added
    # y = ax + b*1
    # that is why 1 is added, since it describes the bias
    data = np.insert(data, 0, 1, axis = 1) # intercept like sciklit learn is added into the weights
    #print(data)

    for epoch in range(max_epochs):      
      # Clamp to the data and sample from the hidden units. 
      # (This is the "positive CD phase", aka the reality phase.)
      #print("DATA")
      #print(data)
      pos_hidden_activations = np.dot(data, self.weights)
      #print("POS HIDDEN ACTIVATIONS")
      #print(pos_hidden_activations)
      #print(pos_hidden_activations)
      pos_hidden_probs = self._logistic(pos_hidden_activations)
      #print("POS HIDDEN PROBS")
      #print(pos_hidden_probs)
      #print(pos_hidden_probs)
      pos_hidden_probs[:,0] = 1 # Fix the bias unit.
      #print("POS HIDDEN PROBS FIX")
      #print(pos_hidden_probs)
      #print(pos_hidden_probs)
      pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
      #print("POS HIDDEN STATES")
      #print(pos_hidden_states)
      #print(">>>>>>>>>>>>>>>>>>>>>>")
      #print(pos_hidden_states)
      # Note that we're using the activation *probabilities* of the hidden states, not the hidden states       
      # themselves, when computing associations. We could also use the states; see section 3 of Hinton's 
      # "A Practical Guide to Training Restricted Boltzmann Machines" for more.
      pos_associations = np.dot(data.T, pos_hidden_probs)
      #print("POS ASSOCIATIONS")
      #print(pos_associations)
      #print(pos_associations)

      # Reconstruct the visible units and sample again from the hidden units.
      # (This is the "negative CD phase", aka the daydreaming phase.)
      neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
     # print("NEG VISIBLE ACTIVATIONS")
      #print(neg_visible_activations)
      #print(neg_visible_activations)
      neg_visible_probs = self._logistic(neg_visible_activations)
     # print("NEG VISIBLE PROBS LOGISTICS")
      #print(neg_visible_probs)
      #print(neg_visible_probs)
      neg_visible_probs[:,0] = 1 # Fix the bias unit.
     # print("NEG VISIBLE PROBS FIX")
     # print(neg_visible_probs)
      #print(neg_visible_probs)
      neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
     # print("NEG HIDDEN ACTIVATIONS")
     # print(neg_hidden_activations)
      #print(neg_hidden_activations)
      neg_hidden_probs = self._logistic(neg_hidden_activations)
     # print("NEG HIDDEN PROBS LOGISTICS")
     # print(neg_hidden_probs)
      #print(neg_hidden_probs)
      # Note, again, that we're using the activation *probabilities* when computing associations, not the states 
      # themselves.
      neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)
     # print("NEG ASSOCIATIONS")
     # print(neg_associations)
      #print(neg_associations)

      # Update weights.
      self.weights += learning_rate * ((pos_associations - neg_associations) / num_examples)
     # print("WEIGHT UPDATE")
     # print(self.weights)
      #print(self.weights)

      error = np.sum((data - neg_visible_probs) ** 2)
      #print(error)
      if self.debug_print:
        print("Epoch %s: error is %s" % (epoch, error))

  def run_visible(self, data):
    """
    Assuming the RBM has been trained (so that weights for the network have been learned),
    run the network on a set of visible units, to get a sample of the hidden units.
    
    Parameters
    ----------
    data: A matrix where each row consists of the states of the visible units.
    
    Returns
    -------
    hidden_states: A matrix where each row consists of the hidden units activated from the visible
    units in the data matrix passed in.
    """
    
    num_examples = data.shape[0]
    
    # Create a matrix, where each row is to be the hidden units (plus a bias unit)
    # sampled from a training example.
    hidden_states = np.ones((num_examples, self.num_hidden + 1))
    
    # Insert bias units of 1 into the first column of data.
    data = np.insert(data, 0, 1, axis = 1)

    # Calculate the activations of the hidden units.
    hidden_activations = np.dot(data, self.weights)
    # Calculate the probabilities of turning the hidden units on.
    hidden_probs = self._logistic(hidden_activations)
    # Turn the hidden units on with their specified probabilities.
    hidden_states[:,:] = hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
    # Always fix the bias unit to 1.
    # hidden_states[:,0] = 1
  
    # Ignore the bias units.
    hidden_states = hidden_states[:,1:]
    return hidden_states
    
  # TODO: Remove the code duplication between this method and `run_visible`?
  def run_hidden(self, data):
    """
    Assuming the RBM has been trained (so that weights for the network have been learned),
    run the network on a set of hidden units, to get a sample of the visible units.

    Parameters
    ----------
    data: A matrix where each row consists of the states of the hidden units.

    Returns
    -------
    visible_states: A matrix where each row consists of the visible units activated from the hidden
    units in the data matrix passed in.
    """

    num_examples = data.shape[0]

    # Create a matrix, where each row is to be the visible units (plus a bias unit)
    # sampled from a training example.
    visible_states = np.ones((num_examples, self.num_visible + 1))

    # Insert bias units of 1 into the first column of data.
    data = np.insert(data, 0, 1, axis = 1)

    # Calculate the activations of the visible units.
    visible_activations = np.dot(data, self.weights.T)
    # Calculate the probabilities of turning the visible units on.
    visible_probs = self._logistic(visible_activations)
    # Turn the visible units on with their specified probabilities.
    visible_states[:,:] = visible_probs > np.random.rand(num_examples, self.num_visible + 1)
    # Always fix the bias unit to 1.
    # visible_states[:,0] = 1
    
    # Ignore the bias units.
    visible_states = visible_states[:,1:]
    return visible_states
    
  def daydream(self, num_samples):
    """
    Randomly initialize the visible units once, and start running alternating Gibbs sampling steps
    (where each step consists of updating all the hidden units, and then updating all of the visible units),
    taking a sample of the visible units at each step.
    Note that we only initialize the network *once*, so these samples are correlated.

    Returns
    -------
    samples: A matrix, where each row is a sample of the visible units produced while the network was
    daydreaming.
    """

    # Create a matrix, where each row is to be a sample of of the visible units 
    # (with an extra bias unit), initialized to all ones.
    samples = np.ones((num_samples, self.num_visible + 1))

    # Take the first sample from a uniform distribution.
    samples[0,1:] = np.random.rand(self.num_visible)

    # Start the alternating Gibbs sampling.
    # Note that we keep the hidden units binary states, but leave the
    # visible units as real probabilities. See section 3 of Hinton's
    # "A Practical Guide to Training Restricted Boltzmann Machines"
    # for more on why.
    for i in range(1, num_samples):
      visible = samples[i-1,:]

      # Calculate the activations of the hidden units.
      hidden_activations = np.dot(visible, self.weights)      
      # Calculate the probabilities of turning the hidden units on.
      hidden_probs = self._logistic(hidden_activations)
      # Turn the hidden units on with their specified probabilities.
      hidden_states = hidden_probs > np.random.rand(self.num_hidden + 1)
      # Always fix the bias unit to 1.
      hidden_states[0] = 1

      # Recalculate the probabilities that the visible units are on.
      visible_activations = np.dot(hidden_states, self.weights.T)
      visible_probs = self._logistic(visible_activations)
      visible_states = visible_probs > np.random.rand(self.num_visible + 1)
      samples[i,:] = visible_states

    # Ignore the bias units (the first column), since they're always set to 1.
    return samples[:,1:]        
      
  def _logistic(self, x):
    return 1.0 / (1 + np.exp(-x))
  
  # Nathan's definitions that have been added
  # weights in the sciklit learn is known as components and here it is weights
  # create the collowing:

  #def fit(self, data, y=None):
  #  #fix the way the weights are generated....its incorrect size
  #  n_samples = data.shape[0]
  # vis = data.shape[1]
  #  hid = data.shape[0] * data.shape[1]
  #  np_rng = np.random.RandomState(1234)
  #  self.weights = np.asarray(np_rng.uniform(low=-0.1 * np.sqrt(6. / (hid + vis)), high=0.1 * np.sqrt(6. / (hid + vis)),size=(vis, hid)))
  #  self.weights = np.insert(self.weights, 0, 0, axis = 0)
  #  self.weights = np.insert(self.weights, 0, 0, axis = 1)

  # n_batches = int(np.ceil(float(n_samples) / 4))
  #batch_slices = list(gen_even_slices(n_batches * 4,n_batches, n_samples))
  #  batch_slices = list(data[0:n_batches*4:n_batches])
  #  for iteration in range(1, 5000 + 1):
  #          for batch_slice in batch_slices:
  #              #self._fit(X[batch_slice])
  #              self._fit(data[batch_slice])
  #  print("Iteration %d, pseudo-likelihood = %.2f" % (iteration, self.score_samples(X).mean()))

  #idk why but this is an error
  #def _fit(self, data):
    #h_pos = self._mean_hiddens(v_pos)
    #    v_neg = self._sample_visibles(self.h_samples_, rng)
    #    h_neg = self._mean_hiddens(v_neg)

    #    lr = float(self.learning_rate) / v_pos.shape[0]
    #    update = safe_sparse_dot(v_pos.T, h_pos, dense_output=True).T
    #    update -= np.dot(h_neg.T, v_neg)
    #    self.components_ += lr * update
    #    self.intercept_hidden_ += lr * (h_pos.sum(axis=0) - h_neg.sum(axis=0))
    #    self.intercept_visible_ += lr * (np.asarray(
    #                                     v_pos.sum(axis=0)).squeeze() -
    #                                     v_neg.sum(axis=0))

    #    h_neg[rng.uniform(size=h_neg.shape) < h_neg] = 1.0  # sample binomial
    #    self.h_samples_ = np.floor(h_neg, h_neg)
    #pos_hidden_activations = np.dot(data, self.weights)      
    #pos_hidden_probs = self._logistic(pos_hidden_activations)
    #pos_hidden_probs[:,0] = 1 # Fix the bias unit.
    #pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
    # Note that we're using the activation *probabilities* of the hidden states, not the hidden states       
    # themselves, when computing associations. We could also use the states; see section 3 of Hinton's 
    # "A Practical Guide to Training Restricted Boltzmann Machines" for more.
    #pos_associations = np.dot(data.T, pos_hidden_probs)

    # Reconstruct the visible units and sample again from the hidden units.
    # (This is the "negative CD phase", aka the daydreaming phase.)
    #neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
    #neg_visible_probs = self._logistic(neg_visible_activations)
    #neg_visible_probs[:,0] = 1 # Fix the bias unit.
    #neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
    #neg_hidden_probs = self._logistic(neg_hidden_activations)
    # Note, again, that we're using the activation *probabilities* when computing associations, not the states 
    # themselves.
    #neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

    # Update weights.
    #self.weights += learning_rate * ((pos_associations - neg_associations) / num_examples)

    #error = np.sum((data - neg_visible_probs) ** 2)
    #if self.debug_print:
    #  print("Fit Iteration %s: error is %s" % (epoch, error))
  #def score_samples():

if __name__ == '__main__':
  r = RBM(num_visible = 6, num_hidden = 2)
  training_data = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]])
  #r.train(training_data, max_epochs = 5000) #default
  r.train(training_data, max_epochs = 5000)
  #r.fit(training_data)
  print("Weights:")
  print(r.weights)
  print()
  user = np.array([[0,0,0,1,1,0]])
  print("Get the category preference:")
  print(r.run_visible(user))
  print()
  userPref = np.array([[1,0]])
  print("Get the movie preference:")
  print(r.run_hidden(userPref))
  print()
  print("Daydream for (5):")
  movieWatch = r.daydream(5)
  print(movieWatch)
  #print(movieWatch[:,3])
  #print("Get daydream category preferences:")
  #print(r.run_visible(np.array(movieWatch[:,2])))