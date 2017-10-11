import hmmlearn.hmm as hm
import numpy as np

hmm = hm.MultinomialHMM(n_components=3, verbose=True, n_iter=1000)
# hmm = hm.GaussianHMM(n_components=4, verbose=True, n_iter=1000)

X1 = np.array([[1],[0],[0],[1]])
X2 = np.array([[2],[1],[2],[1],[0],[1]])

# X1 = np.random.randint(3, size=(10, 2))
# X2 = np.random.randint(3, size=(5, 2))

print(X2)

X = np.concatenate([X1, X2])
lengthsX = np.array([len(X1), len(X2)])

model = hmm.fit(X, lengths=lengthsX)

print('X.shape = ' + str(X.shape))


print('n_features: ' + str(hmm.n_features))

hidden_states = model.predict(X)
print(hidden_states)

print(model.startprob_)

samples,states = model.sample(10)
print(samples)
print(states)


