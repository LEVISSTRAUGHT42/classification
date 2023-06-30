#make_blobs permet de creer ou simuler des données
X,y = make_blobs(n_samples=100, centers=3,n_features=2, random_state=0)
print(X.shape)
#affichage des clusters
X,y = make_blobs(n_samples=100, centers=3)
plt.scatter(X[:,0],X[:,1])
plt.show()

model = KMeans(n_clusters=3)#3 regroupements
model.fit(X) #apprentissage sur les données du vecteur X
model.labels_ #affichage des variables dans la matrice

#prediction
model = KMeans(n_clusters=3)#3 regroupements
model.fit(X)
model.predict(X)
plt.scatter(X[:,0],X[:,1], c=model.predict(X))
plt.show()

#Prediction et affichage des centroides
model = KMeans(n_clusters=3)#3 regroupements
model.fit(X)
model.predict(X)
plt.scatter(X[:,0],X[:,1], c=model.predict(X))
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1],c='r')#afficher les centroides
plt.show()

#evaluation
model.inertia_
model.score(X)

#Determiner le nombre maximal de cluster
inertia = [] #pour determiner le nombre de centroides
k_range = range(1, 20)
for k in k_range:
    model=KMeans(n_clusters=k).fit(X)
    inertia.append(model.inertia_)
plt.plot(k_range, inertia) # ici k clusters
plt.xlabel('nombre clusters')
plt.ylabel('cout du model(inertia)')
